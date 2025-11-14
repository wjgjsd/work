"""
ImportanceSpatialTransformer 학습 스크립트
- SD UNet + ControlNet 가중치 로드
- attn_importance만 학습
- 최종 통합 가중치 저장
"""

import os
from argparse import ArgumentParser
import copy
from pathlib import Path
from diffbir.importance_map_net import SharedEncoderImportanceNet
from peft import LoraConfig, inject_adapter_in_model, TaskType

from omegaconf import OmegaConf
import torch
import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn import functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler


def main(args) -> None:
    # ========== Setup ==========
    accelerator = Accelerator(split_batches=True, mixed_precision='fp16')
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup experiment folder
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # ========== Load ControlLDM with ImportanceSpatialTransformer ==========
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Loading models...")
        print("="*60)
    
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    
    # ========== Load SD UNet weights ==========
    if accelerator.is_main_process:
        print(f"\n① Loading SD checkpoint from: {cfg.train.sd_path}")
    
    sd_checkpoint = torch.load(cfg.train.sd_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd_checkpoint:
        sd = sd_checkpoint["state_dict"]
    else:
        sd = sd_checkpoint
    
    unused, missing = cldm.load_pretrained_sd(sd)
    
    if accelerator.is_main_process:
        print(f"   ✅ Loaded SD UNet weights")
        if unused:
            print(f"   ⚠️  Unused SD weights: {len(unused)} keys")
        if missing:
            print(f"   ⚠️  Missing SD weights: {len(missing)} keys")
            # attn_importance 관련은 정상 (새로 추가된 거니까)
            missing_importance = [k for k in missing if 'attn_importance' in k]
            missing_other = [k for k in missing if 'attn_importance' not in k]
            print(f"      - attn_importance (expected): {len(missing_importance)}")
            print(f"      - others (unexpected): {len(missing_other)}")
            if missing_other:
                print(f"      ⚠️  Warning: {missing_other[:5]}...")

    # ========== Load ControlNet weights ==========
    if accelerator.is_main_process:
        print(f"\n② Loading ControlNet checkpoint")
    
    if cfg.train.resume:
        # 이전 학습 재개
        control_checkpoint = torch.load(cfg.train.resume, map_location="cpu", weights_only=False)
        cldm.load_controlnet_from_ckpt(control_checkpoint)
        if accelerator.is_main_process:
            print(f"   ✅ Resumed from: {cfg.train.resume}")
    
    elif hasattr(cfg.train, 'controlnet_path') and cfg.train.controlnet_path:
        # ControlNet 가중치 로드
        control_checkpoint = torch.load(
            cfg.train.controlnet_path, 
            map_location="cpu", 
            weights_only=False
        )
        
        # state_dict 추출
        if isinstance(control_checkpoint, dict) and 'state_dict' in control_checkpoint:
            control_sd = control_checkpoint['state_dict']
        else:
            control_sd = control_checkpoint
        
        # ControlNet에 로드
        missing_ctrl, unexpected_ctrl = cldm.load_controlnet_from_ckpt(control_sd, strict=False)
        
        if accelerator.is_main_process:
            print(f"   ✅ Loaded ControlNet from: {cfg.train.controlnet_path}")
            if missing_ctrl:
                missing_importance = [k for k in missing_ctrl if 'attn_importance' in k]
                missing_other = [k for k in missing_ctrl if 'attn_importance' not in k]
                print(f"      - Missing attn_importance (expected): {len(missing_importance)}")
                if missing_other:
                    print(f"      ⚠️  Missing others: {len(missing_other)}")
    
    else:
        # UNet에서 초기화
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(f"   ✅ Initialized ControlNet from UNet")
            print(f"      - New zero conv: {len(init_with_new_zero)}")
            print(f"      - From scratch: {len(init_with_scratch)}")
    
    # ========== Load Cleaner (SwinIR) - FROZEN ==========
    if accelerator.is_main_process:
        print(f"\n③ Loading SwinIR (cleaner)")
    
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir_weight = torch.load(cfg.train.swinir_path, map_location="cpu", weights_only=False)
    if "state_dict" in swinir_weight:
        swinir_weight = swinir_weight["state_dict"]
    swinir_weight = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in swinir_weight.items()
    }
    swinir.load_state_dict(swinir_weight, strict=True)
    swinir.eval().to(device)
    for p in swinir.parameters():
        p.requires_grad = False
    
    if accelerator.is_main_process:
        print(f"   ✅ Loaded frozen SwinIR")

    # ========== Load Importance Net - FROZEN ==========
    if accelerator.is_main_process:
        print(f"\n④ Loading Importance Network")
    
    importance_net = SharedEncoderImportanceNet(in_channels=3)
    importance_net.load_state_dict(
        torch.load(cfg.train.importance_net_path, map_location=device, weights_only=False)
    )
    importance_net.to(device)
    importance_net.eval()
    for param in importance_net.parameters():
        param.requires_grad = False
    
    if accelerator.is_main_process:
        print(f"   ✅ Loaded frozen Importance Network")
    
    # ========== Load Diffusion ==========
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    # ========== Freeze All & Make attn_importance Trainable ==========
    if accelerator.is_main_process:
        print(f"\n⑤ Setting up trainable parameters")
    
    for param in cldm.parameters():
        param.requires_grad = False
    
    trainable_params = []
    attn_importance_count = 0
    unet_importance_count = 0
    controlnet_importance_count = 0
    
    for module_name, module in cldm.named_modules():
        module_type = type(module).__name__
        
        if 'ImportanceBasicTransformerBlock' in module_type:
            attn_importance_count += 1
            
            if hasattr(module, 'attn_importance'):
                # attn_importance 파라미터 trainable
                for param in module.attn_importance.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                
                # norm_importance도 trainable
                if hasattr(module, 'norm_importance'):
                    for param in module.norm_importance.parameters():
                        param.requires_grad = True
                        trainable_params.append(param)
                
                # Count per network
                if 'unet.' in module_name:
                    unet_importance_count += 1
                elif 'controlnet.' in module_name:
                    controlnet_importance_count += 1

    # Summary
    trainable_count = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in cldm.parameters())

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Setup Summary")
        print(f"{'='*60}")
        print(f"ImportanceBasicTransformerBlock modules:")
        print(f"   - UNet: {unet_importance_count}")
        print(f"   - ControlNet: {controlnet_importance_count}")
        print(f"   - Total: {attn_importance_count}")
        print(f"\nParameters:")
        print(f"   - Total: {total_params:,}")
        print(f"   - Trainable (attn_importance): {trainable_count:,} ({trainable_count/total_params*100:.4f}%)")
        print(f"   - Frozen: {total_params - trainable_count:,} ({(total_params - trainable_count)/total_params*100:.2f}%)")
        print(f"{'='*60}\n")

    if len(trainable_params) == 0:
        raise ValueError("❌ No trainable parameters found!")

    # ========== Optimizer ==========
    opt = torch.optim.AdamW(trainable_params, lr=cfg.train.learning_rate)

    # ========== Data Setup ==========
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training
    cldm.train().to(device)
    swinir.eval()
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Training variables
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    # ========== Training Loop ==========
    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt = batch 
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # Crop for memory efficiency
            TARGET_SIZE = 1024
            b_size, _, h_init, w_init = gt.size()
            
            if h_init >= TARGET_SIZE and w_init >= TARGET_SIZE:
                h_c, w_c = gt.shape[2], gt.shape[3]
                rand_h = random.randint(0, h_c - TARGET_SIZE)
                rand_w = random.randint(0, w_c - TARGET_SIZE)
                gt = gt[:, :, rand_h:rand_h + TARGET_SIZE, rand_w:rand_w + TARGET_SIZE]
                
                SCALE_FACTOR = h_init // lq.shape[2] 
                TARGET_LQ_SIZE = TARGET_SIZE // SCALE_FACTOR
                rand_h_lq = rand_h // SCALE_FACTOR 
                rand_w_lq = rand_w // SCALE_FACTOR
                lq = lq[:, :, rand_h_lq:rand_h_lq + TARGET_LQ_SIZE, rand_w_lq:rand_w_lq + TARGET_LQ_SIZE]
            
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                
                if clean.shape[2] != gt.shape[2] or clean.shape[3] != gt.shape[3]:
                    clean = F.interpolate(
                        clean, size=gt.shape[2:],
                        mode='bicubic', align_corners=False
                    )
                
                lq_upsampled = F.interpolate(
                    lq, size=clean.shape[2:],
                    mode='bicubic', align_corners=False
                )
                importance_map = importance_net(clean, lq_upsampled)
                
                cond = pure_cldm.prepare_condition(clean, prompt)
                
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
                
                importance_map_latent = F.interpolate(
                    importance_map, size=z_0.shape[2:],
                    mode='bilinear', align_corners=False
                )

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug, importance_map_latent)
            
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint (intermediate - attn_importance only)
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # 중간 저장: attn_importance만
                    adapter_state_dict = {}
                    for name, param in pure_cldm.named_parameters():
                        if 'attn_importance' in name and param.requires_grad:
                            adapter_state_dict[name] = param.cpu().clone()

                    ckpt_path = f"{ckpt_dir}/{global_step:07d}_attn_importance_only.pt"
                    torch.save(adapter_state_dict, ckpt_path)
                    print(f"   💾 Saved intermediate checkpoint: {ckpt_path}")

            # Log images
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_clean = clean[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            ("image/condition", log_clean),
                            ("image/condition_decoded", (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2),
                            ("image/condition_aug_decoded", (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2),
                            ("image/prompt", (log_txt_as_img((512, 512), log_prompt) + 1) / 2),
                            ("image/importance_map", importance_map[:N]),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1

    # ========== Save Final Integrated Checkpoint ==========
    if accelerator.is_main_process:
        save_final_checkpoint(pure_cldm, cfg, exp_dir)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
        print("Training completed!")


def save_final_checkpoint(cldm, cfg, exp_dir):
    """
    최종 가중치를 2개 파일로 저장:
    1. SD UNet with importance (for inference/finetuning)
    2. ControlNet with importance (for ControlLDM)
    """
    print("\n" + "="*80)
    print("Saving FINAL CHECKPOINTS (2 files)")
    print("="*80)
    
    # ========== Step 1: Load Original SD Checkpoint ==========
    print("\n① Loading original SD checkpoint...")
    sd_checkpoint = torch.load(cfg.train.sd_path, map_location="cpu", weights_only=False)
    
    if 'state_dict' in sd_checkpoint:
        sd_state = sd_checkpoint['state_dict']
    else:
        sd_state = sd_checkpoint
    
    print(f"   ✅ Loaded: {cfg.train.sd_path}")
    print(f"   Total keys: {len(sd_state)}")
    
    # ========== Step 2: Extract Current Model State ==========
    print("\n② Extracting trained weights...")
    
    current_state = cldm.state_dict()
    
    # Separate by network
    unet_weights = {}
    controlnet_weights = {}
    
    for key, value in current_state.items():
        if key.startswith('unet.'):
            unet_weights[key] = value.cpu()
        elif key.startswith('controlnet.'):
            controlnet_weights[key] = value.cpu()
    
    print(f"   - UNet weights: {len(unet_weights)}")
    print(f"   - ControlNet weights: {len(controlnet_weights)}")
    
    # Count attn_importance
    unet_importance = [k for k in unet_weights.keys() if 'attn_importance' in k]
    ctrl_importance = [k for k in controlnet_weights.keys() if 'attn_importance' in k]
    
    print(f"\n   attn_importance parameters:")
    print(f"   - UNet: {len(unet_importance)}")
    print(f"   - ControlNet: {len(ctrl_importance)}")
    
    # ========== Step 3: Build SD Checkpoint (UNet + VAE + CLIP + importance) ==========
    print("\n③ Building SD checkpoint with importance...")
    
    sd_final = {}
    
    # Add original SD components (VAE, CLIP, etc.)
    for key, value in sd_state.items():
        if key.startswith('first_stage_model.') or \
           key.startswith('cond_stage_model.'):
            sd_final[key] = value
    
    # Add UNet weights (with attn_importance)
    for key, value in unet_weights.items():
        # unet.* → model.diffusion_model.*
        new_key = key.replace('unet.', 'model.diffusion_model.')
        sd_final[new_key] = value
    
    # Count what we added
    unet_base = [k for k in sd_final.keys() if k.startswith('model.diffusion_model.') and 'attn_importance' not in k]
    unet_imp_added = [k for k in sd_final.keys() if k.startswith('model.diffusion_model.') and 'attn_importance' in k]
    vae_keys = [k for k in sd_final.keys() if k.startswith('first_stage_model.')]
    clip_keys = [k for k in sd_final.keys() if k.startswith('cond_stage_model.')]
    
    print(f"\n   SD Checkpoint composition:")
    print(f"   - UNet (base): {len(unet_base)}")
    print(f"   - UNet attn_importance (NEW): {len(unet_imp_added)}")
    print(f"   - VAE: {len(vae_keys)}")
    print(f"   - CLIP: {len(clip_keys)}")
    print(f"   ────────────────────────")
    print(f"   Total: {len(sd_final)}")
    
    # ========== Step 4: Build ControlNet Checkpoint (with importance) ==========
    print("\n④ Building ControlNet checkpoint with importance...")
    
    controlnet_final = {}
    
    # Add ControlNet weights (with attn_importance)
    for key, value in controlnet_weights.items():
        # controlnet.* → keep as is or remove prefix
        # DiffBIR v2.pth는 보통 prefix 없이 저장됨
        new_key = key.replace('controlnet.', '')
        controlnet_final[new_key] = value
    
    ctrl_base = [k for k in controlnet_final.keys() if 'attn_importance' not in k]
    ctrl_imp_added = [k for k in controlnet_final.keys() if 'attn_importance' in k]
    
    print(f"\n   ControlNet Checkpoint composition:")
    print(f"   - ControlNet (base): {len(ctrl_base)}")
    print(f"   - ControlNet attn_importance (NEW): {len(ctrl_imp_added)}")
    print(f"   ────────────────────────")
    print(f"   Total: {len(controlnet_final)}")
    
    # ========== Step 5: Save SD Checkpoint ==========
    print("\n⑤ Saving SD checkpoint...")
    
    sd_output_path = Path(exp_dir) / "sd_v2_with_importance.ckpt"
    
    sd_checkpoint_to_save = {
        'state_dict': sd_final,
        'training_info': {
            'original_sd': cfg.train.sd_path,
            'unet_attn_importance_modules': len([k for k in unet_imp_added if 'to_q.weight' in k]),
            'description': 'Stable Diffusion v2.1 with ImportanceSpatialTransformer in UNet',
        }
    }
    
    torch.save(sd_checkpoint_to_save, sd_output_path)
    sd_size_gb = sd_output_path.stat().st_size / (1024**3)
    
    print(f"   ✅ Saved SD checkpoint: {sd_output_path}")
    print(f"   File size: {sd_size_gb:.2f} GB")
    
    # ========== Step 6: Save ControlNet Checkpoint ==========
    print("\n⑥ Saving ControlNet checkpoint...")
    
    ctrl_output_path = Path(exp_dir) / "controlnet_v2_with_importance.pth"
    
    ctrl_checkpoint_to_save = {
        'state_dict': controlnet_final,
        'training_info': {
            'original_controlnet': cfg.train.get('controlnet_path', 'initialized_from_unet'),
            'controlnet_attn_importance_modules': len([k for k in ctrl_imp_added if 'to_q.weight' in k]),
            'description': 'ControlNet with ImportanceSpatialTransformer',
        }
    }
    
    torch.save(ctrl_checkpoint_to_save, ctrl_output_path)
    ctrl_size_gb = ctrl_output_path.stat().st_size / (1024**3)
    
    print(f"   ✅ Saved ControlNet checkpoint: {ctrl_output_path}")
    print(f"   File size: {ctrl_size_gb:.2f} GB")
    
    # ========== Step 7: Verification ==========
    print("\n⑦ Verifying checkpoints...")
    
    # Verify SD
    verify_sd = torch.load(sd_output_path, map_location="cpu", weights_only=False)
    verify_sd_state = verify_sd['state_dict']
    
    has_unet = any(k.startswith('model.diffusion_model.') for k in verify_sd_state.keys())
    has_vae = any(k.startswith('first_stage_model.') for k in verify_sd_state.keys())
    has_clip = any(k.startswith('cond_stage_model.') for k in verify_sd_state.keys())
    has_unet_importance = any('model.diffusion_model.' in k and 'attn_importance' in k for k in verify_sd_state.keys())
    
    print(f"\n   SD Checkpoint:")
    print(f"   {'✅' if has_unet else '❌'} UNet")
    print(f"   {'✅' if has_vae else '❌'} VAE")
    print(f"   {'✅' if has_clip else '❌'} CLIP")
    print(f"   {'✅' if has_unet_importance else '❌'} UNet attn_importance")
    
    # Verify ControlNet
    verify_ctrl = torch.load(ctrl_output_path, map_location="cpu", weights_only=False)
    verify_ctrl_state = verify_ctrl['state_dict']
    
    has_ctrl_importance = any('attn_importance' in k for k in verify_ctrl_state.keys())
    
    print(f"\n   ControlNet Checkpoint:")
    print(f"   {'✅' if len(verify_ctrl_state) > 0 else '❌'} ControlNet weights")
    print(f"   {'✅' if has_ctrl_importance else '❌'} ControlNet attn_importance")
    
    # Sample keys
    if has_unet_importance:
        unet_imp_samples = [k for k in verify_sd_state.keys() if 'attn_importance' in k][:3]
        print(f"\n   Sample UNet attn_importance keys:")
        for k in unet_imp_samples:
            print(f"      - {k}: {verify_sd_state[k].shape}")
    
    if has_ctrl_importance:
        ctrl_imp_samples = [k for k in verify_ctrl_state.keys() if 'attn_importance' in k][:3]
        print(f"\n   Sample ControlNet attn_importance keys:")
        for k in ctrl_imp_samples:
            print(f"      - {k}: {verify_ctrl_state[k].shape}")
    
    print("\n" + "="*80)
    print("✅ FINAL CHECKPOINTS SAVED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📦 Output files:")
    print(f"   1. {sd_output_path} ({sd_size_gb:.2f} GB)")
    print(f"   2. {ctrl_output_path} ({ctrl_size_gb:.2f} GB)")
    print("\n💡 Usage:")
    print(f"   - Use sd_v2_with_importance.ckpt as base SD model")
    print(f"   - Use controlnet_v2_with_importance.pth for ControlLDM")
    
    return sd_output_path, ctrl_output_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)