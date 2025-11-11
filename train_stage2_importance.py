"""
ImportanceSpatialTransformer 학습 스크립트 (train_stage2.py 기반)
- ControlLDM의 ImportanceSpatialTransformer (LoRA)만 학습시킵니다.
"""

import os
from argparse import ArgumentParser
import copy
from diffbir.importance_map_net import SharedEncoderImportanceNet # ⭐ 추가
from peft import LoraConfig, inject_adapter_in_model, TaskType # ⭐ 추가

from omegaconf import OmegaConf
import torch
import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn import functional as F # ⭐ F.interpolate 사용을 위해 추가
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, SwinIR, Diffusion # SwinIR은 train_stage2에서 사용
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler


def main(args) -> None:
    # ========== Setup ==========
    accelerator = Accelerator(split_batches=True, mixed_precision='fp16') # ⭐ fp16 추가
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # ========== Load ControlLDM ==========
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd_checkpoint = torch.load(cfg.train.sd_path, map_location="cpu")
    if "state_dict" in sd_checkpoint:
        sd = sd_checkpoint["state_dict"]
    else:
        sd = sd_checkpoint
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    # Load ControlNet
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )
    
    # ========== Load Cleaner (SwinIR) - FROZEN ==========
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir_weight = torch.load(cfg.train.swinir_path, map_location="cpu")
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
        print(f"✅ Loaded frozen SwinIR from {cfg.train.swinir_path}")

    # ========== Load Importance Net - FROZEN (⭐ 추가) ==========
    importance_net = SharedEncoderImportanceNet(in_channels=3)
    importance_net.load_state_dict(
        torch.load(cfg.train.importance_net_path, map_location=device)
    )
    importance_net.to(device)
    importance_net.eval()
    for param in importance_net.parameters():
        param.requires_grad = False
    if accelerator.is_main_process:
        print("✅ Loaded frozen importance_net")
    
    # ========== Load Diffusion ==========
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    # ========== Freeze All & Apply LoRA to ImportanceSpatialTransformer ==========
    for param in cldm.parameters():
        param.requires_grad = False
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )

    trainable_params = []
    trainable_module_names = []
    
    # ImportanceSpatialTransformer 찾아서 LoRA 적용
    for name, module in cldm.named_modules():
        module_type = type(module).__name__
        if 'ImportanceSpatialTransformer' in module_type:
            trainable_module_names.append(name)
            try:
                inject_adapter_in_model(lora_config, module, adapter_name="default")
                if accelerator.is_main_process:
                    print(f"  ✅ Applied LoRA to: {name}")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"  ⚠️  Failed to apply LoRA to {name}: {e}")

    # LoRA 파라미터만 수집
    trainable_count = 0
    for name, param in cldm.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
            trainable_count += param.numel()

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in cldm.parameters())
        print(f"\n{'='*60}")
        print(f"Found {len(trainable_module_names)} ImportanceSpatialTransformer modules.")
        print(f"\n📊 Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable (LoRA): {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
        print(f"   Frozen: {total_params - trainable_count:,}")
        print(f"{'='*60}\n")

    if len(trainable_params) == 0:
        raise ValueError(
            "❌ No trainable LoRA parameters found! Check if LoRA injection succeeded."
        )

    # Setup optimizer:
    opt = torch.optim.AdamW(trainable_params, lr=cfg.train.learning_rate) # ⭐ ControlNet 대신 LoRA 파라미터로 수정

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

    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval()
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
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
            # ⭐ train_stage2.py 기반 Unpack
            gt, lq, prompt = batch 
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # --- ⭐ HQ/LQ 크롭 로직 (메모리 부족 방지 위해 다시 삽입) ---
            TARGET_SIZE = 128
            b_size, _, h_init, w_init = gt.size() # [B, C, H, W]
            
            if h_init >= TARGET_SIZE and w_init >= TARGET_SIZE:
                
                # H, W 슬라이싱이 아닌 C, H, W 슬라이싱을 위해 B, C, H, W 순서로 가정
                h_c, w_c = gt.shape[2], gt.shape[3]
                
                # 크롭 시작 지점 무작위 선택
                rand_h = random.randint(0, h_c - TARGET_SIZE)
                rand_w = random.randint(0, w_c - TARGET_SIZE)
                
                # HQ 이미지 크롭 (gt는 [B, C, H, W])
                gt = gt[:, :, rand_h:rand_h + TARGET_SIZE, rand_w:rand_w + TARGET_SIZE]
                
                # LQ 이미지 크롭 (lq는 [B, C, H_lq, W_lq])
                SCALE_FACTOR = h_init // lq.shape[2] 
                TARGET_LQ_SIZE = TARGET_SIZE // SCALE_FACTOR
                
                rand_h_lq = rand_h // SCALE_FACTOR 
                rand_w_lq = rand_w // SCALE_FACTOR
                
                lq = lq[:, :, rand_h_lq:rand_h_lq + TARGET_LQ_SIZE, rand_w_lq:rand_w_lq + TARGET_LQ_SIZE]
            
            # --- ⭐ 크롭 로직 종료 ---
            
            with torch.no_grad():
                # ⭐ VAE 인코딩 (GT)
                z_0 = pure_cldm.vae_encode(gt)
                
                # ⭐ Cleaner (SwinIR) 호출
                clean = swinir(lq)
                
                # HQ 크기에 맞춰 Clean 이미지 리사이즈 (train_stage2.py에는 없는 로직)
                # VAE 인코딩 전에 Clean 이미지 크기를 Latent Space와 일치시켜야 할 수 있음
                # 현재는 Cleaner의 출력이 VAE 입력 크기에 맞아야 하므로, 
                # clean.shape[2:]가 512x512를 유지하도록 보장해야 합니다.
                
                # LQ upsampling (cleaner의 출력 크기가 4x라고 가정)
                if clean.shape[2] != gt.shape[2] or clean.shape[3] != gt.shape[3]:
                    clean = F.interpolate(
                        clean,
                        size=gt.shape[2:],
                        mode='bicubic',
                        align_corners=False
                    )
                
                # ⭐ Importance Map 생성 로직 (⭐ 추가)
                lq_upsampled = F.interpolate(
                    lq,
                    size=clean.shape[2:],
                    mode='bicubic',
                    align_corners=False
                )
                importance_map = importance_net(clean, lq_upsampled) # ⭐ importance_map 생성
                
                # ⭐ Condition 준비 (train_stage2.py 기반)
                cond = pure_cldm.prepare_condition(clean, prompt)
                
                # noise augmentation
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
                
                # Importance Map 크기를 Latent Size에 맞게 조정 (⭐ 추가)
                """importance_map_latent = F.interpolate(
                    importance_map,
                    size=z_0.shape[2:], # Latent Size (예: 64x64)
                    mode='bilinear',
                    align_corners=False
                )"""

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            # ⭐ Forward 호출 시 importance_map_latent 전달
            loss = diffusion.p_losses(cldm, z_0, t, cond_aug, importance_map)
            
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

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
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

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # ⭐ LoRA/Adapter 가중치만 저장
                    adapter_state_dict = {}
                    for name, param in pure_cldm.named_parameters():
                        if 'lora' in name.lower() and param.requires_grad:
                            adapter_state_dict[name] = param.cpu().clone()

                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(adapter_state_dict, ckpt_path) # ⭐ Adapter만 저장하도록 수정

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                # ... (로깅 로직은 train_stage2.py와 동일하게 유지)
                # ...
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
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/condition_aug_decoded",
                                (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                            # ⭐ 중요도 맵 로깅 추가
                            ("image/importance_map", importance_map[:N]), 
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        # ... (후반 로깅 및 종료 로직은 동일)
        # ...

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)