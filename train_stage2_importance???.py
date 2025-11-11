"""
ImportanceSpatialTransformer 학습 스크립트
- Cleaner를 선택적으로 사용 (또는 사용 안 함)
- 기존 train_stage2.py와 동일한 구조
"""

import os
from argparse import ArgumentParser
from diffbir.importance_map_net import SharedEncoderImportanceNet
from peft import LoraConfig, inject_adapter_in_model, TaskType

from omegaconf import OmegaConf
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn import functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler


def main(args):
    # ========== Setup ==========
    accelerator = Accelerator(split_batches=True, mixed_precision='fp16')
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    
    # ========== Load ControlLDM ==========
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    
    # Load pretrained SD
    sd_checkpoint = torch.load(cfg.train.sd_path, map_location="cpu")
    if "state_dict" in sd_checkpoint:
        sd = sd_checkpoint["state_dict"]
    else:
        sd = sd_checkpoint  # 직접 state_dict인 경우
    unused, missing = cldm.load_pretrained_sd(sd)
    
    # Load ControlNet
    if cfg.train.resume:
        control_sd = torch.load(cfg.train.resume, map_location="cpu")
        cldm.load_controlnet_from_ckpt(control_sd, strict=False)
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
    
    # ========== Load Cleaner (SwinIR/BSRNet) - FROZEN ==========
    cleaner = instantiate_from_config(cfg.model.cleaner)
    cleaner_weight = torch.load(cfg.train.cleaner_path, map_location="cpu")
    if "state_dict" in cleaner_weight:
        cleaner_weight = cleaner_weight["state_dict"]
    cleaner.load_state_dict(cleaner_weight, strict=True)
    cleaner.eval().to(device)
    
    # Freeze cleaner
    for param in cleaner.parameters():
        param.requires_grad = False
    
    if accelerator.is_main_process:
        print("✅ Loaded frozen cleaner (SwinIR/BSRNet)")
    
    # ========== Load Importance Net - FROZEN ==========
    importance_net = SharedEncoderImportanceNet(in_channels=3)
    importance_net.load_state_dict(
        torch.load(cfg.train.importance_net_path, map_location=device)
    )
    importance_net.to(device)
    importance_net.eval()
    
    # Freeze importance net
    for param in importance_net.parameters():
        param.requires_grad = False
    
    if accelerator.is_main_process:
        print("✅ Loaded frozen importance_net")
    
    # ========== Load Diffusion ==========
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    # ========== Freeze All Except importance_spatial_transformer ==========
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
    trainable_count = 0

    # ⭐ ImportanceSpatialTransformer 찾아서 LoRA 적용
    for name, module in cldm.named_modules():
        module_type = type(module).__name__
        
        if 'ImportanceSpatialTransformer' in module_type:
            trainable_module_names.append(name)
            
            # ⭐⭐⭐ 이 모듈에 LoRA 주입!
            try:
                # PEFT의 inject_adapter_in_model 사용
                inject_adapter_in_model(lora_config, module, adapter_name="default")
                
                if accelerator.is_main_process:
                    print(f"  ✅ Applied LoRA to: {name}")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"  ⚠️  Failed to apply LoRA to {name}: {e}")

    # ⭐ LoRA 파라미터만 수집 (이름에 'lora' 포함)
    for name, param in cldm.named_parameters():
        if 'lora' in name.lower():  # LoRA 파라미터 찾기
            param.requires_grad = True
            trainable_params.append(param)
            trainable_count += param.numel()

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in cldm.parameters())
        print(f"\n{'='*60}")
        print(f"Found {len(trainable_module_names)} ImportanceSpatialTransformer modules:")
        for name in trainable_module_names[:5]:
            print(f"  - {name}")
        if len(trainable_module_names) > 5:
            print(f"  ... and {len(trainable_module_names) - 5} more")
        print(f"\n📊 Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable (LoRA): {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
        print(f"   Frozen: {total_params - trainable_count:,}")
        print(f"{'='*60}\n")

    if len(trainable_params) == 0:
        raise ValueError(
            "❌ No trainable LoRA parameters found!\n"
            "   Check if ImportanceSpatialTransformer modules exist and have target_modules."
        )

    opt = torch.optim.AdamW(trainable_params, lr=cfg.train.learning_rate)
    
    # ========== Dataset ==========
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    
    # ========== Batch Transform ==========
    batch_transform = instantiate_from_config(cfg.batch_transform)
    
    # ========== Prepare ==========
    cldm.train().to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    
    noise_aug_timestep = cfg.train.noise_aug_timestep
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    
    # ========== Training Loop ==========
    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            total=len(loader),
            initial=0,
            desc=f"Epoch: {epoch}",
        )
        
        for batch in loader:
            # ===== 1. Batch Transform =====
            batch = batch_transform(batch)
            
            # ===== 2. Unpack Batch =====
            
            TARGET_SIZE = 512 # HQ 목표 해상도 (예: 512x512)

            # 배치에서 텐서를 가져옵니다. (HQ, LQ 모두 [B, C, H, W] 형태라고 가정)
            hq = batch['hq']  
            lq = batch['lq']  

            b, w, h, c = hq.size()
            
            # 1. 스케일 팩터 및 LQ 목표 크기 계산
            # SCALE_FACTOR: hq의 높이(h)와 lq의 높이(lq.shape[2])를 이용해 배율을 계산합니다.
            SCALE_FACTOR = h // lq.shape[2] 
            TARGET_LQ_SIZE = TARGET_SIZE // SCALE_FACTOR
            
            # 2. HQ 크기가 목표 크기보다 클 때만 크롭을 수행합니다.
            print("H :", h, "W :", w, "TARGET_SIZE :", TARGET_SIZE)
            if h >= TARGET_SIZE and w >= TARGET_SIZE:
                print("Applying random crop to HQ and LQ images.")
                # 2-1. 크롭 시작 지점 무작위 선택
                rand_h = random.randint(0, h - TARGET_SIZE)
                rand_w = random.randint(0, w - TARGET_SIZE)
                
                # 2-2. HQ 이미지 크롭
                # [B, C, H_start:H_end, W_start:W_end]
                hq = hq[:, rand_w:rand_w + TARGET_SIZE, rand_h:rand_h + TARGET_SIZE, :]
                batch['hq'] = hq
                
                # 2-3. LQ 이미지 크롭 (HQ와 정확히 대응되는 영역)
                
                # hq 크롭 시작 지점을 스케일링하여 lq 크롭 시작 지점 결정
                rand_h_lq = rand_h // SCALE_FACTOR 
                rand_w_lq = rand_w // SCALE_FACTOR
                
                # LQ 이미지 크롭 (LQ 목표 크기 사용)
                lq = lq[:, rand_w_lq:rand_w_lq + TARGET_LQ_SIZE, rand_h_lq:rand_h_lq + TARGET_LQ_SIZE, :]
                batch['lq'] = lq
            
            # =========================================================
            # ⭐ 크롭 로직 종료
            # =========================================================

            # ===== 2. Unpack Batch =====
            # 이제 hq와 lq는 크롭된 텐서로 언팩됩니다.
            hq = batch['hq'] 
            lq = batch['lq']
            txt = batch.get('txt', [''] * len(hq))
            
            # ===== 3. NumPy → Torch & Transpose =====
            hq = hq.permute(0, 3, 1, 2).contiguous()
            lq = lq.permute(0, 3, 1, 2).contiguous()

            # ===== 4. Normalize & Move to Device =====
            hq = hq.to(torch.float32).to(device)  # [-1, 1]
            lq = (lq.to(torch.float32) * 2 - 1).to(device)  # [0,1] → [-1,1]
            bs = hq.shape[0]
            
            # ===== 5. Apply Cleaner (LQ → Clean) =====
            with torch.no_grad():
                # LQ → Cleaner → Clean (4x upscale)
                clean_upscaled = cleaner(lq)  # [B, 3, H*4, W*4]
                
                # HQ 크기로 resize
                clean = F.interpolate(
                    clean_upscaled,
                    size=hq.shape[2:],
                    mode='bicubic',
                    align_corners=False
                )
                
                # LQ upsampling
                lq_upsampled = F.interpolate(
                    lq,
                    size=clean.shape[2:],
                    mode='bicubic',
                    align_corners=False
                )
                print("Shape of LQ:", lq.shape, "Shape of HQ:", hq.shape, "Shape of Clean:", clean.shape)
                # Generate Importance Map
                importance_map = importance_net(clean, lq_upsampled)
                print("map done!!")
                # ⭐ Clean → Latent (condition)
                clean_latent = pure_cldm.vae_encode(clean)
                print("clean latent done!!")
                # ⭐ HQ → Latent (target)
                hq_latent = pure_cldm.vae_encode(hq)
                print("hq latent done!!")
                # Text embedding
                txt_emb = pure_cldm.clip.encode(txt)
                print("txt emb done!!")
                # Importance map resize
                '''importance_map_latent = F.interpolate(
                    importance_map,
                    size=clean_latent.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )'''
            
            # ===== 6. Sample Timestep =====
            t = torch.randint(
                low=0, high=diffusion.num_timesteps,
                size=(bs,), device=device, dtype=torch.long
            )
            
            # ===== 7. Add Noise to HQ latent =====
            noise = torch.randn_like(hq_latent)
            noised = diffusion.q_sample(hq_latent, t, noise)  # ⭐ HQ에 노이즈 추가
            print("noised done!!")
            # ===== 8. Prepare Condition =====
            cond = {
                "c_txt": txt_emb,
                "c_img": clean_latent,  # Clean latent as condition
            }
            
            # ===== 9. Forward =====
            # ⭐ Noised HQ latent → Denoise → Predict noise
            print("noised shape :", noised.shape, "t shape :", t.shape, "importance_map shape :", importance_map.shape,"cond c_img shape :", cond['c_img'].shape)
            pred = cldm(noised, t, cond, importance_map)
            print("forward done!!")
            # ===== 10. Compute Loss =====
            # ⭐ Predict한 noise vs 실제 noise
            loss = F.mse_loss(pred, noise, reduction="mean")
            
            # ===== 11. Backward =====
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
                    
            # ===== 12. Logging =====
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)
                
            global_step += 1
            if global_step >= max_steps:
                break
                
            pbar.close()
            epoch += 1
    
    # ========== 12. Final Save ==========
    if accelerator.is_main_process:
        importance_state_dict = {}
        for name, param in pure_cldm.named_parameters():
            if 'importance_embed' in name and param.requires_grad:
                importance_state_dict[name] = param.cpu().clone()
        
        final_path = os.path.join(ckpt_dir, f"importance_final_step_{global_step}.pt")
        torch.save(importance_state_dict, final_path)
        print(f"\n🎉 Training complete! Final weights: {final_path}")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)