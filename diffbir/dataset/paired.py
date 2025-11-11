# diffbir/dataset/paired.py (새로 만들기)
import os
from typing import Dict, Union
import numpy as np
from PIL import Image
import torch.utils.data as data
import io

from .utils import load_file_list
from ..utils.common import instantiate_from_config


class PairedDataset(data.Dataset):
    """
    HQ와 LQ가 쌍으로 존재하는 데이터셋
    파일명 규칙: image.jpg (HQ), image_x4.jpg (LQ)
    """
    
    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Dict,
        lq_suffix: str = "x4",
    ):
        super().__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.lq_suffix = lq_suffix
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # HQ 파일 경로
        hq_path = self.image_files[index]["image_path"]
        prompt = self.image_files[index].get("prompt", "")
        
        # LQ 파일 경로 생성
        # /path/to/image.jpg → /path/to/image_x4.jpg
        base_path, ext = os.path.splitext(hq_path)
        lq_path = f"{base_path}{self.lq_suffix}{ext}"
        
        # HQ 이미지 로드
        hq_bytes = self.file_backend.get(hq_path)
        hq_image = Image.open(io.BytesIO(hq_bytes)).convert("RGB")
        hq = np.array(hq_image)  # [H, W, C], RGB, [0, 255]
        
        # LQ 이미지 로드
        lq_bytes = self.file_backend.get(lq_path)
        lq_image = Image.open(io.BytesIO(lq_bytes)).convert("RGB")
        lq = np.array(lq_image)  # [H_lq, W_lq, C], RGB, [0, 255]
        
        # 정규화
        # HQ: [0, 255] → [-1, 1]
        hq = (hq / 255.0 * 2 - 1).astype(np.float32)
        # LQ: [0, 255] → [0, 1]
        lq = (lq / 255.0).astype(np.float32)
        
        return {
            "hq": hq,  # [H, W, C], [-1, 1]
            "lq": lq,  # [H_lq, W_lq, C], [0, 1]
            "txt": prompt
        }
    
    def __len__(self):
        return len(self.image_files)