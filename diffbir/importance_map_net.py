# importance_map_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SharedEncoderImportanceNet(nn.Module):
    """
    Shared Encoder 기반 중요도 맵 네트워크
    SR 이미지와 HR 이미지를 분리해서 입력받음
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 공유 encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 나머지 네트워크
        self.enc2 = self.conv_block(64, 64)
        self.enc3 = self.conv_block(64, 128)
        
        self.bottleneck = self.conv_block(128, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(96, 32)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sr_image, hr_image):
        """
        Args:
            sr_image: (B, 3, H, W)
            hr_image: (B, 3, H, W)
        """
        # 공유 encoder로 feature 추출
        feat_sr = self.shared_encoder(sr_image)
        feat_hr = self.shared_encoder(hr_image)
        
        # Concat
        enc1 = torch.cat([feat_sr, feat_hr], dim=1)
        
        # Encoder
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder
        dec3 = self.upconv3(bottleneck)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.out_conv(dec1)
        out = torch.sigmoid(out)
        
        return out
    
    def fit(self, training_data, num_epochs=100, learning_rate=0.001, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            save_path='./importance_net_weights.pth'):
        """학습 메서드"""
        
        print("="*60)
        print("Training Importance Map Network")
        print("="*60)
        
        if isinstance(training_data, dict):
            training_data = [training_data]
        
        if len(training_data) == 0:
            raise ValueError("training_data is empty!")
        
        print(f"Training with {len(training_data)} samples")
        
        self.to(device)
        self.train()  # nn.Module.train() 모드
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, data in enumerate(training_data):
                # 타겟 계산
                target_importance = self.compute_target_importance(data)
                
                # 입력 전처리 (2개 반환!)
                sr_tensor, hr_tensor = self.preprocess_input(
                    data['sr_lr_son'], 
                    data['lr_input']
                )
                target_tensor = self.preprocess_target(target_importance)
                
                # 디바이스 이동
                sr_tensor = sr_tensor.to(device)
                hr_tensor = hr_tensor.to(device)
                target_tensor = target_tensor.to(device)
                
                # Forward (2개 입력!)
                pred_importance = self(sr_tensor, hr_tensor)
                
                # 크기 맞추기
                if pred_importance.shape != target_tensor.shape:
                    pred_importance = F.interpolate(
                        pred_importance, 
                        size=target_tensor.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Loss
                loss = criterion(pred_importance, target_tensor)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(training_data)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), save_path)
                if (epoch + 1) % 10 == 0:
                    print(f"✓ Best model saved! Loss: {best_loss:.6f}")
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best Loss: {best_loss:.6f}")
        print(f"{'='*60}\n")
        
        return best_loss
    
    def compute_target_importance(self, data):
        """타겟 중요도 맵 계산"""
        from imresize import imresize
        
        gt_image = data['gt_image']
        sr_lr_son = data['sr_lr_son']
        lr_input = data['lr_input']
        
        # Bicubic baseline
        bicubic_up = imresize(lr_input, output_shape=gt_image.shape[:2], kernel='cubic')
        sr_lr_son_resized = imresize(sr_lr_son, output_shape=gt_image.shape[:2], kernel='cubic')

        # 에러 계산
        bicubic_error = np.square(gt_image - bicubic_up)
        sr_error = np.square(gt_image - sr_lr_son_resized)
        
        if len(bicubic_error.shape) == 3:
            bicubic_error = np.mean(bicubic_error, axis=2)
            sr_error = np.mean(sr_error, axis=2)
        
        # Z-score + Sigmoid로 정규화
        sr_error_mean = sr_error.mean()
        sr_error_std = sr_error.std()
        sr_error_z = (sr_error - sr_error_mean) / (sr_error_std + 1e-10)
        
        # 품질 = 1 / (1 + exp(z))
        # 에러 작으면 quality 높음
        quality = 1.0 / (1.0 + np.exp(sr_error_z))

        return quality

    def preprocess_input(self, sr_lr_son, hr_input):
        """입력 전처리 - 2개 반환!"""
        # Grayscale → RGB
        if len(sr_lr_son.shape) == 2:
            sr_lr_son = np.expand_dims(sr_lr_son, axis=2)
            sr_lr_son = np.repeat(sr_lr_son, 3, axis=2)
        
        if len(hr_input.shape) == 2:
            hr_input = np.expand_dims(hr_input, axis=2)
            hr_input = np.repeat(hr_input, 3, axis=2)
        
        # 각각 변환
        sr_tensor = torch.FloatTensor(sr_lr_son).permute(2, 0, 1).unsqueeze(0)
        hr_tensor = torch.FloatTensor(hr_input).permute(2, 0, 1).unsqueeze(0)
        
        return sr_tensor, hr_tensor  # 2개!
    
    def preprocess_target(self, target_importance):
        """타겟 전처리"""
        target_tensor = torch.FloatTensor(target_importance).unsqueeze(0).unsqueeze(0)
        return target_tensor
    
    def predict(self, sr_lr_son, lr_input, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """예측"""
        self.eval()
        self.to(device)
        
        with torch.no_grad():
            sr_tensor, hr_tensor = self.preprocess_input(sr_lr_son, lr_input)
            sr_tensor = sr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            
            pred_tensor = self(sr_tensor, hr_tensor)  # 2개 입력!
            importance_map = pred_tensor.squeeze().cpu().numpy()
        
        return importance_map