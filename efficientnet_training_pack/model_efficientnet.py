import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EfficientNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(EfficientNetUNet, self).__init__()
        
        # --- ENCODER (EfficientNet-B0) ---
        # Load weights if requested, otherwise random init
        weights = 'DEFAULT' if pretrained else None
        self.encoder = models.efficientnet_b0(weights=weights)
        
        # Extract specific feature layers
        # EfficientNet features are in .features
        # Indices for B0:
        # 0: Conv3x3 (stride 2) -> 32 channels
        # 1: MBConv1 (32->16)
        # 2: MBConv6 (16->24) stride 2
        # 3: MBConv6 (24->40) stride 2
        # 4: MBConv6 (40->80) 
        # 5: MBConv6 (80->112) 
        # 6: MBConv6 (112->192) stride 2
        # 7: MBConv6 (192->320)
        # 8: Conv1x1 (320->1280)
        
        # We need output at strides 2, 4, 8, 16, 32
        # Stride 2:  features[1] (16 ch)
        # Stride 4:  features[2] (24 ch)
        # Stride 8:  features[3] (40 ch)
        # Stride 16: features[5] (112 ch) (Actually index 4 is stride 16? Need to verify carefully. 
        # Let's assume generic indices and map dimensions dynamically if needed, but standard is:
        # x1: stride 2 (16)
        # x2: stride 4 (24)
        # x3: stride 8 (40)
        # x4: stride 16 (112)
        # x5: stride 32 (320) -> features[7] or [8]
        
        self.filters = [16, 24, 40, 112, 320] 
        
        # --- DECODER ---
        
        self.up4 = self._up_block(self.filters[4], self.filters[3])
        self.up3 = self._up_block(self.filters[3], self.filters[2])
        self.up2 = self._up_block(self.filters[2], self.filters[1])
        self.up1 = self._up_block(self.filters[1], self.filters[0])
        
        # Final upsampling to restore stride 2 -> stride 1
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.filters[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c + out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input normalization (Imagenet stats)
        # Assuming input is [0, 1] RGB
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Encoder Steps
        # We manually run through the sequential blocks to grab intermediate outputs
        features = self.encoder.features
        
        x_in = x
        
        # Stage 1 (Stride 2)
        x1 = features[0](x_in)
        x1 = features[1](x1) # 16ch
        
        # Stage 2 (Stride 4)
        x2 = features[2](x1) # 24ch
        
        # Stage 3 (Stride 8)
        x3 = features[3](x2) # 40ch
        
        # Stage 4 (Stride 16)
        x4 = features[4](x3)
        x4 = features[5](x4) # 112ch
        
        # Stage 5 (Stride 32)
        x5 = features[6](x4)
        x5 = features[7](x5) # 320ch
        
        # Decoder
        # Up from x5 (320) -> concat x4 (112) -> out (112)
        d4 = self.up4(x5) # Upsampled x5
        # Fix potential rounding errors in dimensions
        if d4.size() != x4.size():
            d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        
        # Up from d4 (112) -> concat x3 (40) -> out (40)
        d3 = self.up3(d4) # actually block takes concat inside? No my _up_block definition is weird.
        # My _up_block expects (in_c + out_c) at the Conv2d, but takes (in_c, out_c) in init.
        # Let's re-check definition.
        # __init__: Conv2d(in_c + out_c, ...) 
        # forward: we pass `d4` which has `in_c` channels? No.
        # My _up_block logic:
        # input is tensor FROM BELOW (lower res, `in_c` channels)
        # inside block: upsample it.
        # THEN it expects concatenation? No, my block doesn't handle concat. I handle concat in forward.
        # So input to block should be the CONCATENATED tensor?
        # No, usually you upsample, then concat, then conv.
        
        # Let's fix logic inline:
        # d4_up = upsample(x5)
        # d4_cat = cat([d4_up, x4])
        # d4_out = conv(d4_cat)
        
        # Re-running logic with corrected forward:
        pass # (See corrected file content)

        return self.final_up(d1)

# REDEFINING CLASS TO FIX LOGIC COMPLETELY
class EfficientNetUNetFixed(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(EfficientNetUNetFixed, self).__init__()
        
        weights = 'DEFAULT' if pretrained else None
        self.encoder = models.efficientnet_b0(weights=weights)
        
        # Encoder Channels
        # x1: 16, x2: 24, x3: 40, x4: 112, x5: 320
        self.filters = [16, 24, 40, 112, 320] 
        
        # Decoder Blocks
        # UpBlock takes (InChannels, SkipChannels, OutChannels)
        self.up4 = self._make_up_block(self.filters[4], self.filters[3], self.filters[3]) # 320 + 112 -> 112
        self.up3 = self._make_up_block(self.filters[3], self.filters[2], self.filters[2]) # 112 + 40 -> 40
        self.up2 = self._make_up_block(self.filters[2], self.filters[1], self.filters[1]) # 40 + 24 -> 24
        self.up1 = self._make_up_block(self.filters[1], self.filters[0], self.filters[0]) # 24 + 16 -> 16
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.filters[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def _make_up_block(self, in_c, skip_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Norm
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Extract Features
        features = self.encoder.features
        x1 = features[1](features[0](x)) # 16
        x2 = features[2](x1)             # 24
        x3 = features[3](x2)             # 40
        x4 = features[5](features[4](x3))# 112
        x5 = features[7](features[6](x4))# 320
        
        # Decode
        u4 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.up4(torch.cat([u4, x4], dim=1))
        
        u3 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.up3(torch.cat([u3, x3], dim=1))
        
        u2 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.up2(torch.cat([u2, x2], dim=1))
        
        u1 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.up1(torch.cat([u1, x1], dim=1))
        
        return self.final_up(d1)
