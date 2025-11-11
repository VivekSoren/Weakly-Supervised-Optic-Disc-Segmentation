import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDisc(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        def Conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), 
                nn.BatchNorm2d(out_c), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(out_c, out_c, 3, padding=1), 
                nn.BatchNorm2d(out_c), 
                nn.ReLU(inplace=True)
            )
        self.enc1 = Conv(in_channels, base)
        self.enc2 = Conv(base, base*2)
        self.enc3 = Conv(base*2, base*4)
        self.enc4 = Conv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.mid = Conv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = Conv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = Conv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = Conv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = Conv(base*2, base)
        self.final = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.mid(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)   # logits (B, 1, H, W)

class UNetCup(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.d1 = DoubleConv(3,64)
        self.d2 = DoubleConv(64,128)
        self.d3 = DoubleConv(128,256)
        self.d4 = DoubleConv(256,512)

        self.u1 = DoubleConv(256+512, 256)
        self.u2 = DoubleConv(128+256, 128)
        self.u3 = DoubleConv(64+128, 64)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))

        u1 = self.up(c4)
        u1 = self.u1(torch.cat([u1, c3], dim=1))
        u2 = self.up(u1)
        u2 = self.u2(torch.cat([u2, c2], dim=1))
        u3 = self.up(u2)
        u3 = self.u3(torch.cat([u3, c1], dim=1))

        return torch.sigmoid(self.outc(u3))


class Discriminator(nn.Module):
    def __init__(self, in_ch=4, base=64):
        super().__init__()
        def D_block(in_c, out_c, stride=2, normalization=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            D_block(in_ch, base, normalization=False), 
            D_block(base, base*2),
            D_block(base*2, base*4), 
            D_block(base*4, base*8, stride=1), 
            nn.Conv2d(base*8, 1, 4, padding=1), 
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Sigmoid()
        )
    
    def forward(self, image, segmentation):
        x = torch.cat([image, segmentation], dim=1)
        return self.model(x)

class GAN_Segmentation(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, device='cuda'):
        super().__init__()
        self.generator = UNetDisc(in_ch, out_ch, base)
        self.discriminator = Discriminator(in_ch + out_ch, base)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.generator(x)

    def _expand_mask_dims(self, masks):
        if masks.dim() == 3:
            return masks.unsqueeze(1)
        return masks

    def compute_dice_score(self, pred_logits, target_masks, threshold=0.5):
        # Expand dims
        target_masks_expanded = self._expand_mask_dims(target_masks)
        # Convert logits into probabilities and then to binary
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > threshold).float()
        # Compute dice score
        intersection = (pred_binary * target_masks_expanded).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target_masks_expanded.sum(dim=(2, 3))
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return dice.mean()

    def generator_step(self, real_images, real_masks):
        ### Generator training step
        ### real_images: Input Images (B, 3, H, W)
        ### real_masks: Ground truth masks (B, 1, H, W)
        # Expand mask dimension if needed
        real_masks_expanded = self._expand_mask_dims(real_masks)
        # Generate fake masks
        fake_masks = self.generator(real_images)
        # Compute dice score for monitoring
        dice_score = self.compute_dice_score(fake_masks, real_masks)
        # Discriminator should think hey are equal
        validity = self.discriminator(real_images, fake_masks)
        # Generator loss: adversarial loss + segmentation loss
        adversarial_loss = F.binary_cross_entropy(validity, torch.ones_like(validity))
        segmentation_loss = F.binary_cross_entropy_with_logits(fake_masks, real_masks_expanded)

        # Combine losses 
        generator_loss = adversarial_loss + 100 * segmentation_loss     # Weight for segmentation

        return {
            'generator_loss': generator_loss, 
            'adversarial_loss': adversarial_loss, 
            'segmentation_loss': segmentation_loss, 
            'fake_masks': fake_masks, 
            'dice_score': dice_score
        }

    def discriminator_step(self, real_images, real_masks):
        ### Discriminator training step
        ### real_images: Input image (B, 3, H, W)
        ### real_masks: Ground truth masks (B, 1, H, W)
        # Expand masks if needed
        real_masks_expanded = self._expand_mask_dims(real_masks)
        # Generate fake masks
        with torch.no_grad(): 
            fake_masks = self.generator(real_images)
        # Real samples
        real_validity = self.discriminator(real_images, real_masks_expanded)
        real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))

        # Fake samples
        fake_validity = self.discriminator(real_images, fake_masks)
        fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))

        # Total discriminator loss
        discriminator_loss = (real_loss + fake_loss) / 2

        return { 
            'discriminator_loss': discriminator_loss, 
            'real_score': real_validity.mean(), 
            'fake_score': fake_validity.mean()
        }

        