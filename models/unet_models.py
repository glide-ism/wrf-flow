import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models.
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention computation
        q = q.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        k = k.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        v = v.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        
        # Compute attention
        scale = 1.0 / math.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.bmm(attn, v)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        
        h = self.to_out(h)
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2, downsample=True, attention=False):
        super().__init__()
        self.layers = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        self.attention = AttentionBlock(out_channels) if attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1) if downsample else None
    
    def forward(self, x, time_emb):
        for layer in self.layers:
            x = layer(x, time_emb)
        
        if self.attention:
            x = self.attention(x)
        
        if self.downsample:
            x = self.downsample(x)
        
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, skip_channels=None, num_layers=2, upsample=True, attention=False):
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels
            
        self.layers = nn.ModuleList([
            ResBlock((in_channels + skip_channels) if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        self.attention = AttentionBlock(out_channels) if attention else None
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1) if upsample else None
    
    def forward(self, x, skip, time_emb):
        x = torch.cat([x, skip], dim=1)
        
        for layer in self.layers:
            x = layer(x, time_emb)
        
        if self.attention:
            x = self.attention(x)
        
        if self.upsample:
            x = self.upsample(x)
        
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_heads=1,
        time_embedding_dim=None,
    ):
        super().__init__()
        
        if time_embedding_dim is None:
            time_embedding_dim = model_channels * 4
        
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Calculate channel multipliers for each resolution
        ch_mult = (1,) + tuple(channel_mult)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        now_ch = model_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = DownBlock(
                    now_ch, 
                    out_ch, 
                    time_embedding_dim,
                    num_layers=1,
                    downsample=False,
                    attention=(32 // (2**i)) in attention_resolutions
                )
                self.down_blocks.append(layers)
                channels.append(out_ch)
                now_ch = out_ch
            
            if i < len(channel_mult) - 1:  # Don't downsample on the last block
                layers = DownBlock(
                    now_ch,
                    now_ch,
                    time_embedding_dim,
                    num_layers=1,
                    downsample=True,
                    attention=False
                )
                self.down_blocks.append(layers)
                channels.append(now_ch)
        
        # Middle blocks
        self.middle_block1 = ResBlock(now_ch, now_ch, time_embedding_dim)
        self.middle_attn = AttentionBlock(now_ch)
        self.middle_block2 = ResBlock(now_ch, now_ch, time_embedding_dim)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = UpBlock(
                    now_ch,
                    out_ch,
                    time_embedding_dim,
                    skip_channels=skip_ch,
                    num_layers=1,
                    upsample=(j == num_res_blocks and i > 0),
                    attention=(32 // (2**i)) in attention_resolutions
                )
                self.up_blocks.append(layers)
                now_ch = out_ch
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            #nn.LayerNorm(now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        time_emb = get_timestep_embedding(timesteps, self.time_embedding_dim // 4)
        time_emb = self.time_embed(time_emb)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Store skip connections
        skips = [x]
        
        # Downsampling
        for block in self.down_blocks:
            x = block(x, time_emb)
            skips.append(x)
        
        # Middle
        x = self.middle_block1(x, time_emb)
        x = self.middle_attn(x)
        x = self.middle_block2(x, time_emb)
        
        # Upsampling
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, time_emb)
        
        # Final output
        x = self.final_conv(x)
        
        return x


# Example usage and model configurations
def create_cifar10_unet():
    """Create UNet for CIFAR-10 (32x32) as described in the paper"""
    return UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16,),  # Self-attention at 16x16 resolution
        channel_mult=(1, 2, 2, 2),   # Four resolution levels: 32, 16, 8, 4
        dropout=0.1,
    )

def create_mnist_unet():
    """Create UNet for MNIST"""
    return UNet(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(14,),  # Self-attention at 16x16 resolution
        channel_mult=(1, 2, 2),   # Four resolution levels: 28, 14, 7
        dropout=0.1,
    )    


def create_celeba_unet():
    """Create UNet for CelebA-HQ (256x256) as described in the paper"""
    return UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16,),  # Self-attention at 16x16 resolution  
        channel_mult=(1, 1, 2, 2, 4, 4),  # Six resolution levels for 256x256
        dropout=0.0,  # No dropout for larger images as mentioned in paper
    )


if __name__ == "__main__":
    # Test the model
    model = create_cifar10_unet()
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
