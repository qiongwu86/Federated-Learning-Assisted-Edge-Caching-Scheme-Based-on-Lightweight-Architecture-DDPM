import torch
import torch.nn.functional as F
import torch.nn as nn

# UNet3次上采样下采样
class LightweightUNet1D(nn.Module):
    def __init__(self, in_channels=16, base_channels=64, channel_mults=(1, 2, 4), time_emb_dim=128, groups=1):
        super().__init__()
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # 编码器通道配置
        channels = [in_channels + time_emb_dim] + [base_channels * m for m in channel_mults]

        # 下采样模块
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.GroupNorm(groups, channels[i + 1]),
                    nn.SiLU()
                )
            )
        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels[-1]),
            nn.SiLU()
        )

        # 上采样模块
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(channels[i + 1] * 2, channels[i], kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(groups, channels[i]),
                    nn.SiLU()
                )
            )

        # 输出 projection
        self.output_conv = nn.Conv1d(channels[0], in_channels, kernel_size=1)

    def forward(self, x, t):
        t = t.float().unsqueeze(-1)
        t_emb = self.time_embed(t)
        x_in = x.unsqueeze(-1)
        t_in = t_emb.unsqueeze(-1).repeat(1, 1, x_in.size(-1))
        out = torch.cat([x_in, t_in], dim=1)

        skips = []
        for down in self.down_blocks:
            out = down(out)
            skips.append(out)
            if out.size(-1) >= 2:
                out = F.avg_pool1d(out, 2)

        # 瓶颈
        out = self.bottleneck(out)

        # 上采样
        for up in self.up_blocks:
            skip = skips.pop()
            out = F.interpolate(out, size=skip.size(-1), mode='nearest')
            out = torch.cat([out, skip], dim=1)
            out = up(out)

        # 输出
        out = self.output_conv(out)
        out = out.mean(dim=-1)
        return out

class GaussianMultinomialDiffusion(nn.Module):
    def __init__(self, num_numerical_features, denoise_fn, num_timesteps, device='cpu'):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        alphas = 1. - self.betas
        self.sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(alphas, dim=0)).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - torch.cumprod(alphas, dim=0)).to(device)

    def extract(self, arr, timesteps, broadcast_shape):
        res = arr.gather(-1, timesteps).float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_t * x_start + sqrt_one_minus_alphas_t * noise

    def forward(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        x_noisy = self.q_sample(x, t, noise)
        model_out = self.denoise_fn(x_noisy, t)
        return model_out, noise

    def compute_loss(self, x, t):
        model_out, noise = self.forward(x, t)
        return F.mse_loss(model_out, noise)

    def reverse_diffusion_step(self, x, t):
        # 对 batch 中所有样本重复 t
        batch_size = x.size(0)
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=x.device)
        betas_t = self.extract(self.betas, t_tensor, x.shape)
        one_minus_alpha_bar_sqrt = self.extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
        eps_theta = self.denoise_fn(x, t_tensor)
        coeff = betas_t / one_minus_alpha_bar_sqrt
        mean = (1 / (1 - betas_t).sqrt()) * (x - coeff * eps_theta)
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        sigma_t = betas_t.sqrt()
        return mean + sigma_t * z

    def sample(self, num_samples):
        samples = torch.randn((num_samples, self.num_numerical_features)).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            samples = self.reverse_diffusion_step(samples, t)
        return samples
