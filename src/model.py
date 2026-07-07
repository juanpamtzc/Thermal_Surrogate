import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """2D Fourier layer: performs FFT, linear transform in frequency domain, and inverse FFT."""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Number of Fourier modes to keep (lower frequencies contain most of the physical information)
        self.modes1 = modes1 
        self.modes2 = modes2

        # Complex weights for the spectral multiplication
        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication using Einstein summation
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # 1. Compute 2D Fast Fourier Transform
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. Return to physical space using Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class ThermalFNO(nn.Module):
    """
    The full Fourier Neural Operator architecture.
    Maps [Mask, U_inlet, Grid_X, Grid_Y] -> [Temperature]
    """
    def __init__(self, modes1=12, modes2=32, width=32):
        super(ThermalFNO, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # "Lifting" layer: projects the 4 input channels to a higher-dimensional latent space
        self.p = nn.Linear(4, self.width) 

        # 4 layers of Spectral Convolutions (The core FNO blocks)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # 1x1 convolutions for the "skip connections" (preserves high-frequency local details)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # "Projection" layer: maps the latent space back down to the 1 output channel (Temperature)
        self.q = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Permute from [Batch, Channels, Y, X] to [Batch, Y, X, Channels] for Linear layer
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        # FNO Block 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        # FNO Block 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # FNO Block 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        # FNO Block 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        # Project back to Temperature field
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = x.permute(0, 3, 1, 2)
        
        return x