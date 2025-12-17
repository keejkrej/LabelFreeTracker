"""UNet model architecture for label-free microscopy prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Encoder block with convolutions, pooling, and batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv: int = 2,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        pool_size: tuple[int, int, int] = (2, 2, 2),
        pool_stride: tuple[int, int, int] = (2, 2, 2),
        dropout: bool = False,
    ):
        super().__init__()
        self.n_conv = n_conv
        self.dropout = dropout

        self.convs = nn.ModuleList()
        curr_in = in_channels
        for _ in range(n_conv):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            self.convs.append(
                nn.Conv3d(curr_in, out_channels, kernel_size, padding=padding)
            )
            curr_in = out_channels

        self.pool = nn.MaxPool3d(
            kernel_size=pool_size, stride=pool_stride, padding=0, ceil_mode=True
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout3d(x, p=0.5, training=self.training)

        to_concat = x
        x = self.pool(x)
        x = self.bn(x)

        return x, to_concat


class DeconvBlock(nn.Module):
    """Decoder block with transposed convolution, skip connection, and batch normalization."""

    def __init__(
        self,
        in_channels: int,
        concat_channels: int,
        out_channels: int,
        n_conv: int = 2,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (2, 2, 2),
        dropout: bool = False,
    ):
        super().__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        output_padding = (stride[0] - 1, stride[1] - 1, stride[2] - 1)

        self.upconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        self.convs = nn.ModuleList()
        curr_in = out_channels
        for _ in range(n_conv):
            p = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            self.convs.append(nn.Conv3d(curr_in, out_channels, kernel_size, padding=p))
            curr_in = out_channels

        self.bn = nn.BatchNorm3d(out_channels + concat_channels)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)

        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)

        if x.shape[2:] != skip_connection.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="nearest")

        x = torch.cat([x, skip_connection], dim=1)
        x = self.bn(x)
        return x


class LabelFreeUNet(nn.Module):
    """3D UNet for predicting fluorescence from transmitted light images.

    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        z_patch_size: Z dimension of input patches, affects 2D/3D kernel selection
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        z_patch_size: int = 16,
    ):
        super().__init__()

        f = [3, 16, 64, 128, 256]

        def get_kernels(
            make_2d: bool,
            base_k: tuple[int, int, int] = (3, 3, 3),
            base_p: tuple[int, int, int] = (2, 2, 2),
            base_s: tuple[int, int, int] = (2, 2, 2),
        ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
            if make_2d:
                k = (1, base_k[1], base_k[2])
                p = (1, base_p[1], base_p[2])
                s = (1, base_s[1], base_s[2])
                return k, p, s
            return base_k, base_p, base_s

        # Down blocks
        k1, p1, s1 = (1, 3, 3), (1, 2, 2), (1, 2, 2)
        self.down1 = ConvBlock(
            in_channels, f[1], n_conv=2, kernel_size=k1, pool_size=p1, pool_stride=s1
        )

        k2, p2, s2 = get_kernels(z_patch_size <= 4)
        self.down2 = ConvBlock(
            f[1], f[2], n_conv=2, kernel_size=k2, pool_size=p2, pool_stride=s2
        )

        k3, p3, s3 = get_kernels(z_patch_size <= 2)
        self.down3 = ConvBlock(
            f[2], f[3], n_conv=2, kernel_size=k3, pool_size=p3, pool_stride=s3
        )

        k4, p4, s4 = get_kernels(z_patch_size <= 1)
        self.down4 = ConvBlock(
            f[3], f[4], n_conv=2, kernel_size=k4, pool_size=p4, pool_stride=s4
        )

        # Up blocks
        uk1, _, us1 = get_kernels(z_patch_size <= 1)
        self.up1 = DeconvBlock(f[4], f[4], f[4], n_conv=2, kernel_size=uk1, stride=us1)

        uk2, _, us2 = get_kernels(z_patch_size <= 2)
        self.up2 = DeconvBlock(
            f[4] + f[4], f[3], f[3], n_conv=2, kernel_size=uk2, stride=us2
        )

        uk3, _, us3 = get_kernels(z_patch_size <= 4)
        self.up3 = DeconvBlock(
            f[3] + f[3], f[2], f[2], n_conv=2, kernel_size=uk3, stride=us3
        )

        uk4, _, us4 = (1, 3, 3), (1, 2, 2), (1, 2, 2)
        self.up4 = DeconvBlock(
            f[2] + f[2], f[1], f[1], n_conv=2, kernel_size=uk4, stride=us4
        )

        # Final layers
        self.final_bn = nn.BatchNorm3d(f[1] + f[1])
        self.final_conv = nn.Conv3d(f[1] + f[1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, Z, Y, X)

        Returns:
            Output tensor of shape (B, C, Z, Y, X)
        """
        # Encoder
        x1_pool, x1_skip = self.down1(x)
        x2_pool, x2_skip = self.down2(x1_pool)
        x3_pool, x3_skip = self.down3(x2_pool)
        x4_pool, x4_skip = self.down4(x3_pool)

        # Decoder
        u1 = self.up1(x4_pool, x4_skip)
        u2 = self.up2(u1, x3_skip)
        u3 = self.up3(u2, x2_skip)
        u4 = self.up4(u3, x1_skip)

        # Output
        out = self.final_bn(u4)
        out = self.final_conv(out)
        out = F.relu(out)

        return out


if __name__ == "__main__":
    model = LabelFreeUNet(z_patch_size=16)
    x = torch.randn(1, 1, 16, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
