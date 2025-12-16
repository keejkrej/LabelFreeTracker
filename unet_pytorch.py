import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_conv=2,
        kernel_size=(3, 3, 3),
        pool_size=(2, 2, 2),
        pool_stride=(2, 2, 2),
        dropout=False,
    ):
        super(ConvBlock, self).__init__()
        self.n_conv = n_conv
        self.dropout = dropout

        self.convs = nn.ModuleList()
        curr_in = in_channels
        for _ in range(n_conv):
            # TensorFlow: padding='same' -> PyTorch: padding = kernel_size // 2
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            self.convs.append(
                nn.Conv3d(curr_in, out_channels, kernel_size, padding=padding)
            )
            curr_in = out_channels

        self.pool = nn.MaxPool3d(
            kernel_size=pool_size, stride=pool_stride, padding=0, ceil_mode=True
        )  # padding='same' in TF for pool often acts like ceil_mode=True or manual padding.
        self.bn = nn.BatchNorm3d(
            out_channels
        )  # Applied after pooling in the original TF code?
        # Wait, TF code:
        # Loop Conv -> Dropout
        # to_concat = layer
        # Pool
        # BN
        # return layer, to_concat

        # So BN is applied to the pooled output.
        # But 'to_concat' is the UNPOOLED output (after convs).

    def forward(self, x):
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
    def __init__(
        self,
        in_channels,
        concat_channels,
        out_channels,
        n_conv=2,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        dropout=False,
    ):
        super(DeconvBlock, self).__init__()
        # TF logic:
        # 1. Conv3DTranspose (upsample)
        # 2. Loop Conv3D
        # 3. Concat with skip connection
        # 4. BN

        # Note on padding for ConvTranspose3d:
        # To match TF 'same' with stride 2 and kernel 3, we usually need padding=1, output_padding=1
        # But here kernel/stride can vary.

        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        output_padding = (stride[0] - 1, stride[1] - 1, stride[2] - 1)
        # Simplistic output_padding logic. If stride is 1, out_pad is 0. If stride is 2, out_pad is 1.
        # This assumes even spatial dims for stride 2.

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
            # Conv3D in the block has padding='same'
            p = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            self.convs.append(nn.Conv3d(curr_in, out_channels, kernel_size, padding=p))
            curr_in = out_channels

        self.bn = nn.BatchNorm3d(out_channels + concat_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)

        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            # Original code has dropout check inside loop, but passes dropout=False to up2/3/4 and only uses it (maybe) implicitly?
            # Looking at code: _deconv_block call for 'up4' has dropout=False. Others use default False.

        # Handle shape mismatch due to padding/odd dims if necessary
        # (Resize x to match skip_connection if needed, or crop skip_connection)
        if x.shape[2:] != skip_connection.shape[2:]:
            # Simple interpolation to match skip connection size if slight mismatch
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="nearest")

        x = torch.cat([x, skip_connection], dim=1)  # dim 1 is channels
        x = self.bn(x)
        return x


class LabelFreeUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, z_patch_size=16):
        super(LabelFreeUNet, self).__init__()

        # Configuration logic mapped from _network.py
        # filter_sizes = [3, 16, 64, 128, 256]
        f = [3, 16, 64, 128, 256]

        # Helper to determine 2D/3D kernels based on z_patch_size
        def get_kernels(make_2d, base_k=(3, 3, 3), base_p=(2, 2, 2), base_s=(2, 2, 2)):
            if make_2d:
                k = (1, base_k[1], base_k[2])
                p = (1, base_p[1], base_p[2])
                s = (1, base_s[1], base_s[2])
                return k, p, s
            return base_k, base_p, base_s

        # DOWN 1
        # kernel=(1, 3, 3), pool=(1, 2, 2). "make_2d" logic only affects if z<=1 which changes nothing for (1,x,x).
        # Actually in TF code:
        # down1: make_2d=(z<=1). Default kernel=(1,3,3), pool=(1,2,2).
        # If make_2d is True, kernel becomes (1,3,3) anyway.
        k1, p1, s1 = (1, 3, 3), (1, 2, 2), (1, 2, 2)
        self.down1 = ConvBlock(
            in_channels, f[1], n_conv=2, kernel_size=k1, pool_size=p1, pool_stride=s1
        )

        # DOWN 2
        # make_2d=(z<=4). Default k=(3,3,3), p=(2,2,2).
        k2, p2, s2 = get_kernels(z_patch_size <= 4)
        self.down2 = ConvBlock(
            f[1], f[2], n_conv=2, kernel_size=k2, pool_size=p2, pool_stride=s2
        )

        # DOWN 3
        # make_2d=(z<=2).
        k3, p3, s3 = get_kernels(z_patch_size <= 2)
        self.down3 = ConvBlock(
            f[2], f[3], n_conv=2, kernel_size=k3, pool_size=p3, pool_stride=s3
        )

        # DOWN 4
        # make_2d=(z<=1).
        k4, p4, s4 = get_kernels(z_patch_size <= 1)
        self.down4 = ConvBlock(
            f[3], f[4], n_conv=2, kernel_size=k4, pool_size=p4, pool_stride=s4
        )

        # UP 1 (corresponds to down4 reverse)
        # filters=256 (f[4]). input=down4 output (f[4]). concat=down4 skip (f[4]).
        # make_2d=(z<=1)
        uk1, _, us1 = get_kernels(z_patch_size <= 1)
        self.up1 = DeconvBlock(f[4], f[4], f[4], n_conv=2, kernel_size=uk1, stride=us1)

        # UP 2 (corresponds to down3 reverse)
        # filters=128 (f[3]). input=up1 output (f[4]+f[4]=512). concat=down3 skip (f[3]).
        # make_2d=(z<=2)
        uk2, _, us2 = get_kernels(z_patch_size <= 2)
        self.up2 = DeconvBlock(
            f[4] + f[4], f[3], f[3], n_conv=2, kernel_size=uk2, stride=us2
        )

        # UP 3 (corresponds to down2 reverse)
        # filters=64 (f[2]). input=up2 output (f[3]+f[3]=256). concat=down2 skip (f[2]).
        # make_2d=(z<=4)
        uk3, _, us3 = get_kernels(z_patch_size <= 4)
        self.up3 = DeconvBlock(
            f[3] + f[3], f[2], f[2], n_conv=2, kernel_size=uk3, stride=us3
        )

        # UP 4 (corresponds to down1 reverse)
        # filters=16 (f[1]). input=up3 output (f[2]+f[2]=128). concat=down1 skip (f[1]).
        # make_2d=(z<=1). kernel=(1,3,3), stride=(1,2,2)
        # In TF code: up4 uses kernel=(1,3,3), strides=(1,2,2) explicitly.
        uk4, _, us4 = (1, 3, 3), (1, 2, 2), (1, 2, 2)
        self.up4 = DeconvBlock(
            f[2] + f[2], f[1], f[1], n_conv=2, kernel_size=uk4, stride=us4
        )

        # Final
        # Input: up4 output (f[1]+f[1]=32)
        self.final_bn = nn.BatchNorm3d(f[1] + f[1])
        self.final_conv = nn.Conv3d(f[1] + f[1], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch, Channels, Z, Y, X)

        # Down
        x1_pool, x1_skip = self.down1(x)
        x2_pool, x2_skip = self.down2(x1_pool)
        x3_pool, x3_skip = self.down3(x2_pool)
        x4_pool, x4_skip = self.down4(x3_pool)

        # Up
        # up1 inputs: x4_pool (input), x4_skip (concat)
        u1 = self.up1(x4_pool, x4_skip)

        # up2 inputs: u1 (input), x3_skip (concat)
        u2 = self.up2(u1, x3_skip)

        # up3 inputs: u2 (input), x2_skip (concat)
        u3 = self.up3(u2, x2_skip)

        # up4 inputs: u3 (input), x1_skip (concat)
        u4 = self.up4(u3, x1_skip)

        # Final
        out = self.final_bn(u4)
        out = self.final_conv(out)
        out = F.relu(out)  # TF code has activation='relu' in final conv

        return out


if __name__ == "__main__":
    # Simple test
    model = LabelFreeUNet(z_patch_size=16)
    # (B, C, Z, Y, X)
    x = torch.randn(1, 1, 16, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
