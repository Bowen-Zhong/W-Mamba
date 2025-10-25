import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultimodalLayer(nn.Module):
    """
    Gated Multimodal Layer (Conv Version)
    Adapted for 4D inputs: (batch_size, channels, height, width)

    Fuses two feature maps with learned gates.
    Based on 'Gated Multimodal Networks' - Arevalo et al. (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, channels_in1, channels_in2, channels_out=16):
        super(GatedMultimodalLayer, self).__init__()
        self.channels_in1 = channels_in1
        self.channels_in2 = channels_in2
        self.channels_out = channels_out

        # Use 1x1 convolutions instead of Linear layers
        self.conv1 = nn.Conv2d(channels_in1, channels_out, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(channels_in2, channels_out, kernel_size=1, bias=True)

        # Gate computation: input is concatenated along channel dimension
        self.gate_conv = nn.Conv2d(channels_out * 2, 1, kernel_size=1, bias=True)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor (batch_size, channels_in1, height, width)
            x2: Tensor (batch_size, channels_in2, height, width)
        Returns:
            Fused tensor (batch_size, channels_out, height, width)
        """
        h1 = self.tanh_f(self.conv1(x1))  # (batch_size, channels_out, H, W)
        h2 = self.tanh_f(self.conv2(x2))  # (batch_size, channels_out, H, W)

        concat = torch.cat((h1, h2), dim=1)  # (batch_size, 2*channels_out, H, W)

        z = self.sigmoid_f(self.gate_conv(concat))  # (batch_size, 1, H, W)

        fused = z * h1 + (1 - z) * h2  # Broadcast z over channel dimension

        return fused


# === Example usage ===
if __name__ == "__main__":
    input1 = torch.rand(1, 112, 256, 256)
    input2 = torch.rand(1, 112, 256, 256)

    # Initialize the module
    gml = GatedMultimodalLayer(channels_in1=112, channels_in2=112, channels_out=224)

    # Forward pass
    output = gml(input1, input2)

    print("Output shape:", output.shape)  # Should print (1, 32, 64, 64)
