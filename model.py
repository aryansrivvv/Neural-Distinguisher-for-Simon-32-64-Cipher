import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# Component 1: Squeeze-and-Excitation (SE) Block [cite: 588]
# ---------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Global Average Pooling is implicitly done if we reduce to (Batch, Channel, 1)
        # However, for 1D data of size L, we pool over L.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Two Dense layers (FC) to calculate attention weights [cite: 632]
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # Output weights between 0 and 1 [cite: 633]
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Calculate weights
        y = self.fc(y).view(b, c, 1)
        # Scale: Multiply input features by weights [cite: 634]
        return x * y.expand_as(x)

# ---------------------------------------------------------
# Component 2: Residual Block with Attention (Module 2) [cite: 588]
# ---------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResNetBlock, self).__init__()
        padding = kernel_size // 2

        # First Conv1D layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second Conv1D layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE Block embedded in the residual unit [cite: 590]
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Attention
        out = self.se(out)

        # Residual Connection
        out += residual
        out = self.relu(out)
        return out

# ---------------------------------------------------------
# Main Network: SimonDistinguisher [cite: 482]
# ---------------------------------------------------------
class SimonDistinguisher(nn.Module):
    def __init__(self, input_size=64, num_filters=64, num_blocks=3):
        super(SimonDistinguisher, self).__init__()

        # --- Module 1: Input Preprocessing  ---
        # Input is (Batch, 64). We reshape to (Batch, 1, 64) for Conv1D.
        self.conv_in = nn.Conv1d(1, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU(inplace=True)

        # --- Module 2: Stack of Residual Blocks [cite: 588] ---
        # The paper uses multiple blocks. We'll use 'num_blocks' (e.g., 3 or 4)
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNetBlock(num_filters, num_filters))
        self.resnet_stack = nn.Sequential(*layers)

        # --- Module 3: Prediction Head  ---
        # Flatten the output of the Conv blocks
        self.flatten_dim = num_filters * input_size

        self.dense1 = nn.Linear(self.flatten_dim, 128) # Dense d=128 [cite: 583]
        self.bn_dense1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5) # Regularization [cite: 584]

        self.dense2 = nn.Linear(128, 64)
        self.bn_dense2 = nn.BatchNorm1d(64)

        self.output_head = nn.Linear(64, 1) # Output 1 value (probability)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, 64) -> Reshape to (Batch, 1, 64) for Conv1D
        x = x.view(x.size(0), 1, -1)

        # Module 1
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)

        # Module 2 (ResNet Stack)
        x = self.resnet_stack(x)

        # Module 3 (Head)
        x = x.view(x.size(0), -1) # Flatten

        x = self.dense1(x)
        x = self.bn_dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn_dense2(x)
        x = self.relu(x)

        x = self.output_head(x)
        return self.sigmoid(x) # Output probability

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate Model
    net = SimonDistinguisher(input_size=64).to(device)

    # Create Dummy Input (Batch=10, 64 bits)
    dummy_input = torch.randn(10, 64).to(device)

    # Forward Pass
    output = net(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Sample Output: {output[0].item()}")
    print("SUCCESS: Model architecture is valid.")
