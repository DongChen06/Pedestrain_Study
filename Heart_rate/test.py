import torch
import torch.nn as nn
from torchvision import models, transforms

class VehicleClassifier(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_classes):
        super(VehicleClassifier, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)
        # Remove the last fully connected layer (classifier)
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(512, hidden_dim, num_layers, batch_first=True)
        # Final classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Reshape to process frames individually
        x = x.view(batch_size * timesteps, C, H, W)
        # Feature extraction through ResNet18
        x = self.features(x)
        # Reshape to (batch_size, timesteps, output_size)
        x = x.view(batch_size, timesteps, -1)
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Only use the last output of the LSTM
        x = lstm_out[:, -1, :]
        # Classification
        x = self.fc(x)
        return x

# Model parameters
hidden_dim = 256
num_layers = 2
num_classes = 2 # Parked vs Moving

# Initialize the model
model = VehicleClassifier(hidden_dim, num_layers, num_classes)

# Example input tensor (batch_size, timesteps, C, H, W)
# For simplicity, using random data
input_tensor = torch.rand(5, 4, 3, 224, 224) # e.g., 5 sequences, each with 4 frames of size 224x224 with 3 color channels

# Forward pass
output = model(input_tensor)
print(output)
