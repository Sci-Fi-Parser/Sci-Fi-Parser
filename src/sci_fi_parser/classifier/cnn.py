import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, pool_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

    def forward(self, x):
        return self.block(x)

class CNNClassifier(nn.Module):

    def __init__(self, num_classes: int, channels_in: int, channels_out: list[int], 
                 conv_kernel_size: int, pool_size: int, linear_layer_neurons: int, input_image_size: int):
        super().__init__()
        self.features = nn.Sequential()
        for ch_out in channels_out:
            self.features.append(ConvBlock(channels_in, ch_out, conv_kernel_size, pool_size))
            channels_in = ch_out
        
        """
        Each convolutional block halves the input image size so the final size is the initial size divided by 2^(number of blocks)
        """
        feature_size = input_image_size // (2 ** len(channels_out))
        flatten_dim = channels_out[-1] * feature_size * feature_size #

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, linear_layer_neurons),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(linear_layer_neurons, num_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = CNNClassifier(num_classes=10, channels_in=3, channels_out=[32, 64, 128, 256],
                          conv_kernel_size=3, pool_size=2, linear_layer_neurons=512, input_image_size=256)

    dummy_input = torch.randn(8, 3, 256, 256)  # batch of 8 RGB images
    output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")          # (8, 10)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")