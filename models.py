import torch
from torch import nn
import torch.nn.functional as F

import torchinfo

# Creating a CNN class
class LPC_V1(nn.Module):
    def __init__(self, num_classes):
        super(LPC_V1, self).__init__()

        # Entry block
        self.conv_1 = nn.Sequential(
                                    # Layer 1
                                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(32)
                                )

        self.conv_2 = nn.Sequential(
                                    # Layer 2
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64)
                                )

        self.seperated_conv_1 = nn.Sequential(
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(32)
                                            )

        self.seperated_conv_2 = nn.Sequential(
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(32),
                                            )
        self.residual_1 = nn.Conv2d(in_channels=64, out_channels=32, stride=2, kernel_size=1, padding=0)

        self.seperated_conv_3 = nn.Sequential(
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(64)
                                            )

        self.seperated_conv_4 = nn.Sequential(
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(64)
                                            )
        self.residual_2 = nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=1, padding=0)

        self.seperated_conv_5 = nn.Sequential(
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(128)
                                            )

        self.seperated_conv_6 = nn.Sequential(
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128),
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(128)
                                    )

        self.residual_3 = nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=1, padding=0)

        self.seperated_conv_7 = nn.Sequential(
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(256)
                                    )

        self.block_output = nn.Sequential(
                                    nn.Dropout(p=0.5),
                                    nn.Linear(256, num_classes),
                                    nn.Softmax(dim=1)
                                        )

    def forward(self, x):
        # Entry Block
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        # Start shortcuts
        ## Block 1
        previous_block_activation = x
        x = self.seperated_conv_1(x)
        x = F.relu(x)
        x = self.seperated_conv_2(x)

        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)

        x += self.residual_1(previous_block_activation)

        previous_block_activation = x

        ## Block 2
        x = F.relu(x)
        x = self.seperated_conv_3(x)
        x = F.relu(x)
        x = self.seperated_conv_4(x)

        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)

        x += self.residual_2(previous_block_activation)

        previous_block_activation = x

        x = F.relu(x)
        x = self.seperated_conv_5(x)
        x = F.relu(x)
        x = self.seperated_conv_6(x)

        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)

        x += self.residual_3(previous_block_activation)
        previous_block_activation = x

        x = self.seperated_conv_7(x)
        x = F.relu(x)

        x = (F.adaptive_avg_pool2d(x, output_size=(1,1)))
        x = x.reshape(x.shape[0], -1)

        x = self.block_output(x)

        return x


# def main():
#     model = LPC_V1(num_classes=4).to(device)
#     print(summary(model, input_size=(1, 3, 75, 75)))

# if __name__ == '__main__':
#     main()