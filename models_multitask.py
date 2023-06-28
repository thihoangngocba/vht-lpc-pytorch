import torch
from torch import nn
import torch.nn.functional as F

import torchinfo

# Creating a CNN class
class LPC_Multitask_Net(nn.Module):
  def __init__(self):
    super(LPC_Multitask_Net, self).__init__()

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

    self.output_1 = nn.Sequential(
                                  #nn.Flatten(),
                                  #nn.AvgPool2d(9),
                                  #nn.AdaptiveAvgPool2d(output_size=(1,1)),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(256, 4),
                                  nn.Softmax(dim=1)
                                  )

    self.output_2 = nn.Sequential(
                              #nn.Flatten(),
                              #nn.AvgPool2d(9),
                              #nn.AdaptiveAvgPool2d(output_size=(1,1)),
                              nn.Dropout(p=0.5),
                              nn.Linear(256, 1),
                              nn.Sigmoid()
                                  )


  def forward(self, x):
        # Entry Block
        #print(x.shape)
        x = self.conv_1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv_2(x)
        x = F.relu(x)
        #print(x.shape)


        # Start shortcuts
        ## Block 1
        previous_block_activation = x
        #print("flag1")
        x = self.seperated_conv_1(x)
        x = F.relu(x)
        #print(x.shape)
        #print("flag2")
        x = self.seperated_conv_2(x)
        #print(x.shape)

        #print("flag3")
        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)
        #print(x.shape)

        ## Add residual_1
        #print("flag4")
        x += self.residual_1(previous_block_activation)
        #print(x.shape)
        #x = x.view(x.size(0),-1)
        # print(x.shape)
        previous_block_activation = x

        ## Block 2
        x = F.relu(x)
        #print("flag5")
        x = self.seperated_conv_3(x)
        x = F.relu(x)
        #print(x.shape)
        #print("flag6")
        x = self.seperated_conv_4(x)
        #print(x.shape)
        #print("flag62")
        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)
        #print(x.shape)

        ## Add residual_2
        #print("flag7")
        x += self.residual_2(previous_block_activation)
        #print(x.shape)
        #x = x.view(x.size(0),-1)
        previous_block_activation = x

        ## Block 3
        #print("flag8")
        x = F.relu(x)
        x = self.seperated_conv_5(x)
        x = F.relu(x)
        #print(x.shape)
        #print("flag9")
        x = self.seperated_conv_6(x)
        #print(x.shape)

        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)
        #print(x.shape)

        ## Add residual_3
        #print("flag10")
        x += self.residual_3(previous_block_activation)
        #x = x.view(x.size(0),-1)
        previous_block_activation = x
        #print(x.shape)

        #print("flag11")
        x = self.seperated_conv_7(x)
        x = F.relu(x)
        #print(x.shape)

        #print("flag12")
        #x = F.flatten(x)
        #print(x.shape)
        x = (F.adaptive_avg_pool2d(x, output_size=(1,1)))
        x = x.reshape(x.shape[0], -1)
        #x = (F.adaptive_avg_pool2d(x, output_size=(1,1))).reshape(x.shape[0], -1)
        #x = torch.squeeze(x, (1,2,3))
        # print(x.shape)
        out_1 = self.output_1(x)
        out_2 = self.output_2(x)

        return out_1, out_2.flatten()
