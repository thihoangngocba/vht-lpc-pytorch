import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

from models_multitask import LPC_Multitask_Net

from utils import export_onnx

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((75,75)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

# DATASET
class LPC_Multitask_Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir,
                 num_plate_classes=4, num_char_classes=2,
                 transform=TRANSFORM_IMG, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        # Set Inputs and Labels
        self.image_paths = []
        self.plate_colors = []
        self.char_colors = []

        self.num_plate_classes = num_plate_classes
        self.num_char_classes = num_char_classes

        df = pd.read_csv(annotations_file)
        for i in range(len(df)):
            image_path = os.path.join(img_dir, df.iloc[i,1])
            self.image_paths.append(image_path)
            self.plate_colors.append(df.iloc[i,2])
            self.char_colors.append(df.iloc[i,3])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load an Image
        image = Image.open(self.image_paths[index]).convert('RGB')
        # image = read_image(self.image_paths[index])
        # Transform it
        image = self.transform(image)
        # Get the Labels
        plate_color = F.one_hot(torch.tensor(self.plate_colors[index]), self.num_plate_classes)
        # char_color = F.one_hot(torch.tensor(self.char_colors[index]), self.num_char_classes)

        # plate_color = self.plate_colors[index]
        char_color = self.char_colors[index]
        # Return the sample of the dataset
        sample = {'image':image, 'plate_color': plate_color, 'char_color': char_color}

        return sample
      
def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--train_path', default="./data_total/train", help='train folder path', type=str)
    parser.add_argument('--val_path', default="./data_total/val", help='val folder path', type=str)
    parser.add_argument('--model_path', default=None, help='model file path', type=str)
    parser.add_argument('--output_path', default="./models/model.h5", help='output model file path', type=str)
    parser.add_argument('--lr', default=0.00001, help='learning rate', type=float)
    parser.add_argument('--batch_size', default=64, help='batch size', type=int)
    parser.add_argument('--epoch', default=10, help='num epoch', type=int)
    parser.add_argument('--size', default=75, help='input size', type=int)
    parser.add_argument('--export_onnx', default=True, help='export ONNX model', type=bool)
    args = parser.parse_args()

    return args

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader):
        inputs = data["image"].to(device=device)

        label_plate = data["plate_color"].to(device=device)
        label_char = data["char_color"].to(device=device)

        pred_plate, pred_char = model(inputs)
        # print(label_plate, label_char)
        # print(pred_plate, pred_char)
        # Loss functions
        loss_1 = loss_fn[0](pred_plate, label_plate.float())
        loss_2 = loss_fn[1](pred_char, label_char.float())
        loss = 0.9*loss_1 + 0.1*loss_2

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(label_plate)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(loss_1.item(), loss_2.item())

def val(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss_1, val_loss_2, val_loss, correct_1, correct_2 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data["image"].to(device=device)

            label_plate = data["plate_color"].to(device=device)
            label_char = data["char_color"].to(device=device)

            pred_plate, pred_char = model(inputs)

            # Loss functions
            val_loss_1 = loss_fn[0](pred_plate, label_plate.float())
            val_loss_2 = loss_fn[1](pred_char, label_char.float())
            val_loss = val_loss_1 + val_loss_2

            # print(label_char, pred_char.round())

            correct_1 += (pred_plate.argmax(1) == label_plate.argmax(1)).type(torch.float).sum().item()
            correct_2 += (pred_char.round() == label_char).float().sum().item()
            # print(correct_1, correct_2)

    val_loss /= num_batches
    correct_1 /= size
    correct_2 /= size

    print("\nVALIDATION:")
    print(f"Plate Color Accuracy: {(100*correct_1):>0.1f}%, loss_1: {val_loss_1:>8f}")
    print(f"Character Color Accuracy: {(100*correct_2):>0.1f}%, loss_2: {val_loss_2:>8f}")
    print(f"Val Loss: {val_loss:>8f} \n")
      
def main():
    # PARSER
    args = parse_args()

    input_size = (args.size, args.size)
    batch_size = args.batch_size

    # DEFINE DEVICE
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load dataset paths and CSV files
    train_dir = args.train_path
    val_dir = args.val_path

    train_csv_path = os.path.join(train_dir, "train.csv")
    val_csv_path = os.path.join(val_dir, "val.csv")
    
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225] )
        ])

    train_data = LPC_Multitask_Dataset(annotations_file=train_csv_path, img_dir=train_dir, transform=TRANSFORM_IMG)
    val_data = LPC_Multitask_Dataset(annotations_file=val_csv_path, img_dir=val_dir, transform=TRANSFORM_IMG)
    
    # DATALOADER (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2
                                               )
    
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2)
    
    # MODEL: LPC_V1
    model = LPC_Multitask_Net().to(device)
    if args.model_path != None:
        model = torch.load(args.model_path)
    print(torchinfo.summary(model, input_size=(1, 3, args.size, args.size)))

    # Define loss functions for each task
    plate_loss = nn.CrossEntropyLoss()# Includes Softmax
    char_loss = nn.BCELoss() # Doesn't include Softmax
    loss_fn = [plate_loss, char_loss]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epoch

    for t in range(epochs):
        print(f"EPOCH {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device=device)
        val(val_loader, model, loss_fn, device=device)
    
    output_dir = args.output_path.replace((args.output_path).split("/")[-1], "")
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    torch.save(model, args.output_path)
    print("DONE! SAVED MODEL TO {}", args.output_path)

    if args.export_onnx == True:
        output_path = (args.output_path).replace(".pth", ".onnx")
        export_onnx(model, [1, 3, args.size, args.size], device=device, output_path=output_path)

if __name__ == '__main__':
    main()
