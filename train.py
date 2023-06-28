import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchinfo

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

from models import LPC_V1

from utils import export_onnx

def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--train_path', default="./data_total/train", help='train folder path', type=str)
    parser.add_argument('--val_path', default="./data_total/val", help='val folder path', type=str)
    parser.add_argument('--model_path', default=None, help='model file path', type=str)
    parser.add_argument('--output_path', default="./models/model.pth", help='output model file path', type=str)
    parser.add_argument('--lr', default=0.00001, help='learning rate', type=float)
    parser.add_argument('--batch_size', default=64, help='batch size', type=int)
    parser.add_argument('--epoch', default=10, help='num epoch', type=int)
    parser.add_argument('--size', default=75, help='input size', type=int)
    parser.add_argument('--export_onnx', default=False, help='export ONNX model', type=bool)
    args = parser.parse_args()

    return args

def train(dataloader, model, loss_fn, optimizer, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val(dataloader, model, loss_fn, device="cuda"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

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


    # DEFINE DATALOADER FOR CUSTOM DATASET
    transforms_img = transforms.Compose([
                                        transforms.Resize(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]   )
                                        ])

    # DATASET FROM DIRECTORY
    train_data = torchvision.datasets.ImageFolder(root=args.train_path, transform=transforms_img)
    val_data = torchvision.datasets.ImageFolder(root=args.val_path, transform=transforms_img)
    # test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transforms_img)

    # DATALOADER (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


    # MODEL: LPC_V1
    model = LPC_V1(num_classes=4).to(device)
    if args.model_path != None:
        model = torch.load(args.model_path)
    print(torchinfo.summary(model, input_size=(1, 3, args.size, args.size)))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epoch

    for t in range(epochs):
        print(f"EPOCH {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        val(val_loader, model, loss_fn)

    if os.path.exists(args.output_path) == False:
        os.makedirs(args.output_path)
    torch.save(model, args.output_path)
    print("DONE! SAVED MODEL TO {}", args.output_path)

    if args.export_onnx == True:
        output_path = (args.output_path).replace(".pth", ".onnx")
        export_onnx(model, [1, 3, args.size, args.size], device=device, output_path=output_path)

if __name__ == '__main__':
    main()
