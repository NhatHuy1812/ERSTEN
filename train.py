import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from pathlib import Path
import math
import random
from model import ERSTEN  # Import the updated model

# Hyperparameters
beta = 1e-2
gamma = 1e-1
mix_ratio = 0.5
feature_channel = 6
show_pic = 0
show_batch = 0
save_epoch = 2
num_classes = 43  # Number of classes in the dataset

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = {
    "lr": 1e-4,
    "batch_size": 64,
    "epochs": 50,
    "resume": None,  # Path to resume a saved model
    "dataset": "traffic_signs",
}
root_results = "./results"
outimg_path = os.path.join(root_results, "output_images")
result_path = os.path.join(root_results, "logs")

# Dataset loaders (assume `trainloader` and `testloader` are predefined)
# Define your `trainloader`, `testloader`, `tr_loader`, and `te_loader` here
# Example:
# trainloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
# testloader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

# Model definition
input_size = 64  # Assume images are 30x30 pixels
extract_chn = [64, 128, 256, 256, 128, 64]
nc = 3  # RGB input

# Initialize model
net = ERSTEN(nc=nc, input_size=input_size, extract_chn=extract_chn, num_classes=num_classes).to(device)

# Resume from checkpoint if specified
if args["resume"] is not None:
    net = torch.load(args["resume"])

# Loss functions
match_loss = nn.MSELoss(reduction="mean")

def loss_class_func(out, target):
    return F.cross_entropy(out, target)

def loss_match_func(feat_sem, temp_sem):
    return match_loss(feat_sem, temp_sem)

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=args["lr"])

def train(epoch):
    net.train()
    print(f"Start training epoch: {epoch}")
    for i, (input, target, template) in enumerate(trainloader):
        optimizer.zero_grad()
        target = target.to(device)
        input, template = input.to(device), template.to(device)

        # Extract features
        feat_sem, feat_illu = net.extract(input, is_warping=True)
        temp_sem, _ = net.extract(template, is_warping=False)

        # Decode features
        recon_feat_sem = net.decode(feat_sem)
        recon_temp_sem = net.decode(temp_sem)

        # Classification
        feature_exc = torch.cat((feat_sem, feat_illu), 1)
        out_exc = net.classify(feature_exc)
        loss_class = loss_class_func(out_exc, target)

        # Calculate other losses
        loss_match = loss_match_func(feat_sem, temp_sem)

        # Total loss
        loss = loss_class + beta * loss_match 
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

def test(epoch, best_acc):
    net.eval()
    print(f"Start testing epoch: {epoch}")
    correct, total = 0, 0
    with torch.no_grad():
        for i, (input, target, template) in enumerate(testloader):
            target = target.to(device)
            input, template = input.to(device), template.to(device)

            # Extract features
            feat_sem, feat_illu = net.extract(input, is_warping=True)
            temp_sem, _ = net.extract(template, is_warping=False)

            # Decode and classify
            feature = torch.cat((feat_sem, feat_illu), 1)
            out = net.classify(feature)
            _, pred = torch.max(out, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")
    if accuracy > best_acc:
        best_acc = accuracy
        save_path = os.path.join(root_results, "saved_models", f"{args['dataset']}_best.pth")
        torch.save(net.state_dict(), save_path)
    return best_acc

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(1, args["epochs"] + 1):
        train(epoch)
        best_acc = test(epoch, best_acc)
        print(f"Best accuracy so far: {best_acc:.4f}")
