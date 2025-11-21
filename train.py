import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import CaSAR, H2OContactDataset, H2OSkeletonDataset, focal_loss


def load_hyperparameters(filepath):
    with open(filepath, "r") as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


def train_f(model, dataloader, optimizer, scheduler, device, num_epochs=100):
    print("Train Contact Aware Module")
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            optimizer.zero_grad()

            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            loss = focal_loss(outputs, targets, alpha=0.5, gamma=4)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        scheduler.step()
        print(f"Epoch {epoch+1} Avg Loss: {running_loss / len(dataloader):.4f}")


def train_g(model, dataloader, optimizer, scheduler, device, num_epochs=600):
    print("Train Action Recognition Module")
    model.train()

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            optimizer.zero_grad()

            _, action_logits = model(inputs)

            loss = criterion(action_logits, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        scheduler.step()
        print(f"Epoch {epoch+1} Avg Loss: {running_loss / len(dataloader):.4f}")


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            _, outputs = model(inputs)

            # targets is one-hot, convert to class index
            target_class = torch.argmax(targets, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)

            total += targets.size(0)
            correct += (predicted_class == target_class).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    config_path = "config.yaml"
    hyperparameters = load_hyperparameters(config_path)

    print(hyperparameters)

    DATA_PATH = hyperparameters["data"]["path"]
    BATCH_SIZE = hyperparameters["data"]["batch_size"]
    NUM_WORKERS = hyperparameters["data"]["num_workers"]

    NUM_FRAMES = hyperparameters["model"]["num_frames"]
    NUM_CLASSES = hyperparameters["model"]["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Load Train Contact Dataset")
    train_contact_dataset = H2OContactDataset(DATA_PATH, split="train")
    train_contact_loader = DataLoader(
        train_contact_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    print("Load Train Skeleton Dataset")
    train_skeleton_dataset = H2OSkeletonDataset(DATA_PATH, split="train", N=NUM_FRAMES)
    train_skeleton_loader = DataLoader(
        train_skeleton_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    print("Load Val Skeleton Dataset")
    val_skeleton_dataset = H2OSkeletonDataset(DATA_PATH, split="val", N=NUM_FRAMES)
    val_skeleton_loader = DataLoader(
        val_skeleton_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CaSAR(num_frames=NUM_FRAMES, num_classes=NUM_CLASSES).to(device)

    f = model.contact_aware_module
    g = model.action_recognition_module

    optimizer_f = optim.Adam(
        f.parameters(), lr=hyperparameters["training"]["optimizer_f"]["lr"]
    )
    optimizer_g = optim.Adam(
        g.parameters(), lr=hyperparameters["training"]["optimizer_g"]["lr"]
    )

    scheduler_f = optim.lr_scheduler.StepLR(
        optimizer_f,
        step_size=hyperparameters["training"]["scheduler_f"]["step_size"],
        gamma=hyperparameters["training"]["scheduler_f"]["gamma"],
    )
    scheduler_g = optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=hyperparameters["training"]["scheduler_g"]["step_size"],
        gamma=hyperparameters["training"]["scheduler_g"]["gamma"],
    )

    for param in g.parameters():
        param.requires_grad = False
    train_f(f, train_contact_loader, optimizer_f, scheduler_f, device)

    for param in f.parameters():
        param.requires_grad = False
    for param in g.parameters():
        param.requires_grad = True
    train_g(model, train_skeleton_loader, optimizer_g, scheduler_g, device)

    accuracy = evaluate(model, val_skeleton_loader, device)
    print(f"Validation Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "casar_model.pth")


if __name__ == "__main__":
    main()
