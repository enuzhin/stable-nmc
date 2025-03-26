import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import wandb
from omegaconf import OmegaConf
import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as T

# Training function
def train_epoch(model, device, train_loader, optimizer, epoch, criterion, log_every,enable_noise_and_clamp = True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear gradients from the previous step

        if enable_noise_and_clamp:
            output = model.noisy_forward(data)
        else:
            output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights


        if (batch_idx+1) % log_every == 0:
            loss_val = loss.item()
            # Calculate batch accuracy
            pred = output.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            batch_accuracy = 100. * batch_correct / data.size(0)
            samples_processed = min((batch_idx+1) * train_loader.batch_size,len(train_loader.dataset))
            print(f'Train Epoch: {epoch} [{samples_processed}/{len(train_loader.dataset)}]  Loss: {loss_val:.6f}  Accuracy: {batch_accuracy:.2f}%')
            wandb.log({"train_loss": loss_val, "train_percentage_error": 100-batch_accuracy, "epoch": epoch})

    loss_val = loss.item()
    # Calculate batch accuracy
    pred = output.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
    batch_correct = pred.eq(target.view_as(pred)).sum().item()
    batch_accuracy = 100. * batch_correct / data.size(0)
    samples_processed = min((batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset))
    print(
        f'Train Epoch: {epoch} [{samples_processed}/{len(train_loader.dataset)}]  Loss: {loss_val:.6f}  Accuracy: {batch_accuracy:.2f}%')
    wandb.log({"train_loss": loss_val, "train_percentage_error": 100 - batch_accuracy, "epoch": epoch})




def train(model, lr, num_epochs, batch_size, num_workers, cfg,enable_noise_and_clamp = True):
    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_loader, test_loader = create_dataloaders(num_workers, batch_size, cfg.dataset)

    # Initialize WandB for logging
    run_name = f"{cfg.model.type}_scale{cfg.model.scale}_wmax{cfg.model.w_max}_noise{cfg.model.noise_spread}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="mnn", config=wandb_config,name=run_name,mode=cfg.save.wandb_mode)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_epoch(model, device, train_loader, optimizer, epoch, criterion,cfg.save.log_every,enable_noise_and_clamp)
        model.eval()
        (test_loss_clear,test_accuracy_clear),(test_loss_noisy,test_accuracy_noisy) = eval(model, device, test_loader, criterion,enable_noise_and_clamp)
        wandb.log({"epoch": epoch, "test_loss_clear": test_loss_clear, "test_percentage_error_clear": 100-test_accuracy_clear, "test_loss_noisy": test_loss_noisy, "test_percentage_error_noisy": 100-test_accuracy_noisy})

        # Save model every `save_every` epochs
        if epoch % cfg.save.save_every == 0:
            save_model(model, epoch, cfg.save.path, cfg.model.type)
    save_model(model, epoch, cfg.save.path, cfg.model.type)
    print("Training Finished!")


# Testing function
def eval(model, device, test_loader, criterion,enable_noise_and_clamp = True):
    model.eval()  # Set the model to evaluation mode
    test_loss_clear = 0
    correct_clear = 0
    test_loss_noisy = 0
    correct_noisy= 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_clear = model(data)
            if enable_noise_and_clamp:
                output_noisy = model.noisy_forward(data)
            else:
                output_noisy = model(data)
            test_loss_clear += criterion(output_clear, target).item()  # Sum up batch loss
            pred_clear = output_clear.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            correct_clear += pred_clear.eq(target.view_as(pred_clear)).sum().item()

            test_loss_noisy += criterion(output_noisy, target).item()  # Sum up batch loss
            pred_noisy = output_noisy.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            correct_noisy += pred_noisy.eq(target.view_as(pred_noisy)).sum().item()


    test_loss_clear /= len(test_loader.dataset)
    test_accuracy_clear = 100. * correct_clear / len(test_loader.dataset)

    test_loss_noisy /= len(test_loader.dataset)
    test_accuracy_noisy = 100. * correct_noisy / len(test_loader.dataset)
    print(
        f'Test Set:  Loss (clear): {test_loss_clear:.4f}, Accuracy (clear): {correct_clear}/{len(test_loader.dataset)} ({test_accuracy_clear:.2f}%)')
    print(
        f'Test Set:  Loss (noisy): {test_loss_noisy:.4f}, Accuracy (noisy): {correct_noisy}/{len(test_loader.dataset)} ({test_accuracy_noisy:.2f}%)')
    return (test_loss_clear,test_accuracy_clear),(test_loss_noisy,test_accuracy_noisy)

def save_model(model, epoch, save_path, model_name):
    """Saves the model to the specified path."""
    os.makedirs(save_path, exist_ok=True)
    filename = f"{model_name}_epoch{epoch}.pth"
    save_file = os.path.join(save_path, filename)
    torch.save(model.state_dict(), save_file)
    print(f"Model saved: {save_file}")


def find_latest_checkpoint(model_name, save_path):
    """Finds the latest saved model checkpoint."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Checkpoint directory {save_path} does not exist.")

    checkpoints = [f for f in os.listdir(save_path) if f.startswith(model_name) and f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for model {model_name} in {save_path}")

    # Sort by epoch number (assumes filename format: ModelName_epochX.pth)
    checkpoints.sort(key=lambda x: int(x.split("_epoch")[-1].split(".pth")[0]), reverse=True)
    latest_checkpoint = os.path.join(save_path, checkpoints[0])

    print(f"Loading model from: {latest_checkpoint}")
    return latest_checkpoint




def create_dataloaders(num_workers, batch_size, dataset):
    # Create an Albumentations pipeline
    albumentations_pipeline = A.Compose([
        # ElasticTransform without the unsupported alpha_affine parameter.
        A.ElasticTransform(alpha=34, sigma=4, p=0.5),
        # OpticalDistortion without the unsupported shift_limit parameter.
        A.OpticalDistortion(distort_limit=0.05, p=0.5),
        # RandomBrightnessContrast remains unchanged.
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # GaussNoise without the unsupported var_limit parameter.
        A.GaussNoise(p=0.5),
        # CoarseDropout without unsupported parameters.
        A.CoarseDropout(p=0.5),
        # Normalize using MNIST mean and standard deviation.
        A.Normalize(mean=(0.1307,), std=(0.3081,)),
        # Convert image to PyTorch tensor.
        ToTensorV2()
    ])

    # Full training transform combining torchvision and Albumentations.
    train_transforms = T.Compose([
        T.Pad(padding=4),  # Add 4 pixels of padding.
        T.RandomCrop(28),  # Randomly crop back to 28x28.
        T.RandomAffine(
            degrees=10,  # Rotate by ±10°.
            translate=(0.1, 0.1),  # Translate up to 10%.
            scale=(0.9, 1.1)  # Scale between 0.9 and 1.1.
        ),
        T.Lambda(lambda img: np.array(img)),  # Convert PIL image to numpy array.
        # Apply Albumentations augmentations:
        T.Lambda(lambda np_img: albumentations_pipeline(image=np_img)['image']),
        T.RandomErasing(p=0.5, scale=(0.02, 0.15))  # Additional Cutout effect.
    ])


    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
