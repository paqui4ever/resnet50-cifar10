"""
Training script for ResNet50 on CIFAR-10 dataset.

This script trains a ResNet50 model from scratch on the CIFAR-10 dataset 
with data augmentation and saves the trained model for later evaluation.
TensorBoard logging is enabled for monitoring training progress.
"""

import os
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.transforms import get_transforms_train, get_transforms_test


def main():
    # Set device
    print("Setting up device...")
    device = torch_directml.device()
    print(f"Using device: {device}")
    
    # Setup TensorBoard writer
    writer = SummaryWriter('runs/resnet50_cifar10')
    print("TensorBoard logging enabled. Run: tensorboard --logdir=runs")
    
    # Load transforms
    print("\nLoading data transformations...")
    transform_train = get_transforms_train()
    transform_test = get_transforms_test()
    
    # Load datasets
    print("Loading CIFAR-10 datasets...")
    full_train_ds = torchvision.datasets.CIFAR10(
        root="./data", 
        train=True, 
        transform=transform_train, 
        download=True
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", 
        train=False, 
        transform=transform_test, 
        download=True
    )
    
    # Split training data into train and validation sets
    print("Splitting training data (40k train, 10k validation)...")
    train_ds, val_ds = torch.utils.data.random_split(full_train_ds, [40000, 10000])
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=False
    )
    
    # Setup model (train from scratch - no pretrained weights)
    print("\nSetting up ResNet50 model (training from scratch)...")
    resnet50 = torchvision.models.resnet50(pretrained=False).to(device)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, 10).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss().to(device)
    num_epochs = 100
    optimizer = optim.SGD(resnet50.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    num_batches_per_epoch = len(train_loader)
    num_steps = num_batches_per_epoch * num_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 128")
    print(f"  Learning rate: 0.1 (initial)")
    print(f"  Optimizer: SGD with momentum=0.9, weight_decay=5e-4")
    print(f"  LR Scheduler: CosineAnnealingLR")
    print(f"  Pretrained weights: No (training from scratch)")
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    
    for epoch in range(num_epochs):
        resnet50.train(mode=True)
        epoch_loss = 0.0
        
        # Training phase
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = resnet50(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            
            # Log loss and learning rate
            epoch_loss += loss.item()
            current_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/LearningRate', current_lr, global_step)
            
            lr_scheduler.step()
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Evaluation phase
        resnet50.eval()
        
        # Validation accuracy
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_inputs, val_targets in tqdm(val_loader, desc='Validation accuracy'):
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = resnet50(val_inputs)
                _, predicted = torch.max(val_outputs, 1)
                correct_val += (predicted.cpu() == val_targets.cpu()).sum().item()
                total_val += val_targets.size(0)
        
        accuracy_val = (correct_val / total_val) * 100
        
        # Test accuracy
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for test_inputs, test_targets in tqdm(test_loader, desc='Test accuracy'):
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_outputs = resnet50(test_inputs)
                _, predicted = torch.max(test_outputs, 1)
                correct_test += (predicted.cpu() == test_targets.cpu()).sum().item()
                total_test += test_targets.size(0)
        
        accuracy_test = (correct_test / total_test) * 100
        
        # Training accuracy
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for train_inputs, train_targets in tqdm(train_loader, desc='Training accuracy'):
                train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
                train_outputs = resnet50(train_inputs)
                _, predicted = torch.max(train_outputs, 1)
                correct_train += (predicted == train_targets).sum().item()
                total_train += train_targets.size(0)
        
        accuracy_train = (correct_train / total_train) * 100
        
        # Log accuracies to TensorBoard
        writer.add_scalar('Accuracy/Train', accuracy_train, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy_val, epoch)
        writer.add_scalar('Accuracy/Test', accuracy_test, epoch)
        writer.add_scalar('Loss/Epoch', avg_epoch_loss, epoch)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Training accuracy: {accuracy_train:.2f}%")
        print(f"  Validation accuracy: {accuracy_val:.2f}%")
        print(f"  Test accuracy: {accuracy_test:.2f}%")
        print(f"  Learning rate: {current_lr:.6f}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Save model
    print("\nSaving trained model...")
    torch.save(resnet50.state_dict(), 'resnet50_cifar10.pth')
    print("Model saved as 'resnet50_cifar10.pth'")
    
    print("\nTraining complete!")
    print(f"Final test accuracy: {accuracy_test:.2f}%")
    print(f"View training logs with: tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
