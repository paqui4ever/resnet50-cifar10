import torch
import torch_directml
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.transforms import get_transforms_test


# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


def show_image(image, title=""):
    """Display a single image."""
    plt.figure(figsize=(3, 3))
    # Denormalize for display
    img = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and return accuracy.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        float: Test accuracy percentage
    """
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for test_inputs, test_targets in tqdm(test_loader, desc='Calculating test accuracy'):
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            test_outputs = model(test_inputs)
            _, predicted = torch.max(test_outputs, 1)
            correct_predictions += (predicted.cpu() == test_targets.cpu()).sum().item()
            total_predictions += test_targets.size(0)
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def visualize_predictions(model, test_ds, device, num_samples=5):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: The neural network model
        test_ds: Test dataset
        device: Device to run predictions on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    print(f"\nVisualizing {num_samples} random predictions...")
    indices = torch.randperm(len(test_ds))[:num_samples]
    
    for idx in indices:
        image, label = test_ds[idx]
        
        # Get prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            predicted_class = output.argmax(dim=1).item()
        
        # Display result
        true_label = CLASS_NAMES[label]
        predicted_label = CLASS_NAMES[predicted_class]
        is_correct = "✓" if predicted_class == label else "✗"
        
        title = f"{is_correct} True: {true_label}\nPred: {predicted_label}"
        show_image(image, title)
        
        print(f"Image {idx}: True={true_label}, Predicted={predicted_label} {is_correct}")


def main():
    # Set device
    print("Setting up device...")
    device = torch_directml.device()
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    transform_test = get_transforms_test()
    test_ds = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transform_test,
        download=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load trained model
    print("\nLoading trained model...")
    resnet50 = torchvision.models.resnet50(pretrained=False).to(device)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, 10).to(device)
    
    try:
        resnet50.load_state_dict(torch.load('resnet50_cifar10.pth'))
        print("Model loaded successfully from 'resnet50_cifar10.pth'")
    except FileNotFoundError:
        print("Error: Model file 'resnet50_cifar10.pth' not found!")
        print("Please run train.py first to train and save the model.")
        return
    
    # Evaluate model
    accuracy = evaluate_model(resnet50, test_loader, device)
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    # Visualize sample predictions
    visualize_predictions(resnet50, test_ds, device, num_samples=5)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
