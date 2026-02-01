from torchvision import transforms


def get_transforms_train():
    """
    Get training data transformations with augmentation.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations for training data
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=3, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])


def get_transforms_test():
    """
    Get test/validation data transformations without augmentation.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations for test data
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
