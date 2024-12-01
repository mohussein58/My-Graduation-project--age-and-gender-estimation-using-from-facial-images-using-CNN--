from torchvision import transforms

def lenet_transform(size=32):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.5]),
    ])

def alexnet_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])