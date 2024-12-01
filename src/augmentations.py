from torchvision import transforms
from random import randint

def get_augs():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAdjustSharpness(sharpness_factor=randint(0,10)),
        transforms.GaussianBlur(randint(0, 20)*2+1),
        transforms.RandomRotation(randint(0, 10), fill=0),
        transforms.RandomPerspective(distortion_scale=0.2, fill=0),
        transforms.RandomGrayscale(p=0.5),
    ])