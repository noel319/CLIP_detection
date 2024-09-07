import os
from PIL import Image
import torchvision.transforms as transforms
import shutil

def augment_images(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path)
        augmented_img = augmentation_transforms(img)
        augmented_img_path = os.path.join(output_folder, img_file)
        augmented_img.save(augmented_img_path)

if __name__ == "__main__":
    augment_images('data/processed/train/', 'data/augmented/train/')
    augment_images('data/processed/val/', 'data/augmented/val/')
    augment_images('data/processed/test/', 'data/augmented/test/')
