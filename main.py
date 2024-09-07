from src.data_collection import scrape_brands, download_images
from src.data_augmentation import augment_images
from src.model_training import train_model
from src.model_testing import evaluate_model, cross_validate

def main():
    # Data Collection
    scrape_brands('https://en.wikipedia.org/wiki/Category:Home_appliance_brands', 'data/raw/brands.txt')
    # Define image_urls based on scraping
    # download_images(image_urls, 'data/raw/images/')
    
    # Data Augmentation
    augment_images('data/processed/train/', 'data/augmented/train/')
    augment_images('data/processed/val/', 'data/augmented/val/')
    augment_images('data/processed/test/', 'data/augmented/test/')
    
    # Model Training
    train_paths = []  # Populate with paths to training images
    train_labels = [] # Populate with corresponding labels
    train_model(train_paths, train_labels)
    
    # Model Testing
    test_paths = []  # Populate with paths to test images
    test_labels = [] # Populate with corresponding labels
    evaluate_model(test_paths, test_labels)
    cross_validate(train_paths, train_labels)

if __name__ == "__main__":
    main()
