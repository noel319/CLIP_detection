import requests
from bs4 import BeautifulSoup
import os
import urllib.request
from time import sleep
# Function to scrape brand names
def scrape_brands(url, output_file):
    for i in range(3):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Check if the request was successful
            soup = BeautifulSoup(response.text, 'html.parser')
            brand_names = [item.get_text() for item in soup.find_all('div', class_='mw-category-group')]
    
            with open(output_file, 'w') as f:
                for brand in brand_names:
                    f.write(f"{brand}\n")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i + 1} failed: {e}")
            if i < 3 - 1:
                sleep(5)  # Wait before retrying
    

# Function to download images (you need to define the image URLs or sources)
def download_images(image_urls, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, url in enumerate(image_urls):
        image_path = os.path.join(output_folder, f"image_{i}.jpg")
        urllib.request.urlretrieve(url, image_path)

if __name__ == "__main__":
    scrape_brands('https://en.wikipedia.org/wiki/Category:Home_appliance_brands', 'data/raw/brands.txt')
    # Example usage of download_images
    # download_images(image_urls, 'data/raw/images/')
