import os
import pandas as pd

# Folder containing images
image_folder = "data/processed/train"


# Get all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Create data for the CSV
data = {
    'image': [file for file in image_files],  # Construct the full URL
    'caption': [os.path.splitext(file)[0] for file in image_files]  # Use file name as caption without extension
}

# Create DataFrame
df = pd.DataFrame(data)
df_repeated = pd.concat([df] * 15, ignore_index=True)
df_repeated['id'] = range(1, len(df_repeated) + 1)
# Save to CSV
csv_file_path = "data/processed/train/captions.csv"  # You can set the path where you want to save your CSV file
df_repeated.to_csv(csv_file_path, index=False)

print(f"CSV file created at {csv_file_path} data: {df}")
