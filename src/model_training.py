import torch
import open_clip
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_paths, train_labels, num_epochs=10, batch_size=32, learning_rate=5e-6):
    # Load the pre-trained CLIP model
    model, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)
    # DataLoader setup
    train_dataset = datasets.ImageFolder('data/augmented/train/', transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=5e-6)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        for batch_images, batch_labels in train_loader:
            image_features = model.encode_image(batch_images)
            text_inputs = tokenizer(batch_labels, padding=True, return_tensors="pt").input_ids
            text_features = model.encode_text(text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = (image_features @ text_features.T) * 100
            labels = torch.arange(len(batch_images)).to(device)
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

if __name__ == "__main__":
    # Example data paths and labels
    train_paths = []  # Populate with paths to training images
    train_labels = [] # Populate with corresponding labels
    train_model(train_paths, train_labels)
