import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

#Configuration
batch_size = 64
num_classes = 65
# Paths to your data
train_csv = 'hw1_data/p1_data/office/train.csv'
train_img_dir = 'hw1_data/p1_data/office/train'
output_dir = 'p1/visualization_plots'

class OfficeHomeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # Load the CSV file
        self.img_dir = img_dir  # Directory where images are stored
        self.transform = transform  # Apply image transformations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the image file name and label from the CSV
        img_name = os.path.join(self.img_dir, self.annotations.iloc[index, 1])
        label = int(self.annotations.iloc[index, 2])
        
        # Load the image
        image = Image.open(img_name).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the data augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128),  # Randomly crop and resize to 128x128
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15),      # Randomly rotate by up to 15 degrees
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Random color adjustments
    transforms.ToTensor(),              # Convert image to a PyTorch tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # Normalize to ImageNet standards
])
# Create Dataset objects for training and validation
train_dataset = OfficeHomeDataset(csv_file=train_csv, img_dir=train_img_dir, transform=train_transform)

# Load your datasets (train dataset in this case)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Function to load model checkpoint and extract features from the second-to-last layer
def load_model_and_extract_features(checkpoint_path, train_loader, device):
    # Load model
    model = models.resnet50()
    # Modify the final layer for Office-Home classification (assume 65 classes)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.fc = nn.Identity()  # Replace the fully connected layer to extract features
    model = model.to(device)
    model.eval()

    features = []
    labels = []
    with torch.no_grad():
        for imgs, label in train_loader:
            imgs = imgs.to(device)
            output = model(imgs)  # Get the features from the second-to-last layer
            features.append(output.cpu().numpy())
            labels.append(label.numpy())

    # Concatenate all the features and labels
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels

# Function to apply t-SNE and PCA and plot the results
def visualize_features(features, labels, method='tsne', epoch='first'):
    if method == 'tsne':
        tsne = TSNE(n_components=2, init='pca', random_state=501)
        reduced_features = tsne.fit_transform(features)
    elif method == 'pca':
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

    # Plot the features in 2D space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='jet', s=5)
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} Visualization of Feature Embeddings')
    # Save the plot
    plot_filename = f'{method}_epoch_{epoch}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()  # Close the figure to avoid displaying

# Device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the features from the first epoch and the last epoch
features_epoch1, labels_epoch1 = load_model_and_extract_features('p1/checkpoints_fintune_full_no_norm/resnet50_best_epoch_1.pth', train_loader, device)
features_epoch_last, labels_epoch_last = load_model_and_extract_features('p1/checkpoints_fintune_full_no_norm/resnet50_best_epoch_92.pth', train_loader, device)

# Visualize using t-SNE and PCA for both epochs
print("t-SNE for First Epoch")
visualize_features(features_epoch1, labels_epoch1, method='tsne', epoch='first')
print("PCA for First Epoch")
visualize_features(features_epoch1, labels_epoch1, method='pca', epoch='first')

print("t-SNE for Last Epoch")
visualize_features(features_epoch_last, labels_epoch_last, method='tsne', epoch='last')
print("PCA for Last Epoch")
visualize_features(features_epoch_last, labels_epoch_last, method='pca', epoch='last')
