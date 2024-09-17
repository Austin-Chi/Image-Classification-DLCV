import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
import csv

# Configuration
logging.basicConfig(filename='p1_09170123fintune_full.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
checkpoint_dir = './p1/checkpoints_fintune_full_no_norm'
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = 'validation_log.csv'
backbone_path = 'p1/checkpoints/best_checkpoint.pth'
# Paths to your data
train_csv = 'hw1_data/p1_data/office/train.csv'
val_csv = 'hw1_data/p1_data/office/val.csv'
train_img_dir = 'hw1_data/p1_data/office/train'
val_img_dir = 'hw1_data/p1_data/office/val'

batch_size = 64
num_epochs = 100
learning_rate = 3e-4
num_classes = 65
image_size = (128, 128)
best_val_accuracy = 0.0  # Track the best validation accuracy
# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(device)

# Initialize the CSV file for logging validation accuracy
with open(log_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Accuracy', 'Validation Accuracy'])


# Function to save a checkpoint
def save_checkpoint(epoch, model, optimizer, val_accuracy, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)

# Validation function to calculate accuracy
def validate(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total  # Return validation accuracy


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

# No augmentation for validation set, just resize and normalize
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),      # Resize to 128x128
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset objects for training and validation
train_dataset = OfficeHomeDataset(csv_file=train_csv, img_dir=train_img_dir, transform=train_transform)
val_dataset = OfficeHomeDataset(csv_file=val_csv, img_dir=val_img_dir, transform=val_transform)

# DataLoader to iterate over batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Load the pre-trained ResNet50 backbone
resnet = models.resnet50(pretrained=False)
backbone = torch.load(backbone_path, map_location=device)
resnet.load_state_dict(backbone['model_state_dict'])

# Modify the final layer for Office-Home classification (assume 65 classes)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)

# Set up optimizer and loss function
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Full training loop
# Training loop (simplified)
# Training loop with validation and checkpoint saving
for epoch in range(num_epochs):  # Adjust the number of epochs as needed
    # Training phase
    resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = correct / total  # Calculate training accuracy

    # Validation phase
    val_accuracy = validate(resnet, val_loader, device)
    logging.info(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Save validation accuracy to CSV file
    with open(log_file, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_accuracy, val_accuracy])
    
    # Save the model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint_path = os.path.join(checkpoint_dir, f'resnet50_best_epoch_{epoch+1}.pth')
        save_checkpoint(epoch + 1, resnet, optimizer, val_accuracy, checkpoint_path)
        logging.info(f'Best model saved at epoch {epoch+1} with Val Accuracy: {val_accuracy:.4f}')
