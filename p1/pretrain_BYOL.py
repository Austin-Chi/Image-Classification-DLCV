import torch
import os
import logging
import csv
from byol_pytorch import BYOL
from torchvision import models
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

# Configuration
logging.basicConfig(filename='p1_09151127.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
checkpoint_dir = './p1/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
csv_file_path = os.path.join(checkpoint_dir, "training_losses.csv")
batch_size = 64
num_epochs = 100
learning_rate = 3e-4
image_size = (128, 128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(device)

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # List all files in the image directory
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image

# Data augmentation pipeline for BYOL (removing augmentation here)
base_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure images are resized to correct dimensions
    transforms.ToTensor(),
])

# Custom collate function to generate two views (augmentations) for each image
def custom_collate_fn(batch):
    # Batch contains transformed images, so apply augmentations to create views
    imgs = batch
    
    # Define the data augmentation pipeline
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert tensor images back to PIL images for augmentation
    pil_imgs = [transforms.ToPILImage()(img) for img in imgs]
    
    # Apply augmentation to generate two views
    views1 = [augment_transform(img) for img in pil_imgs]
    #views2 = [augment_transform(img) for img in pil_imgs]

    
    return torch.stack(views1)#, torch.stack(views2)

# Load Mini-ImageNet dataset
# Create dataset
train_dataset = UnlabeledImageDataset(image_dir='hw1_data/p1_data/mini/train', transform=base_transform)

# Create DataLoader with the custom collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Create ResNet50 backbone (without pretrained weights)
resnet = models.resnet50(pretrained=False)

learner = BYOL(
    resnet,
    image_size=image_size[0],
    hidden_layer = 'avgpool'
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr=learning_rate)

# Initialize variables for checkpointing
best_loss = float('inf')
epoch_losses = []  # To track the loss per epoch

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, file_name="best_checkpoint.pth"):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    file_path = os.path.join(checkpoint_dir, file_name)
    torch.save(state, file_path)
    logging.info(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f}")

# Function to save the loss per epoch to a CSV file
def save_losses_to_csv(losses, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        for epoch, loss in enumerate(losses, 1):
            writer.writerow([epoch, loss])
    print(f"Training losses saved to {file_path}")


# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for imgs in train_loader:
        imgs = imgs.to(device)

        # Forward pass and compute BYOL loss
        try:
            loss = learner(imgs)
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            continue

        # Zero gradients, backward pass, and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    # Average loss for the epoch
    epoch_loss /= len(train_loader)
    epoch_losses.append(epoch_loss)  # Save loss for this epoch

    logging.info(f'Epoch {epoch+1}: Loss = {epoch_loss:.4f}')

    # Save checkpoint if the current loss is the best so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_checkpoint(resnet, opt, epoch, epoch_loss, checkpoint_dir)

# Save the backbone
torch.save(resnet.state_dict(), 'pretrained_resnet50_byol.pth')
# Save all epoch losses to CSV
save_losses_to_csv(epoch_losses, csv_file_path)