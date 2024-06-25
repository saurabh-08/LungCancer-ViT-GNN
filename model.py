import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import dgl
from dgl.nn import GATConv
from transformers import ViTModel, ViTConfig
from time import time
from tqdm import tqdm

os.environ['DGLBACKEND'] = 'pytorch'


class LungDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def preprocess_data(image_dir, label_dir, image_size=(128, 128), num_samples=None, depth_size=128):
    start_time = time()
    images = []
    labels = []
    filenames = [f for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
    
    if num_samples:
        filenames = filenames[:num_samples]
    
    for filename in tqdm(filenames, desc="Preprocessing Data"):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".nii.gz", ".nii.gz"))

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        resized_slices = [resize(image[:, :, i], image_size, anti_aliasing=True) for i in range(image.shape[2])]
        image = np.stack(resized_slices, axis=-1)

        resized_label_slices = [resize(label[:, :, i], image_size, anti_aliasing=False, order=0, preserve_range=True) for i in range(label.shape[2])]
        label = np.stack(resized_label_slices, axis=-1)

        images.append(image)
        labels.append(label)

    padded_images = []
    padded_labels = []
    for image, label in zip(images, labels):
        if image.shape[2] < depth_size:
            pad_width = depth_size - image.shape[2]
            padded_image = np.pad(image, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
            padded_label = np.pad(label, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif image.shape[2] > depth_size:
            padded_image = image[:, :, :depth_size]
            padded_label = label[:, :, :depth_size]
        else:
            padded_image = image
            padded_label = label
        
        padded_images.append(padded_image)
        padded_labels.append(padded_label)

    images = np.array(padded_images)
    labels = np.array(padded_labels)

    if images.ndim != 4 or labels.ndim != 4:
        print(f"Images or labels array dimension mismatch: images.ndim={images.ndim}, labels.ndim={labels.ndim}")

    images = np.expand_dims(images, axis=1)
    labels = np.expand_dims(labels, axis=1)

    print(f"Data preprocessing completed in {time() - start_time:.2f} seconds")
    return images, labels

# Directories
image_dir = 'Task06_Lung/imagesTr'
label_dir = 'Task06_Lung/labelsTr'

# Preprocess data
images, labels = preprocess_data(image_dir, label_dir, num_samples=5, depth_size=128)

dataset = LungDataset(images, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Transformer model class
class TransformerMetaLearning(nn.Module):
    def __init__(self):
        super(TransformerMetaLearning, self).__init__()
        config = ViTConfig(image_size=128, patch_size=16, num_channels=1)
        self.model = ViTModel(config)
        self.reduce_dim = nn.Linear(config.hidden_size * 65, 1024)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        transformer_output = self.model(x).last_hidden_state
        print(f"Transformer Output Shape: {transformer_output.shape}")
        
        transformer_output = transformer_output.view(batch_size, depth, -1) 
        reduced_output = self.reduce_dim(transformer_output.view(batch_size * depth, -1))
        reduced_output = reduced_output.view(batch_size, depth, -1)
        print(f"Reduced Transformer Output Shape: {reduced_output.shape}")
        
        return reduced_output

# GNN with Attention class
class GNNWithAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNNWithAttention, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads=4)
        self.layer2 = GATConv(hidden_dim * 4, out_dim, num_heads=1)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = torch.flatten(h, start_dim=1)
        h = self.layer2(g, h)
        return h

# Hybrid model class
class HybridModel(nn.Module):
    def __init__(self, transformer, gnn):
        super(HybridModel, self).__init__()
        self.transformer = transformer
        self.gnn = gnn

    def forward(self, x, g):
        transformer_output = self.transformer(x)
        print(f"Transformer Output Before Flattening: {transformer_output.shape}")
        transformer_output = transformer_output.view(transformer_output.size(0), -1)
        gnn_output = self.gnn(g, transformer_output)
        return gnn_output

# Instantiate models
transformer_meta = TransformerMetaLearning()
gnn = GNNWithAttention(in_dim=1024, hidden_dim=256, out_dim=2)
hybrid_model = HybridModel(transformer_meta, gnn)

if torch.cuda.device_count() > 1:
    hybrid_model = nn.DataParallel(hybrid_model)

# Movingg model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model.to(device)

scaler = GradScaler()

def train_hybrid_model(model, train_loader, optimizer, loss_fn):
    start_time = time()
    model.train()
    for batch in tqdm(train_loader, desc="Training"):
        x, labels = batch
        x = x.clone().detach().float().to(device)
        labels = labels.clone().detach().long().to(device)
        g = dgl.graph(([], []))
        g = g.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = model(x, g)
            loss = loss_fn(output, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Training completed in {time() - start_time:.2f} seconds")

def evaluate_hybrid_model(model, val_loader, loss_fn):
    start_time = time()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            x, labels = batch
            x = x.clone().detach().float().to(device)
            labels = labels.clone().detach().long().to(device)
            g = dgl.graph(([], []))
            g = g.to(device)
            
            with autocast():
                output = model(x, g)
                val_loss += loss_fn(output, labels).item()
    print(f"Evaluation completed in {time() - start_time:.2f} seconds")
    return val_loss / len(val_loader)

# Optimizer and loss function
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train and evaluate
train_hybrid_model(hybrid_model, train_loader, optimizer, loss_fn)
val_loss = evaluate_hybrid_model(hybrid_model, val_loader, loss_fn)
print(f'Validation Loss: {val_loss}')

# visualization of a preprocessed image and its label
def visualize_sample(images, labels, index):
    middle_slice = images[index, 0, :, :, images.shape[4] // 2]
    middle_label = labels[index, 0, :, :, labels.shape[4] // 2]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Image Slice")
    plt.imshow(middle_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Label Slice")
    plt.imshow(middle_label, cmap='gray')
    plt.axis('off')

    plt.show()

visualize_sample(images, labels, index=0)
