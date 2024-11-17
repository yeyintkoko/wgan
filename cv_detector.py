import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torch.utils.data import DataLoader, Dataset
import os

# ===========================================
# 1. Object Detection with Mask R-CNN
# ===========================================
# Load Detectron2 model config and weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detection

# Specify CPU as the device
cfg.MODEL.DEVICE = "cpu"  # Set to "cpu" to ensure it's running on CPU

# Create the predictor (uses CPU)
predictor = DefaultPredictor(cfg)

# Read the input image
image = cv2.imread('imageai/input_image.jpg')

# Run object detection
outputs = predictor(image)

# Get segmentation masks (binary masks)
instances = outputs["instances"]
masks = instances.pred_masks.cpu().numpy()  # Shape: [num_objects, height, width]

# Display detected masks for each object
for idx, mask in enumerate(masks):
    cv2.imshow(f'Mask {idx}', mask.astype('uint8') * 255)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# ===========================================
# 2. U-Net-based GAN Architecture
# ===========================================

# Generator - U-Net Style
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 3 -> 64 channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 128 channels
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128 -> 256 channels
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256 -> 128 channels
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64 channels
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64 -> 3 channels (RGB)
            nn.Tanh()  # Output should be in range [-1, 1] for normalized images
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 3 -> 64 channels
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 128 channels
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),  # 128 -> 1 channel (real/fake output)
            nn.Sigmoid()  # Output a value between 0 and 1 (real/fake)
        )
        
    def forward(self, x):
        return self.model(x)

# ===========================================
# 3. Training the GAN
# ===========================================

# Create the Generator and Discriminator
generator = Generator()  # No .cuda() here, run on CPU
discriminator = Discriminator()  # No .cuda() here, run on CPU

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Function to create a masked image (set detected object to 0)
def mask_image(image, masks):
    masked_image = image.copy()
    for mask in masks:
        masked_image[mask == 1] = 0  # Set masked areas to black
    return masked_image

# ===========================================
# Training Loop
# ===========================================
num_epochs = 100
for epoch in range(num_epochs):
    # Create input tensors (masked image for generator and real image for discriminator)
    masked_image = mask_image(image, masks)  # Image with object removed
    masked_image = torch.tensor(masked_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    real_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Train Discriminator
    optimizer_D.zero_grad()
    
    # Real loss
    real_loss = criterion(discriminator(real_image), torch.ones_like(discriminator(real_image)))
    
    # Fake loss
    fake_image = generator(masked_image)
    fake_loss = criterion(discriminator(fake_image.detach()), torch.zeros_like(discriminator(fake_image)))
    
    # Total discriminator loss
    loss_D = (real_loss + fake_loss) / 2
    loss_D.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    
    # Generator loss (fool discriminator)
    loss_G = criterion(discriminator(fake_image), torch.ones_like(discriminator(fake_image)))  # Try to make fake images look real
    loss_G.backward()
    optimizer_G.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}')

# Save the model after training
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# ===========================================
# 4. Inpainting with the trained Generator
# ===========================================

# Load the trained generator
generator.load_state_dict(torch.load('generator.pth'))

# Prepare the masked image
masked_image = mask_image(image, masks)
masked_image = torch.tensor(masked_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

# Generate inpainted image
with torch.no_grad():
    inpainted_image = generator(masked_image)

# Convert to a valid image for display
inpainted_image = inpainted_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
inpainted_image = (inpainted_image * 255).astype(np.uint8)

# Display result
cv2.imshow('Inpainted Image', inpainted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
