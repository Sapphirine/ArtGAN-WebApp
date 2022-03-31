
import torch.nn as nn
import torch.nn.functional as F


from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile


image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
latent_size = 128

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    n_row = np.sqrt(nmax)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=n_row).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images.cpu(), nmax)
        print(images.shape)
        break

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
          # in: 3 x 64 x 64
          nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.2, inplace=True),
          # out: 64 x 32 x 32

          nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2, inplace=True),
          # out: 128 x 16 x 16

          nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.2, inplace=True),
          # out: 256 x 8 x 8

          nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2, inplace=True),
          # out: 512 x 4 x 4
        )
        self.flat = nn.Flatten()
        self.adv_layer = nn.Sequential(
            nn.Linear(512*4*4, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(512*4*4, 10),
        )


    def forward(self, img):
        out = self.conv_blocks(img)
        flat = self.flat(out)
        
        validity = self.adv_layer(flat)
        label = self.aux_layer(flat)

        return validity, label

class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_block = nn.Sequential(
        nn.ConvTranspose2d(latent_size + 10, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh(),
    )

  def forward(self, input):
    img = self.conv_block(input)

    return img