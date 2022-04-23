
import torch.nn as nn
import torch.nn.functional as F


from torchvision.utils import make_grid
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import numpy as np


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

        self.disc = nn.Sequential(
          # in: 3 x 64 x 64
          nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(0.2),
          # out: 64 x 32 x 32

          nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
          nn.InstanceNorm2d(128, affine=True),
          nn.LeakyReLU(0.2),
          # out: 128 x 16 x 16

          nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
          nn.InstanceNorm2d(256, affine=True),
          nn.LeakyReLU(0.2),
          # out: 256 x 8 x 8

          nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
          nn.InstanceNorm2d(512, affine=True),
          nn.LeakyReLU(0.2),
          # out: 512 x 4 x 4
        )
        self.flat = nn.Flatten()

        self.finalLayer = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0)

        self.aux_layer = nn.Sequential(
            nn.Linear(512*4*4, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )


    def forward(self, img):
        out = self.disc(img)

        validity = self.finalLayer(out)
        label = self.aux_layer(self.flat(out))
        return validity, label

class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_block = nn.Sequential(
        nn.ConvTranspose2d(latent_size + 10, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        # 512, 4, 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        # 256, 8, 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # 128, 16, 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # 64, 32, 32

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )

  def forward(self, input):
    img = self.conv_block(input)

    return img