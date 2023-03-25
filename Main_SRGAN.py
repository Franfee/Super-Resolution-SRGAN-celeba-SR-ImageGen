"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from:
https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
"""


import os
import sys


from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from net.models import *
from utils.datasets import *
from utils.get_parser import get_all_parse


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


opt = get_all_parse()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.Tensor

dataloader = DataLoader(
    ImageDataset("datasets/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch_data in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i

        # Configure model input
        real_images_lr = batch_data["lr"].type(Tensor)
        real_images_hr = batch_data["hr"].type(Tensor)

        # Adversarial ground truths labels
        valid = Tensor(np.ones((real_images_lr.size(0), *discriminator.output_shape)))
        fake = Tensor(np.zeros((real_images_lr.size(0), *discriminator.output_shape)))

        # ---------Train Generators----------
        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(real_images_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(real_images_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()
        # ----------------------------------

        # ----------Train Discriminator-----------
        optimizer_D.zero_grad()

        # Loss of real and fake images
        pred_real = discriminator(real_images_hr)
        pred_fake = discriminator(gen_hr.detach())
        loss_real = criterion_GAN(pred_real, valid)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        # ----------------------------------------

        # ------Log Progress--------
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )
        # -------------------------

        if batches_done % opt.sample_interval == 0:
            # Save image grid with up-sampled inputs and SRGAN outputs
            real_images_lr = nn.functional.interpolate(real_images_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            real_images_lr = make_grid(real_images_lr, nrow=1, normalize=True)
            real_images_hr = make_grid(real_images_hr, nrow=1, normalize=True)

            img_grid = torch.cat((real_images_lr, gen_hr, real_images_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
