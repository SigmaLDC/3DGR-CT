import os
import argparse
import shutil
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d
from ct_geometry_projector import ConeBeam3DProjector
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder)


output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

wandb.init(project="", 
           name = "",
           config=config)

# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])

config['img_size'] = (config['img_size'], config['img_size'], config['img_size']) if type(config['img_size']) == int else tuple(config['img_size'])
slice_idx = list(range(0, config['img_size'][0], int(config['img_size'][0]/config['display_image_num'])))
if config['num_proj'] > config['display_image_num']:
    proj_idx = list(range(0, config['num_proj'], int(config['num_proj']/config['display_image_num'])))
else:
    proj_idx = list(range(0, config['num_proj']))



class OptimizationParams():
    def __init__(self):
        self.position_lr_init = 0.002
        self.position_lr_final = 0.000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.intensity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

def tv_regularization(image):
    return torch.mean(torch.abs(image[:, 1:, :, :, :] - image[:, :-1, :, :, :])) + torch.mean(torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])) + torch.mean(torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :]))


ct_projector_low_reso = ConeBeam3DProjector((config['img_size'][0]//2, config['img_size'][1]//2, config['img_size'][2]//2), config['proj_size'], config['num_proj'])
ct_projector = ConeBeam3DProjector(config['img_size'], config['proj_size'], config['num_proj'])

for it, (grid, image) in enumerate(data_loader):
    grid = grid.cuda()  
    image = image.cuda()  
    
    projs = ct_projector.forward_project(image.transpose(1, 4).squeeze(1)) 

    image_low_resos = []
    projs_low_resos = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                image_low_reso = image[:, i::2, j::2, k::2, :]
                image_low_resos.append(image_low_reso)
                projs_low_resos.append(ct_projector_low_reso.forward_project(image_low_reso.transpose(1, 4).squeeze(1)))
    # FBP recon
    fbp_recon = ct_projector.backward_project(projs)  # [bs, n, h, w] -> [bs, x, y, z]

    # Data loading
    test_data = (grid, image)  # [bs, z, x, y, 1]
    train_data = (grid, projs)  # [bs, n, h, w]

    save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))
    save_image_3d(train_data[1].transpose(2, 3).unsqueeze(-1), proj_idx, os.path.join(image_directory, "train.png"))
    
    fbp_recon_ssim = compare_ssim(fbp_recon.squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), multichannel=True)  
    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4) 
    fbp_recon_psnr = - 10 * torch.log10(torch.mean((fbp_recon-test_data[1])**2))
    save_image_3d(fbp_recon, slice_idx, os.path.join(image_directory, "fbp_recon_{:.4g}dB_ssim{:.4g}.png".format(fbp_recon_psnr, fbp_recon_ssim)))

    # Setup Gaussian Model
    from models.gaussian_model import GaussianModel


    op = OptimizationParams()
    gaussians = GaussianModel()

    grid_low_resos = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                grid_low_resos.append(grid[:, i::2, j::2, k::2, :])
    fbp_recon_low_reso = fbp_recon[:, ::2, ::2, ::2, :]
    

    gaussians.create_from_fbp(fbp_recon_low_reso, air_threshold=config['air_threshold'], ini_intensity=config['ini_intensity'], 
                                ini_sigma=config['ini_sigma'], spatial_lr_scale=config['spatial_lr_scale'], num_samples=config['num_gaussian'],
                                start=config['start'])

    gaussians.training_setup(op)

    # Train model
    for iteration in range(max_iter):

        gaussians.update_learning_rate(iteration)

        if iteration < config['low_reso_stage']:
            train_output = gaussians.grid_sample(grid_low_resos[iteration%8])
        else:
            train_output = gaussians.grid_sample(grid)
        
        # Loss
        if iteration < config['low_reso_stage']:
            train_projs = ct_projector_low_reso.forward_project(train_output.transpose(1, 4).squeeze(1))
        else:
            train_projs = ct_projector.forward_project(train_output.transpose(1, 4).squeeze(1))


        if iteration < config['low_reso_stage']:
            loss = torch.nn.functional.mse_loss(train_projs, projs_low_resos[iteration%8]) + config['tv_weight'] * tv_regularization(train_output)
        else:
            loss = torch.nn.functional.mse_loss(train_projs, projs) + config['tv_weight'] * tv_regularization(train_output)

        loss.backward()

        gaussians.optimizer.step()

        if config['do_density_control']:
            with torch.no_grad():
                # Densification
                if gaussians.get_xyz.shape[-2] < config['max_gaussians'] and iteration < config['densify_until_iter']:
                        if iteration > config['densify_from_iter'] and iteration % config['densification_interval'] == 0:
                            gaussians.densify_and_prune(config['max_grad'], config['min_intensity'], sigma_extent=config['sigma_extent'])
            
        gaussians.optimizer.zero_grad(set_to_none = True)    

        # Compute training psnr
        if (iteration + 1) % config['log_iter'] == 0:
            if iteration < config['low_reso_stage']:
                train_psnr = -10 * torch.log10(torch.mean((train_projs-projs_low_resos[iteration%8])**2)).item()
            else:
                train_psnr = -10 * torch.log10(torch.mean((train_projs-projs)**2)).item()

            train_loss = loss.item()

            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iteration + 1, max_iter, train_loss, train_psnr))
            # Log training metrics to wandb
            wandb.log({
                "Iteration": iteration + 1,
                "Train Loss": train_loss,
                "Train PSNR": train_psnr
            })
        # Compute testing psnr
        if iteration == 0 or (iteration + 1) % config['val_iter'] == 0:

            with torch.no_grad():
                test_output = gaussians.grid_sample(test_data[0])  # [bs, z, x, y, 3]
                test_loss = 0.5 * torch.mean((test_output - test_data[1])**2)
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()
                test_ssim = compare_ssim(test_output.transpose(1,4).squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), multichannel=True)

                test_output_low_reso = gaussians.grid_sample(grid_low_resos[iteration%8])  # [bs, z, x, y, 3]
                test_loss_low_reso = 0.5 * torch.mean((test_output_low_reso - image_low_resos[iteration%8])**2)
                test_psnr_low_reso = - 10 * torch.log10(2 * test_loss_low_reso).item()
                test_loss_low_reso = test_loss_low_reso.item()
                test_ssim_low_reso = compare_ssim(test_output_low_reso.transpose(1,4).squeeze().cpu().numpy(), image_low_reso.transpose(1,4).squeeze().cpu().numpy(), multichannel=True)

            save_image_3d(test_output, slice_idx, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(iteration + 1, test_psnr, test_ssim)))

            wandb.log({
                "Iteration": iteration + 1,
                "Test Loss": test_loss, 
                "Test PSNR": test_psnr, 
                "Test SSIM": test_ssim, 
                "Test Loss-low_reso": test_loss_low_reso, 
                "Test PSNR-low_reso": test_psnr_low_reso, 
                "Test SSIM-low_reso": test_ssim_low_reso, 
            })
