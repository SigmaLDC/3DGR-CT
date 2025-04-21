import torch
from torch.autograd import Function
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load


compute_intensity_cuda = load(
    name='compute_intensity_cuda', 
    sources=['/data/yingtaili/3DGR-CT-archive/gs_utils/discretize_grid.cu'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3']
)


class IntensityComputation(Function):
    @staticmethod
    def forward(ctx, gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
        ctx.save_for_backward(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
        
        # Call the forward CUDA function
        intensity_grid = compute_intensity_cuda.compute_intensity(
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )
        
        return intensity_grid

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid = ctx.saved_tensors
        
        # Call the backward CUDA function
        grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_intensity_grid = compute_intensity_cuda.compute_intensity_backward(
            grad_output,
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )

        return grad_gaussian_centers, None, grad_intensities, grad_inv_covariances, None, grad_intensity_grid


def compute_intensity(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
    return IntensityComputation.apply(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
