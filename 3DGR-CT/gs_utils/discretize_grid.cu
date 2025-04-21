#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

// 
__constant__ int const_num_gaussians;
__constant__ int const_grid_size_z;
__constant__ int const_grid_size_x;
__constant__ int const_grid_size_y;


// CUDA kernel function to compute intensity
__global__ void compute_intensity_forward_kernel(
    const float* __restrict__ grid_points,
    const float* __restrict__ inv_covariances,
    const float* __restrict__ gaussian_centers,
    const float* __restrict__ intensities,
    const float* __restrict__ scalings,
    float* __restrict__ intensity_grid,
    int* work_queue,
    int* work_counter
    ) {

    __shared__ float inv_cov[256][9];
    __shared__ float centers[256][3];
    __shared__ float scales[256][3];

    // 
    int gaussian_idx;
    while ((gaussian_idx = atomicAdd(work_counter, 1)) < const_num_gaussians) {
        if (gaussian_idx >= const_num_gaussians) return;

        for (int i = 0; i < 9; ++i) {
            inv_cov[threadIdx.x][i] = inv_covariances[gaussian_idx * 9 + i];
        }

        centers[threadIdx.x][0] = gaussian_centers[gaussian_idx * 3];
        centers[threadIdx.x][1] = gaussian_centers[gaussian_idx * 3 + 1];
        centers[threadIdx.x][2] = gaussian_centers[gaussian_idx * 3 + 2];

        scales[threadIdx.x][0] = scalings[gaussian_idx * 3];
        scales[threadIdx.x][1] = scalings[gaussian_idx * 3 + 1];
        scales[threadIdx.x][2] = scalings[gaussian_idx * 3 + 2];
        __syncthreads();

        // Compute mean coordinates
        int center_idx = gaussian_idx * 3;
        float mean_z = centers[threadIdx.x][0] * (float)const_grid_size_z;
        float mean_x = centers[threadIdx.x][1] * (float)const_grid_size_x;
        float mean_y = centers[threadIdx.x][2] * (float)const_grid_size_y;

        // 
        float coeff = 2.0;

        float norm_expand_z = coeff * scales[threadIdx.x][0];
        float norm_expand_x = coeff * scales[threadIdx.x][1];
        float norm_expand_y = coeff * scales[threadIdx.x][2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;
        

        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        int grid_xy = const_grid_size_x * const_grid_size_y;
        float intensity = intensities[gaussian_idx];

        for (int z = z_start; z <= z_end; ++z){
            for (int x = x_start; x <= x_end; ++x) {
                for (int y = y_start; y <= y_end; ++y){
                    // Compute distance from the grid point to the Gaussian center
                    int grid_idx = 3 * (z * grid_xy + x * const_grid_size_y + y);
                    float dz = grid_points[grid_idx] - gaussian_centers[center_idx];
                    float dx = grid_points[grid_idx + 1] - gaussian_centers[center_idx + 1];
                    float dy = grid_points[grid_idx + 2] - gaussian_centers[center_idx + 2];

                    // Compute the expoential term
                    float power = -0.5f * (
                        dz * (inv_cov[threadIdx.x][0] * dz + inv_cov[threadIdx.x][1] * dx + inv_cov[threadIdx.x][2] * dy) +
                        dx * (inv_cov[threadIdx.x][3] * dz + inv_cov[threadIdx.x][4] * dx + inv_cov[threadIdx.x][5] * dy) +
                        dy * (inv_cov[threadIdx.x][6] * dz + inv_cov[threadIdx.x][7] * dx + inv_cov[threadIdx.x][8] * dy)
                    );
                    
                    // Compute the intensity value
                    float intensity_value = intensity * __expf(power);

                    // Atomic add
                    atomicAdd(&intensity_grid[z * grid_xy + x * const_grid_size_y + y], intensity_value);

                }
            }
        }
    }
}


torch::Tensor compute_intensity(
    torch::Tensor gaussian_centers,
    torch::Tensor grid_points, //
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings,
    torch::Tensor intensity_grid
    ){
    
    const int num_gaussians = gaussian_centers.size(0);
    const int grid_size_z = intensity_grid.size(1);
    const int grid_size_x = intensity_grid.size(2);
    const int grid_size_y = intensity_grid.size(3);

    // 
    cudaMemcpyToSymbol(const_num_gaussians, &num_gaussians, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_z, &grid_size_z, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_x, &grid_size_x, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_y, &grid_size_y, sizeof(int));

    const int threads_per_block = 256;
    const int num_blocks = (num_gaussians + threads_per_block - 1) / threads_per_block;

    // Initialize work queue and counter
    int* d_work_queue;
    int* d_work_counter;
    cudaMalloc(&d_work_queue, num_gaussians * sizeof(int));
    cudaMalloc(&d_work_counter, sizeof(int));
    cudaMemset(d_work_counter, 0, sizeof(int));

    // Fill the work queue
    thrust::sequence(thrust::device, d_work_queue, d_work_queue + num_gaussians);

    compute_intensity_forward_kernel<<<num_blocks, threads_per_block>>>(
        grid_points.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        gaussian_centers.data_ptr<float>(),
        intensities.data_ptr<float>(),
        scalings.data_ptr<float>(),
        intensity_grid.data_ptr<float>(),
        d_work_queue,
        d_work_counter
    );

    cudaFree(d_work_queue);
    cudaFree(d_work_counter);


    return intensity_grid;
}


__forceinline__ __device__ void compute_grad_gaussian_center(
    float dz, float dx, float dy,
    const float* __restrict__ inv_cov,
    float intensity_value,
    float grad_output_val,
    float* __restrict__ grad_gaussian_center) {

    float common_term = 0.5f * intensity_value * grad_output_val;
    atomicAdd(&grad_gaussian_center[0], common_term * (2 * inv_cov[0] * dz + (inv_cov[1] + inv_cov[3]) * dx + (inv_cov[2] + inv_cov[6]) * dy));
    atomicAdd(&grad_gaussian_center[1], common_term * ((inv_cov[3] + inv_cov[1]) * dz + 2 * inv_cov[4] * dx + (inv_cov[5] + inv_cov[7]) * dy));
    atomicAdd(&grad_gaussian_center[2], common_term * ((inv_cov[6] + inv_cov[2]) * dz + (inv_cov[7] + inv_cov[5]) * dx + 2 * inv_cov[8] * dy));
}

__forceinline__ __device__ void compute_grad_inv_covariance(
    float dz, float dx, float dy,
    float intensity_value,
    float grad_output_val,
    float* __restrict__ grad_inv_covariance) {

    float grad_common = -0.5f * intensity_value * grad_output_val;
    atomicAdd(&grad_inv_covariance[0], grad_common * dz * dz);
    atomicAdd(&grad_inv_covariance[1], grad_common * dz * dx);
    atomicAdd(&grad_inv_covariance[2], grad_common * dz * dy);
    atomicAdd(&grad_inv_covariance[3], grad_common * dx * dz);
    atomicAdd(&grad_inv_covariance[4], grad_common * dx * dx);
    atomicAdd(&grad_inv_covariance[5], grad_common * dx * dy);
    atomicAdd(&grad_inv_covariance[6], grad_common * dy * dz);
    atomicAdd(&grad_inv_covariance[7], grad_common * dy * dx);
    atomicAdd(&grad_inv_covariance[8], grad_common * dy * dy);
}

__global__ void compute_intensity_backward_kernel(
    const float* __restrict__ grad_output, 
    const float* __restrict__ grid_points,
    const float* __restrict__ inv_covariances,
    const float* __restrict__ gaussian_centers,
    const float* __restrict__ intensities,
    const float* __restrict__ scalings,

    float* __restrict__ grad_gaussian_centers,
    float* __restrict__ grad_intensities, 
    float* __restrict__ grad_inv_covariances, // 
    int* work_queue,
    int* work_counter
){

    __shared__ float inv_cov[256][9]; // 
    __shared__ float centers[256][3];
    __shared__ float scales[256][3];

    int gaussian_idx;
    while ((gaussian_idx = atomicAdd(work_counter, 1)) < const_num_gaussians) {
        if (gaussian_idx >= const_num_gaussians) return;

        for (int i = 0; i < 9; ++i) {
            inv_cov[threadIdx.x][i] = inv_covariances[gaussian_idx * 9 + i];
        }

        centers[threadIdx.x][0] = gaussian_centers[gaussian_idx * 3];
        centers[threadIdx.x][1] = gaussian_centers[gaussian_idx * 3 + 1];
        centers[threadIdx.x][2] = gaussian_centers[gaussian_idx * 3 + 2];

        scales[threadIdx.x][0] = scalings[gaussian_idx * 3];
        scales[threadIdx.x][1] = scalings[gaussian_idx * 3 + 1];
        scales[threadIdx.x][2] = scalings[gaussian_idx * 3 + 2];
        __syncthreads();

        // Compute mean coordinates
        int center_idx = gaussian_idx * 3;
        float mean_z = centers[threadIdx.x][0] * (float)const_grid_size_z;
        float mean_x = centers[threadIdx.x][1] * (float)const_grid_size_x;
        float mean_y = centers[threadIdx.x][2] * (float)const_grid_size_y;

        // 
        float coeff = 2.0;
        float norm_expand_z = coeff * scales[threadIdx.x][0];
        float norm_expand_x = coeff * scales[threadIdx.x][1];
        float norm_expand_y = coeff * scales[threadIdx.x][2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;


        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        int grid_xy = const_grid_size_x * const_grid_size_y;
        float intensity = intensities[gaussian_idx];

        for (int z = z_start; z <= z_end; ++z){
            for (int x = x_start; x <= x_end; ++x) {
                for (int y = y_start; y <= y_end; ++y){
                    // Compute distance from the grid point to the Gaussian center
                    int grid_idx = 3 * (z * grid_xy + x * const_grid_size_y + y);
                    float dz = grid_points[grid_idx] - gaussian_centers[center_idx];
                    float dx = grid_points[grid_idx + 1] - gaussian_centers[center_idx + 1];
                    float dy = grid_points[grid_idx + 2] - gaussian_centers[center_idx + 2];

                    // Compute the exponential term
                    float power = -0.5f * (
                        dz * (inv_cov[threadIdx.x][0] * dz + inv_cov[threadIdx.x][1] * dx + inv_cov[threadIdx.x][2] * dy) +
                        dx * (inv_cov[threadIdx.x][3] * dz + inv_cov[threadIdx.x][4] * dx + inv_cov[threadIdx.x][5] * dy) +
                        dy * (inv_cov[threadIdx.x][6] * dz + inv_cov[threadIdx.x][7] * dx + inv_cov[threadIdx.x][8] * dy)
                    );

                    float grad_output_val = grad_output[z * grid_xy + x * const_grid_size_y + y];
                    
                    // Compute the intensity value
                    float intensity_value = intensity * __expf(power);

                    // Compute gradient w.r.t. intensity
                    float grad_intensity = __expf(power) * grad_output_val;
                    atomicAdd(&grad_intensities[gaussian_idx], grad_intensity);

                    // Gradient w.r.t. gaussian centers
                    compute_grad_gaussian_center(
                        dz, dx, dy,
                        inv_cov[threadIdx.x],
                        intensity_value,
                        grad_output_val,
                        &grad_gaussian_centers[center_idx]
                    );

                    // Gradient w.r.t. inverse covariances
                    compute_grad_inv_covariance(
                        dz, dx, dy,
                        intensity_value,
                        grad_output_val,
                        &grad_inv_covariances[gaussian_idx * 9]
                    );

                }
            }
        }
    }
}

std::vector<torch::Tensor> compute_intensity_backward(
    torch::Tensor grad_output,
    torch::Tensor gaussian_centers,
    torch::Tensor grid_points, //
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings, //
    torch::Tensor intensity_grid
){
    const int num_gaussians = gaussian_centers.size(0);
    const int grid_size_z = intensity_grid.size(1);
    const int grid_size_x = intensity_grid.size(2);
    const int grid_size_y = intensity_grid.size(3);

    // 
    cudaMemcpyToSymbol(const_num_gaussians, &num_gaussians, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_z, &grid_size_z, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_x, &grid_size_x, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_y, &grid_size_y, sizeof(int));

    auto grad_gaussian_centers = torch::zeros_like(gaussian_centers);
    auto grad_intensities = torch::zeros_like(intensities);
    auto grad_inv_covariances = torch::zeros_like(inv_covariances);

    const int threads_per_block = 256;
    const int num_blocks = (num_gaussians + threads_per_block - 1) / threads_per_block;

    // Initialize work queue and counter
    int* d_work_queue;
    int* d_work_counter;
    cudaMalloc(&d_work_queue, num_gaussians * sizeof(int));
    cudaMalloc(&d_work_counter, sizeof(int));
    cudaMemset(d_work_counter, 0, sizeof(int));

    // Fill the work queue
    thrust::sequence(thrust::device, d_work_queue, d_work_queue + num_gaussians);

    compute_intensity_backward_kernel<<<num_blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        grid_points.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        gaussian_centers.data_ptr<float>(),
        intensities.data_ptr<float>(),
        scalings.data_ptr<float>(),
        grad_gaussian_centers.data_ptr<float>(),
        grad_intensities.data_ptr<float>(),
        grad_inv_covariances.data_ptr<float>(),
        d_work_queue,
        d_work_counter
    );

    cudaFree(d_work_queue);
    cudaFree(d_work_counter);


    return {grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_output};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("compute_intensity", &compute_intensity, "Compute Intensity (CUDA)");
    m.def("compute_intensity_backward", &compute_intensity_backward, "Compute intensity backward (CUDA)");
}


