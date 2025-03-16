#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

typedef long long ll;

#define TILE_SIZE 32  

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    extern __shared__ long int shared_filter[]; 
    int filter_idx = blockIdx.z;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int filter_size = r * s * c;

    if (tid < filter_size) {
        shared_filter[tid] = filter[filter_idx * filter_size + tid];
    }
    __syncthreads();

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= w || out_y >= h)
        return; 

    ll sum = 0;

    for (int ch = 0; ch < c; ch++)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < s; j++)
            {
                int img_x = out_x + j - s / 2;
                int img_y = out_y + i - r / 2;

                if (img_x >= 0 && img_x < w && img_y >= 0 && img_y < h)
                {
                    int img_idx = (ch * h + img_y) * w + img_x;
                    int filt_idx = ((filter_idx * c + ch) * r + i) * s + j; 

                    sum += matrix[img_idx] * shared_filter[filt_idx];
                }
            }
        }
    }

    result[(filter_idx * h + out_y) * w + out_x] = sum; 
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    long int *d_mat, *d_filter, *d_ans;
    cudaMalloc(&d_mat, sizeof(long int) * h * w * c);
    cudaMalloc(&d_filter, sizeof(long int) * r * s * c * k);
    cudaMalloc(&d_ans, sizeof(long int) * h * w * k);

    cudaMemcpy(d_mat, h_mat, sizeof(long int) * h * w * c, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(long int) * r * s * c * k, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((w + TILE_SIZE - 1) / TILE_SIZE, (h + TILE_SIZE - 1) / TILE_SIZE, k);

    int sharedMemSize = sizeof(long int) * r * s * c;
    dkernel<<<gridDim, blockDim, sharedMemSize>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k);

    cudaMemcpy(h_ans, d_ans, sizeof(long int) * h * w * k, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> elapsed1 = end - start;

    cudaFree(d_mat);
    cudaFree(d_filter);
    cudaFree(d_ans);

    // Write output to file
    std::ofstream file("cuda.out");
    cudaDeviceSynchronize();
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    delete[] h_mat;
    delete[] h_filter;
    delete[] h_ans;

    return 0;
}
