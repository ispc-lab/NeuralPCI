#include "points_query.h"
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>

#define THREAD_PER_BLOCK 256
#define HEAP_SIZE 64

__device__ void swap_float(float &a, float &b)
{
    float temp = a;
    a = b;
    b = temp;
}

__device__ void swap_int64(int64_t &a, int64_t &b)
{
    int64_t temp = a;
    a = b;
    b = temp;
}

__device__ void reheap(float *k_dis,
                       int64_t *k_idx,
                       int k)
{
    int parent = 0;
    int child = 2 * parent + 1;

    while (child < k)
    {
        if (child + 1 < k && k_dis[child + 1] > k_dis[child])
            child++;
        if (k_dis[parent] > k_dis[child])
            break;
        swap_float(k_dis[parent], k_dis[child]);
        swap_int64(k_idx[parent], k_idx[child]);
        parent = child;
        child = child * 2 + 1;
    }
}

__device__ void heap_sort(float *k_dis,
                          int64_t *k_idx,
                          int k)
{
    for (int i = k - 1; i > 0; i--)
    {
        swap_float(k_dis[0], k_dis[i]);
        swap_int64(k_idx[0], k_idx[i]);
        reheap(k_dis, k_idx, i);
    }
}

__global__ void knn_kernel(int k,
                           torch::PackedTensorAccessor32<float, 3> points,
                           torch::PackedTensorAccessor32<float, 3> centroids,
                           torch::PackedTensorAccessor32<int64_t, 3> indices,
                           torch::PackedTensorAccessor32<float, 3> square_dis)
{
    int b = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= centroids.size(1))
    {
        return;
    }

    // 声明大小为HEAP_SIZE的最大堆
    float k_dis[HEAP_SIZE];
    int64_t k_idx[HEAP_SIZE];
    for (int i = 0; i < HEAP_SIZE; i++)
    {
        k_dis[i] = 1e10;
        k_idx[i] = 0;
    }

    float centroid_x = centroids[b][m][0];
    float centroid_y = centroids[b][m][1];
    float centroid_z = centroids[b][m][2];
    for (int i = 0; i < points.size(1); i++)
    {
        float x = points[b][i][0];
        float y = points[b][i][1];
        float z = points[b][i][2];
        float dis = (centroid_x - x) * (centroid_x - x) + (centroid_y - y) * (centroid_y - y) + (centroid_z - z) * (centroid_z - z);
        if (dis < k_dis[0])
        {
            k_dis[0] = dis;
            k_idx[0] = i;
            
            reheap(k_dis, k_idx, k);
        }
    }
    heap_sort(k_dis, k_idx, k);

    for (int i = 0; i < k; i++)
    {
        square_dis[b][m][i] = k_dis[i];
        indices[b][m][i] = k_idx[i];
    }
}

__global__ void ball_query_kernel(int k,
                                  float radius,
                                  torch::PackedTensorAccessor32<float, 3> points,
                                  torch::PackedTensorAccessor32<float, 3> centroids,
                                  torch::PackedTensorAccessor32<int64_t, 3> indices,
                                  torch::PackedTensorAccessor32<float, 3> square_dis)
{
    int b = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= centroids.size(1))
    {
        return;
    }

    // 声明大小为HEAP_SIZE的最大堆
    float k_dis[HEAP_SIZE];
    int64_t k_idx[HEAP_SIZE];
    for (int i = 0; i < HEAP_SIZE; i++)
    {
        k_dis[i] = 1e10;
        k_idx[i] = 0;
    }

    float centroid_x = centroids[b][m][0];
    float centroid_y = centroids[b][m][1];
    float centroid_z = centroids[b][m][2];
    for (int i = 0; i < points.size(1); i++)
    {
        float x = points[b][i][0];
        float y = points[b][i][1];
        float z = points[b][i][2];
        float dis = (centroid_x - x) * (centroid_x - x) + (centroid_y - y) * (centroid_y - y) + (centroid_z - z) * (centroid_z - z);
        if (dis < k_dis[0])
        {
            k_dis[0] = dis;
            k_idx[0] = i;
            
            reheap(k_dis, k_idx, k);
        }
    }
    heap_sort(k_dis, k_idx, k);

    // 将距离大于radius的点筛掉
    for (int i = k - 1; i > 0; i--)
    {
        if (k_dis[i] <= (radius * radius))
            break;
        else 
        {
            k_dis[i] = k_dis[0];
            k_idx[i] = k_idx[0];
        }
    }

    for (int i = 0; i < k; i++)
    {
        square_dis[b][m][i] = k_dis[i];
        indices[b][m][i] = k_idx[i];
    }
}

void knn_launcher(int k,
                  torch::Tensor points,
                  torch::Tensor centroids,
                  torch::Tensor indices,
                  torch::Tensor square_dis)
{
    const at::cuda::OptionalCUDAGuard device_guard(points.device());
    
    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp = centroids.size(1) / THREAD_PER_BLOCK + ((centroids.size(1) % THREAD_PER_BLOCK) > 0);
    dim3 block(temp, centroids.size(0), 1);

    // 启动kernel
    knn_kernel<<<block, thread>>> (k, 
                                   points.packed_accessor32<float, 3>(),
                                   centroids.packed_accessor32<float, 3>(),
                                   indices.packed_accessor32<int64_t, 3>(),
                                   square_dis.packed_accessor32<float, 3>());
}

void ball_query_launcher(int k,
                         float radius,
                         torch::Tensor points,
                         torch::Tensor centroids,
                         torch::Tensor indices,
                         torch::Tensor square_dis)
{
    const at::cuda::OptionalCUDAGuard device_guard(points.device());
    
    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp = centroids.size(1) / THREAD_PER_BLOCK + ((centroids.size(1) % THREAD_PER_BLOCK) > 0);
    dim3 block(temp, centroids.size(0), 1);

    // 启动kernel
    ball_query_kernel<<<block, thread>>> (k, 
                                          radius,
                                          points.packed_accessor32<float, 3>(),
                                          centroids.packed_accessor32<float, 3>(),
                                          indices.packed_accessor32<int64_t, 3>(),
                                          square_dis.packed_accessor32<float, 3>());
}
