#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iomanip>
#include "utils.h"

using Type = float;

// Configurations
const int ROW_WIDTH = 4;
const int COL_WIDTH = 4;
const int ESIZE = 512;

// CUDA kernel for FlashAttention
template<int esize>
__global__ void flash_att_v2_fwd(Type* Q, Type* K, Type* V, Type* O, int seqlen, Type smScale) {
    __shared__ Type shared_Q[ROW_WIDTH][esize];
    __shared__ Type shared_K[COL_WIDTH][esize];
    __shared__ Type shared_V[COL_WIDTH][esize];
    __shared__ Type shared_O[ROW_WIDTH][esize];
    __shared__ Type shared_QK[ROW_WIDTH][COL_WIDTH];
    __shared__ Type shared_SafeE[ROW_WIDTH][COL_WIDTH];
    __shared__ Type shared_Denom[ROW_WIDTH];
    __shared__ Type shared_Max[ROW_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * blockDim.y;

    if (row >= seqlen) return;

    for (int i = 0; i < (esize + COL_WIDTH - 1) / COL_WIDTH; i++) {
        shared_Q[ty][i * COL_WIDTH + tx] = Q[row * esize + i * COL_WIDTH + tx];
        shared_O[ty][i * COL_WIDTH + tx] = 0;
    }

    shared_Max[ty] = -INFINITY;
    shared_Denom[ty] = 0;

    for (int j = 0; j < (seqlen + COL_WIDTH - 1) / COL_WIDTH; j++) {
        if (j * COL_WIDTH + tx < seqlen) {
            for (int i = 0; i < (esize + ROW_WIDTH - 1) / ROW_WIDTH; i++) {
                shared_K[tx][i * ROW_WIDTH + ty] = K[j * COL_WIDTH * esize + tx * esize + i * ROW_WIDTH + ty];
                shared_V[tx][i * ROW_WIDTH + ty] = V[j * COL_WIDTH * esize + tx * esize + i * ROW_WIDTH + ty];
            }
        }

        __syncthreads();

        Type sum = 0;
        for (int i = 0; i < esize; i++) {
            sum += shared_Q[ty][i] * shared_K[tx][i];
        }
        shared_QK[ty][tx] = sum * smScale;

        __syncthreads();

        Type localMax = -INFINITY;
        for (int i = 0; i < COL_WIDTH; i++) {
            localMax = max(localMax, shared_QK[ty][i]);
        }

        Type newMax = max(shared_Max[ty], localMax);

        shared_SafeE[ty][tx] = exp(shared_QK[ty][tx] - newMax);

        Type localDenom = 0;
        for (int i = 0; i < COL_WIDTH; i++) {
            localDenom += shared_SafeE[ty][i];
        }

        Type rescaleOld = exp(shared_Max[ty] - newMax);
        Type newDenom = shared_Denom[ty] * rescaleOld + localDenom;

        for (int i = 0; i < (esize + COL_WIDTH - 1) / COL_WIDTH; i++) {
            shared_O[ty][i * COL_WIDTH + tx] *= rescaleOld;
            for (int k = 0; k < COL_WIDTH; k++) {
                shared_O[ty][i * COL_WIDTH + tx] += shared_SafeE[ty][k] * shared_V[k][i * COL_WIDTH + tx];
            }
        }

        shared_Max[ty] = newMax;
        shared_Denom[ty] = newDenom;
        __syncthreads();
    }

    for (int i = 0; i < (esize + COL_WIDTH - 1) / COL_WIDTH; i++) {
        O[row * esize + i * COL_WIDTH + tx] = shared_O[ty][i * COL_WIDTH + tx] / shared_Denom[ty];
    }
}

// Wrapper function to run the kernel
void run_flash_attn_v2(Type* Q, Type* K, Type* V, Type* O, int m, int n) {
    Type softmax_scale = 1.f / sqrtf(static_cast<Type>(n));
    int gridY = (m + ROW_WIDTH - 1) / ROW_WIDTH;
    dim3 grid(1, gridY);
    dim3 block(COL_WIDTH, ROW_WIDTH);

    flash_att_v2_fwd<ESIZE> << <grid, block >> > (Q, K, V, O, m, softmax_scale);
}

// Testing function
void test_attention(const TestData& test_data) {
    int m = test_data.seq_len;
    int n = test_data.emb_size;

    std::vector<float> h_Q(m * n), h_K(m * n), h_V(m * n), h_O(m * n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            h_Q[idx] = test_data.data[i][j];
            h_K[idx] = test_data.data[i][j];
            h_V[idx] = test_data.data[i][j];
        }
    }

    float* d_Q, * d_K, * d_V, * d_O;
    cudaMalloc(&d_Q, m * n * sizeof(float));
    cudaMalloc(&d_K, m * n * sizeof(float));
    cudaMalloc(&d_V, m * n * sizeof(float));
    cudaMalloc(&d_O, m * n * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    run_flash_attn_v2(d_Q, d_K, d_V, d_O, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("| Seq Len: %d, Emb Size: %d, Time: %.3f ms\n", m, n, elapsed);

    cudaMemcpy(h_O.data(), d_O, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

// Main function
int main() {
    std::string dataset_dir = "../flash_attention_dataset";
    auto dataset = loadData(dataset_dir);

    for (const auto& test_data : dataset) {
        test_attention(test_data);
    }

    return 0;
}
