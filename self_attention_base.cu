#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#include "dataload.h"

#define TILE_SIZE 4

__global__ void MatrixTranspose(float* P, float* P_T, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < M) {
        P_T[row * M + col] = P[col * N + row];
    }
}

__global__ void NaiveMatrixMultiply(float* M, float* N, float* P, int j, int k, int l, float scale_val) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < l && row < j) {
    float result = 0;
    for (int i = 0; i < k; i++) {
      result += (M[k * row + i] * N[i * l + col]);
    }
    P[row * l + col] = result / scale_val;
  }
}

__global__ void TiledMatrixMultiply(float* M, float* N, float* P, int j, int k, int l, float scale_val) {
  __shared__ float M_shared[TILE_SIZE][TILE_SIZE];
  __shared__ float N_shared[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float result = 0;
  for (int tileIdx = 0; tileIdx < (k + TILE_SIZE - 1) / TILE_SIZE; tileIdx++) {
    int tiledRow = tileIdx * TILE_SIZE + threadIdx.y;
    int tiledColumn = tileIdx * TILE_SIZE + threadIdx.x;
    if (row < j && tiledColumn < k) {
      M_shared[threadIdx.y][threadIdx.x] = M[row * k + tiledColumn];
    }
    else {
      M_shared[threadIdx.y][threadIdx.x] = 0;
    }
    if (col < l && tiledRow < k) {
      N_shared[threadIdx.y][threadIdx.x] = N[tiledRow * l + col];
    }
    else {
      N_shared[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    for (int idx = 0; idx < TILE_SIZE; idx++) {
      result += (M_shared[threadIdx.y][idx] * N_shared[idx][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < j && col < l) {
    P[row * l + col] = result / scale_val;
  }
}

extern __shared__ float shared_mem[];
__global__ void RowSoftmax(float *input, float *output, int seq_len) {
  int row = blockIdx.x;
  int col = threadIdx.x;

  if (row < seq_len && col < seq_len) {
      shared_mem[col] = input[row * seq_len + col];
      __syncthreads();

      float max_val = -FLT_MAX;
      for (int i = 0; i < seq_len; ++i) {
          max_val = fmaxf(max_val, shared_mem[i]);
      }
      __syncthreads();

      shared_mem[col] = expf(shared_mem[col] - max_val);
      __syncthreads();

      float sum_val = 0.0f;
      for (int i = 0; i < seq_len; ++i) {
          sum_val += shared_mem[i];
      }
      __syncthreads();

      output[row * seq_len + col] = shared_mem[col] / sum_val;
  }
}

void print_device_matrix(float *dev_ptr, int m, int n) {
  float *host_ptr = new float[m * n];
  cudaMemcpy(host_ptr, dev_ptr, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  print_matrix(host_ptr, m, n);
  free(host_ptr);
}

void naive_self_attention(float *Q, float *K, float *V, float *O, int seq_len, int emb_size) {
  float *KT;
  cudaMalloc((void **)&KT, sizeof(float) * seq_len * emb_size);
  dim3 KT_block((seq_len + TILE_SIZE - 1) / TILE_SIZE, (emb_size + TILE_SIZE - 1) / TILE_SIZE);
  dim3 KT_thread(TILE_SIZE, TILE_SIZE);
  MatrixTranspose<<<KT_block, KT_thread>>>(K, KT, seq_len, emb_size);
  cudaDeviceSynchronize();

  float scale_val = 1.f / sqrtf(static_cast<float>(emb_size));
  float *Q_KT;
  cudaMalloc((void **)&Q_KT, sizeof(float) * seq_len * emb_size);
  dim3 QKT_block((seq_len + TILE_SIZE - 1) / TILE_SIZE, (seq_len + TILE_SIZE - 1) / TILE_SIZE);
  dim3 QKT_thread(TILE_SIZE, TILE_SIZE);
  NaiveMatrixMultiply<<<QKT_block, QKT_thread>>>(Q, KT, Q_KT, seq_len, emb_size, seq_len, scale_val);
  // TiledMatrixMultiply<<<QKT_block, QKT_thread>>>(Q, KT, Q_KT, seq_len, emb_size, seq_len, scale_val);
  cudaDeviceSynchronize();

  // QK[M, M]
  dim3 sftmx_block(seq_len);
  dim3 sftmx_thread(seq_len);
  int shared_mem_size = seq_len * sizeof(float);
  RowSoftmax<<<sftmx_block, sftmx_thread, shared_mem_size, 0 >> >(Q_KT, Q_KT, seq_len);
  cudaDeviceSynchronize();

  // QK[M, M] @ V[M, N]
  dim3 QKV_block((seq_len + TILE_SIZE - 1) / TILE_SIZE, (emb_size + TILE_SIZE - 1) / TILE_SIZE);
  dim3 QKV_thread(TILE_SIZE, TILE_SIZE);
  NaiveMatrixMultiply<<<QKV_block, QKV_thread>>>(Q_KT, V, O, seq_len, seq_len, emb_size, scale_val);
  // TiledMatrixMultiply<<<QKV_block, QKV_thread>>>(Q_KT, V, O, seq_len, seq_len, emb_size, scale_val);
  cudaDeviceSynchronize();

  cudaFree(KT);
  cudaFree(Q_KT);
}

void test_attention(float *h_Q, float *h_K, float *h_V, float* h_O, int seq_len, int emb_size) {
  float *Q_device, *K_device, *V_device, *O_device;
  
  // Malloc device memory
  cudaMalloc((void **)&Q_device, sizeof(float) * seq_len * emb_size);
  cudaMalloc((void **)&K_device, sizeof(float) * seq_len * emb_size);
  cudaMalloc((void **)&V_device, sizeof(float) * seq_len * emb_size);
  cudaMalloc((void **)&O_device, sizeof(float) * seq_len * emb_size);

  // Copy data from host to device
  cudaMemcpy(K_device, h_K, sizeof(float) * seq_len * emb_size, cudaMemcpyHostToDevice);
  cudaMemcpy(Q_device, h_Q, sizeof(float) * seq_len * emb_size, cudaMemcpyHostToDevice);
  cudaMemcpy(V_device, h_V, sizeof(float) * seq_len * emb_size, cudaMemcpyHostToDevice);

  naive_self_attention(Q_device, K_device, V_device, O_device, seq_len, emb_size);

  // Result back to host
  cudaMemcpy(h_O, O_device, sizeof(float) * seq_len * emb_size, cudaMemcpyDeviceToHost);

  cudaFree(Q_device);
  cudaFree(K_device);
  cudaFree(V_device);
  cudaFree(O_device);
}

int main() {
  std::string directory = "flash_attention_dataset";
  std::vector<TestData> dataset = loadData(directory);
  int i = 0;
  for (const auto& basic_data : dataset) {
    float* Q = ConvertVectorToArray(basic_data.data);
    float* K = ConvertVectorToArray(basic_data.data);
    float* V = ConvertVectorToArray(basic_data.data);
    float* O = ConvertVectorToArray(basic_data.data);

    test_attention(Q, K, V, O, basic_data.seq_len, basic_data.emb_size);

    // PrintResult(result, basic_data.seq_len, basic_data.emb_size);

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;

    i += 1;
  }
  return 0;
}
