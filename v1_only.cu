#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

constexpr int BlockRowSize = 4; // rows per block
constexpr int BlockColSize = 4;
using FP = float;

// CUDA error checking function
inline static cudaError_t checkCuda(cudaError_t result, const char* srcStr) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error (src: %s): %s\n", srcStr, cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

struct TestData {
    int seq_len;
    int emb_size;
    std::vector<std::vector<float>> data;
};

std::vector<TestData> loadData(const std::string& directory,
    const std::vector<int>& seq_lengths = { 4, 8, 16, 32 },
    const std::vector<int>& emb_sizes = { 64, 128, 256, 512, 1024 }) {
    std::vector<TestData> dataset;
    for (int seq_len : seq_lengths) {
        for (int emb_size : emb_sizes) {
            std::stringstream filename;
            filename << directory << "/flash_attention_seq" << seq_len << "_emb" << emb_size << ".txt";

            std::ifstream file(filename.str());
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << filename.str() << std::endl;
                continue;
            }

            TestData test_data{ seq_len, emb_size, {} };
            std::string line;
            std::getline(file, line);  // Skip header
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                test_data.data.push_back({ std::istream_iterator<float>(ss), std::istream_iterator<float>() });
            }
            dataset.push_back(std::move(test_data));
        }
    }
    return dataset;
}

__global__ void flash_attention_v1_kernel(FP* Q, FP* K, FP* V, FP* O, int seqLen, int embDim, FP scaleFactor) {
    extern __shared__ FP sharedMem[]; // shared memory
    FP* sharedQ = sharedMem; // shared memory for Q
    FP* sharedK = sharedQ + BlockRowSize * embDim; // shared memory for K
    FP* sharedV = sharedK + BlockColSize * embDim; // shared memory for V

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowIdx = blockIdx.y * blockDim.y + ty;

    if (rowIdx >= seqLen) return;

    // initialize shared memory for Q
    for (int i = tx; i < embDim; i += blockDim.x) {
        sharedQ[ty * embDim + i] = Q[rowIdx * embDim + i];
    }
    __syncthreads();

    FP localMax = -INFINITY;
    FP localDenom = 0.0f;
    FP partialSum = 0.0f;

    for (int k = 0; k < seqLen; k += BlockColSize) {
        if (k + tx < seqLen) {
            for (int i = 0; i < embDim; ++i) {
                sharedK[tx * embDim + i] = K[(k + tx) * embDim + i];
                sharedV[tx * embDim + i] = V[(k + tx) * embDim + i];
            }
        }
        __syncthreads();

        // compute QK
        FP qkSum = 0.0f;
        for (int i = 0; i < embDim; ++i) {
            qkSum += sharedQ[ty * embDim + i] * sharedK[tx * embDim + i];
        }
        qkSum *= scaleFactor;
        localMax = max(localMax, qkSum);
        FP expVal = exp(qkSum);
        localDenom += expVal;

        for (int i = 0; i < embDim; ++i) {
            partialSum += expVal * sharedV[tx * embDim + i];
        }
        __syncthreads();
    }

    // write final result back to global memory
    for (int i = tx; i < embDim; i += blockDim.x) {
        O[rowIdx * embDim + i] = partialSum / localDenom;
    }
}

void flash_attention_v1_cuda(FP* Q, FP* K, FP* V, FP* O, int seqLen, int embDim) {
    FP scaleFactor = 1.f / sqrtf(static_cast<FP>(embDim));
    size_t sharedMemSize = (BlockRowSize * embDim + 2 * BlockColSize * embDim) * sizeof(FP);

    dim3 block(BlockColSize, BlockRowSize);
    dim3 grid(1, (seqLen + BlockRowSize - 1) / BlockRowSize);

    flash_attention_v1_kernel << <grid, block, sharedMemSize >> > (Q, K, V, O, seqLen, embDim, scaleFactor);
    checkCuda(cudaDeviceSynchronize(), "flash_attention_v1_cuda");
}

void test_attention(const TestData& test_data) {
    int seq_len = test_data.seq_len;
    int emb_dim = test_data.emb_size;

    std::vector<FP> h_Q(seq_len * emb_dim, 1.0f);
    std::vector<FP> h_K(seq_len * emb_dim, 1.0f);
    std::vector<FP> h_V(seq_len * emb_dim, 1.0f);
    std::vector<FP> h_O(seq_len * emb_dim, 0.0f);

    FP* d_Q, * d_K, * d_V, * d_O;
    checkCuda(cudaMalloc(&d_Q, seq_len * emb_dim * sizeof(FP)), "cudaMalloc d_Q");
    checkCuda(cudaMalloc(&d_K, seq_len * emb_dim * sizeof(FP)), "cudaMalloc d_K");
    checkCuda(cudaMalloc(&d_V, seq_len * emb_dim * sizeof(FP)), "cudaMalloc d_V");
    checkCuda(cudaMalloc(&d_O, seq_len * emb_dim * sizeof(FP)), "cudaMalloc d_O");

    checkCuda(cudaMemcpy(d_Q, h_Q.data(), seq_len * emb_dim * sizeof(FP), cudaMemcpyHostToDevice), "cudaMemcpy h_Q");
    checkCuda(cudaMemcpy(d_K, h_K.data(), seq_len * emb_dim * sizeof(FP), cudaMemcpyHostToDevice), "cudaMemcpy h_K");
    checkCuda(cudaMemcpy(d_V, h_V.data(), seq_len * emb_dim * sizeof(FP), cudaMemcpyHostToDevice), "cudaMemcpy h_V");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    flash_attention_v1_cuda(d_Q, d_K, d_V, d_O, seq_len, emb_dim);
    cudaEventRecord(stop);

    checkCuda(cudaMemcpy(h_O.data(), d_O, seq_len * emb_dim * sizeof(FP), cudaMemcpyDeviceToHost), "cudaMemcpy h_O");

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << std::fixed << std::setprecision(3)
        << "SeqLen: " << seq_len << ", EmbDim: " << emb_dim
        << "  V1 Execution Time: " << milliseconds << " ms\n";

    checkCuda(cudaFree(d_Q), "cudaFree d_Q");
    checkCuda(cudaFree(d_K), "cudaFree d_K");
    checkCuda(cudaFree(d_V), "cudaFree d_V");
    checkCuda(cudaFree(d_O), "cudaFree d_O");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::string dataset_dir = "C:/Users/PhotonUser/My Files/OneDrive/Files/final/flash_attention_dataset";

    std::vector<TestData> dataset = loadData(dataset_dir);

    for (const auto& test_data : dataset) {
        test_attention(test_data);
    }

    return 0;
}
