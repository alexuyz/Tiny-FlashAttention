#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

// Error checking macro for CUDA
#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      printf("CUDA_CHECK error in line %d of file %s: %s\n",                   \
             __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Struct to hold test data from files
struct TestData {
    int seq_len;
    int emb_size;
    std::vector<std::vector<float>> data; // Each sequence of embeddings
};

// Function to load test data from a directory
std::vector<TestData> loadData(const std::string& directory) {
    std::vector<TestData> dataset;

    for (const auto& seq_len : { 4, 8, 16, 32 }) {
        for (const auto& emb_size : { 64, 128, 256, 512, 1024 }) {
            std::stringstream filename;
            filename << directory << "/flash_attention_seq" << seq_len << "_emb" << emb_size << ".txt";

            std::ifstream file(filename.str());
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << filename.str() << std::endl;
                continue;
            }

            TestData test_data;
            test_data.seq_len = seq_len;
            test_data.emb_size = emb_size;

            std::string line;
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::vector<float> row;
                float value;
                while (ss >> value) {
                    row.push_back(value);
                }
                test_data.data.push_back(row);
            }
            dataset.push_back(test_data);
            file.close();
        }
    }

    return dataset;
}

__global__ void flash_attention_v1_kernel(float* Q, float* K, float* V, float* O, float* gMAX, float* gDenom, int seq_len, int emb_size, float sm_scale) {
    extern __shared__ float shared_mem[];
    float* sQ = shared_mem;
    float* sK = sQ + emb_size;
    float* sV = sK + emb_size;

    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前序列索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前嵌入维度索引

    if (row >= seq_len || col >= emb_size) return;

    sQ[col] = Q[row * emb_size + col];
    sK[col] = K[row * emb_size + col];
    sV[col] = V[row * emb_size + col];
    __syncthreads();

    float max_value = -INFINITY;
    float sum = 0.0f;

    for (int k = 0; k < seq_len; ++k) {
        float dot = 0.0f;
        for (int d = 0; d < emb_size; ++d) {
            dot += sQ[d] * sK[d]; // 点积计算
        }
        dot *= sm_scale;

        max_value = fmaxf(max_value, dot);
        sum += __expf(dot - max_value);
    }

    gMAX[row] = max_value;
    gDenom[row] = sum;

    O[row * emb_size + col] = __expf((sQ[col] * sK[col] * sm_scale) - max_value) / sum;
    __syncthreads();
}

void run_flash_attention(const TestData& test_data) {
    int seq_len = test_data.seq_len;
    int emb_size = test_data.emb_size;

    float* d_Q, * d_K, * d_V, * d_O, * d_MAX, * d_DENOM;

    size_t matrix_size = sizeof(float) * seq_len * emb_size;
    CUDA_CHECK(cudaMalloc(&d_Q, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_K, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_V, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_O, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_MAX, sizeof(float) * seq_len));
    CUDA_CHECK(cudaMalloc(&d_DENOM, sizeof(float) * seq_len));

    std::vector<float> flattened_matrix;
    for (const auto& row : test_data.data) {
        flattened_matrix.insert(flattened_matrix.end(), row.begin(), row.end());
    }
    CUDA_CHECK(cudaMemcpy(d_Q, flattened_matrix.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, flattened_matrix.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, flattened_matrix.data(), matrix_size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((emb_size + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
    size_t shared_memory_size = 4 * emb_size * sizeof(float);
    float sm_scale = 1.0f / sqrtf(static_cast<float>(emb_size));

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    flash_attention_v1_kernel << <grid, block, shared_memory_size >> > (d_Q, d_K, d_V, d_O, d_MAX, d_DENOM, seq_len, emb_size, sm_scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Print execution time in the desired format
    std::cout << "Seq len: " << seq_len
        << ", Emb size: " << emb_size
        << ", Execution Time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_MAX));
    CUDA_CHECK(cudaFree(d_DENOM));
}

int main() {
    std::string directory = "C:/Users/PhotonUser/My Files/OneDrive/Files/final/flash_attention_dataset";

    auto dataset = loadData(directory);
    if (dataset.empty()) {
        std::cerr << "No data loaded!" << std::endl;
        return EXIT_FAILURE;
    }

    for (const auto& test_data : dataset) {
        run_flash_attention(test_data);
    }

    return 0;
}
