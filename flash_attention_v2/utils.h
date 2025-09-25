#pragma once
#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

// Data structure for input data
struct TestData {
    int seq_len;
    int emb_size;
    std::vector<std::vector<float>> data;  // Each sequence of embeddings
};

// Load dataset from files
std::vector<TestData> loadData(const std::string& directory) {
    std::vector<TestData> dataset;

    for (const auto& seq_len : { 4, 8, 16, 32 }) {
        for (const auto& emb_size : { ESIZE }) {
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
            std::getline(file, line);  // Skip header
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


inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

bool verification(float *A, float *B, int m, int n, float tol = 1e-5) {
    for (int i = 0; i < m * n; i++) {
        if (fabs(A[i] - B[i]) > tol) {
            printf("Mismatch at index %d: A[%d] = %f, B[%d] = %f\n", i, i, A[i], i, B[i]);
            return false;
        }
    }
    return true;
}

// print matrix
void print_host_matrix(float *matrix, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", matrix[i * n + j]);
    }
    printf("\n");
  }
}

void print_device_matrix(float *dev_ptr, int m, int n) {
  float *host_ptr = new float[m * n];
  cudaMemcpy(host_ptr, dev_ptr, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", host_ptr[i * n + j]);
    }
    printf("\n");
  }
  free(host_ptr);
}
#endif