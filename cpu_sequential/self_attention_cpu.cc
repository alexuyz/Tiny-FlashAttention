#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "../dataload.h"

using std::cout;
using std::endl;

float* gemm(float* A, float* B, int m, int n, int k) {
	float* C = new float[m * k];
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
				C[i * k + j] = 0;
				for (int p = 0; p < n; ++p) {
					C[i * k + j] += A[i * n + p] * B[p * k + j];
				}
		}
	}
	return C;
} 

float* transpose(float* A, int m, int n) {
	float* B = new float[n * m];
    for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				B[j * m + i] = A[i * n + j];
			}
    }
	return B;
} 

float* scaled_softmax(float* A, int m, int n, float scale_factor) {
	float* B = new float[m * n];
	for (int i = 0; i < m; ++i) {
		float max_val = A[i * n];
		for (int j = 1; j < n; ++j) {
			if (A[i * n + j] > max_val) {
				max_val = A[i * n + j];
			}
		}
		max_val = max_val * scale_factor;

		float sum = 0.0;
		for (int j = 0; j < n; ++j) {
			B[i * n + j] = exp((A[i * n + j] * scale_factor) - max_val);
			sum += B[i * n + j];
		}

		for (int j = 0; j < n; ++j) {
			B[i * n + j] /= sum;
		}
	}
	return B;
}

float* self_attn(float* Q, float* K, float* V, int seq_len, int dim) {
	float* K_T = transpose(K, seq_len, dim);
	float* QK_T = gemm(Q, K_T, seq_len, dim, seq_len);

	float scale_factor = 1.0 / sqrt(dim);
	float* attention_weights = scaled_softmax(QK_T, seq_len, seq_len, scale_factor);
	float* output = gemm(attention_weights, V, seq_len, seq_len, dim);

	delete[] K_T;
	delete[] QK_T;
	delete[] attention_weights;

	return output;
}

int main() {
    std::string directory = "../flash_attention_dataset";
    std::vector<TestData> dataset = loadData(directory);
	std::vector<float> running_time;
	for (const auto& basic_data : dataset) {
		float* Q = ConvertVectorToArray(basic_data.data);
		float* K = ConvertVectorToArray(basic_data.data);
		float* V = ConvertVectorToArray(basic_data.data);

		auto start = std::chrono::high_resolution_clock::now();
		float* result = self_attn(Q, K, V, basic_data.seq_len, basic_data.emb_size);
		auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;

		// print_matrix(result, basic_data.seq_len, basic_data.emb_size);
		cout << "size: (" << basic_data.seq_len << " " << basic_data.emb_size << "), running time: " << duration.count() << endl;
		running_time.push_back(duration.count());

		delete[] Q;
		delete[] K;
		delete[] V;
		delete[] result;	
    }
	return 0;
}
