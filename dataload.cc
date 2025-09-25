#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

#include "dataload.h"

std::vector<TestData> loadData(const std::string& directory) {
    std::vector<TestData> dataset;

    // Loop over all files in the directory (you may want to use filesystem API)
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
            std::getline(file, line);  // Skip the first line (header)

            // Read the data line by line
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

void printTestData(const std::vector<TestData>& dataset) {
    for (const auto& test_data : dataset) {
        std::cout << "Sequence Length: " << test_data.seq_len
            << ", Embedding Size: " << test_data.emb_size << std::endl;
        for (const auto& row : test_data.data) {
            for (const auto& value : row) {
                std::cout << std::fixed << std::setprecision(6) << value << " ";
            }
            std::cout << std::endl;
        }
    }
}

float* ConvertVectorToArray(std::vector<std::vector<float> > data) {
	int totalSize = 0;
    for (const auto& row : data) {
        totalSize += row.size();
    }

    float* linearArray = new float[totalSize];
    int index = 0;
    for (const auto& row : data) {
        for (const auto& value : row) {
            linearArray[index++] = value;
        }
    }
    return linearArray;
}

void print_matrix(float *matrix, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", matrix[i * n + j]);
    }
    printf("\n");
  }
}
