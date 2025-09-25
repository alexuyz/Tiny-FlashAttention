#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

struct TestData {
    int seq_len;
    int emb_size;
    std::vector<std::vector<float> > data;  // Each sequence of embeddings
};

std::vector<TestData> loadData(const std::string& directory);

void printTestData(const std::vector<TestData>& dataset);

float* ConvertVectorToArray(std::vector<std::vector<float> > data);

void print_matrix(float *matrix, int m, int n);
