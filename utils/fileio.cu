#include <vector>
#include <unordered_map>
#include <fstream>
#include <string>
#include <random>
#include <memory>
#include <iostream>
#include "fileio.h"
#include "graph.h"
#include "node.h"

std::string op2String(Oper op) {
    switch (op) {
        case Oper::INPUT : return "INPUT";
        case Oper::MATMUL : return "MATMUL";
        case Oper::ADD : return "ADD";
        case Oper::ReLU : return "ReLU";
        default: return "Undefined";
    }
}

std::vector<float> readFloatsFromFile(std::string filename, size_t bytes_to_read) {
    std::vector<float> data;

    size_t num_floats = bytes_to_read / sizeof(float);

    if (num_floats == 0) {
        std::cerr << "Warning: Requested bytes (" << bytes_to_read 
                  << ") is smaller than the size of one float." << std::endl;
        return data;
    }

    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    data.resize(num_floats);

    file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));

    std::streamsize bytes_read = file.gcount();
    size_t floats_read = bytes_read / sizeof(float);

    if (floats_read < num_floats) {
        data.resize(floats_read);
    }

    return data;
}

void writeVectorToFile(const float *data, int N, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    for (int i = 0; i < N; i++) {
        outFile << data[i] << "\n";
    }

    outFile.close();
    std::cout << "Successfully wrote " << N << " elements to " << filename << std::endl;
}

void generateAndSaveInput(Node* node) {
    int size = 1;
    std::mt19937 rng(42);
    for (int d : node->shape) size *= d;

    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> cpuBuf(size);
    for (float& f : cpuBuf) f = dist(rng);

    std::string filename = node->name + ".bin";
    std::ofstream file(filename, std::ios::binary);
    file.write((char*)cpuBuf.data(), size * sizeof(float));
}