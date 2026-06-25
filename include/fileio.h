#pragma once

#include <vector>
#include <unordered_map>
#include <string>

std::string op2String(Oper op);

std::vector<float> readFloatsFromFile(std::string filename, size_t bytes_to_read);

void writeVectorToFile(const float *data, int N, const std::string& filename);

void generateAndSaveInput(Node* node);