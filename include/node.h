#pragma once

#include <vector>

enum class Oper {
    INPUT, MATMUL, ADD, ReLU
};


class Node {
public:
    int id;
    std::string name;
    Oper operation;
    std::vector<Node*> inputs;
    std::vector<int> shape;
};