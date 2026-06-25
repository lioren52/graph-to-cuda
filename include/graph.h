#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "node.h"
#include "fileio.h"

class Graph {
    std::vector<std::unique_ptr<Node>> nodes;
    Node* outputNode;
    std::vector<Node*> sorted;
public: 
    float* bufferAlloc(Node* node);

    std::unordered_map<int, float*> nodeMem();

    void generator();

    void execute();

    void setOutput(Node* node);

    Node* getOutput();

    Node* addInput(std::string nm, std::vector<int> shp);

    Node* addNode(std::string nm, Oper op, std::vector<Node*> in);

    void printGraph();

    std::vector<Node*> topoSort();
};