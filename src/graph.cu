#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <queue>
#include <fstream>
#include <unordered_map>
#include <random>
#include <fileio.h>
#include <graph.h>
#include <node.h>


float* Graph::bufferAlloc(Node* node) {
    float* address;
    size_t bytesNode = node->shape[0] * node->shape[1];

    cudaMalloc((void**)&address, bytesNode * sizeof(float));

    return address;
}

std::unordered_map<int, float*> Graph::nodeMem() {
    std::unordered_map<int, float*> mp;


    for (const std::unique_ptr<Node>& node : nodes) {
        Node* raw = node.get();
        mp[raw->id] = bufferAlloc(raw);
    }

    return mp;
}

void Graph::generator() {
    for (Node* item : sorted) {
        if (item->operation == Oper::INPUT) {
            generateAndSaveInput(item);
        }
    }
}

void Graph::execute() {
    std::unordered_map<int, float*> nodeMemMap = nodeMem();
    int inputTill = 0;

    for (int i = 0; i < sorted.size(); i++) {
        if (sorted[i]->operation == Oper::INPUT) {
            size_t byteSize = sorted[i]->shape[0] * sorted[i]->shape[1] * sizeof(float);
            std::vector<float> cont = readFloatsFromFile(sorted[i]->name+".bin", byteSize);

            cudaMemcpy(nodeMemMap[sorted[i]->id], cont.data(), byteSize, cudaMemcpyHostToDevice);
            std::cout << "Inputing from " << sorted[i]->name << std::endl;
        } else {
            inputTill = i;
            break;
        }
    }

    for (int i = inputTill; i < sorted.size(); i++) {
        if (sorted[i]->operation == Oper::MATMUL) {
            int row_A = sorted[i]->inputs[0]->shape[0];
            int N     = sorted[i]->inputs[0]->shape[1];
            int col_B = sorted[i]->inputs[1]->shape[1];
            dim3 threadPerBlock(16, 16);
            dim3 blocks((col_B + 15) / 16, (row_A + 15) / 16);
            matrixMul<<<blocks, threadPerBlock>>>(
                nodeMemMap[sorted[i]->inputs[0]->id],
                nodeMemMap[sorted[i]->inputs[1]->id],
                nodeMemMap[sorted[i]->id],
                row_A, N, col_B
            );
        } else if (sorted[i]->operation == Oper::ADD) {
            int height = sorted[i]->shape[0];
            int width  = sorted[i]->shape[1];
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
            matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(
                nodeMemMap[sorted[i]->inputs[0]->id],
                nodeMemMap[sorted[i]->inputs[1]->id],
                nodeMemMap[sorted[i]->id],
                height, width
            );
        } else if (sorted[i]->operation == Oper::ReLU) {
            int height = sorted[i]->shape[0];
            int width  = sorted[i]->shape[1];
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

            matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(
                nodeMemMap[sorted[i]->inputs[0]->id],
                nodeMemMap[sorted[i]->id],
                height, width
            );
        }

        std::cout << "At " << sorted[i]->name << std::endl;
    }

    Node* node = getOutput();
    std::vector<float> output(node->shape[0] * node->shape[1]);
    std::cout << std::endl;
    std::cout << "Final Node: " << node->name << std::endl;
    cudaMemcpy(output.data(), nodeMemMap[node->id], node->shape[0] * node->shape[1] * sizeof(float), cudaMemcpyDeviceToHost);
    writeVectorToFile(output.data(), output.size(), node->name);
}


void Graph::setOutput(Node* node) {
    outputNode = node;
}

Node* Graph::getOutput() {
    return outputNode;
}


Node* Graph::addInput(std::string nm, std::vector<int> shp) {
    std::unique_ptr<Node> newNode = std::make_unique<Node>();
    newNode->id = nodes.size();
    newNode->name = nm;
    newNode->shape = shp;
    newNode->operation = Oper::INPUT;
    Node* raw = newNode.get();
    nodes.push_back(std::move(newNode));

    return raw;
}

Node* Graph::addNode(std::string nm, Oper op, std::vector<Node*> in) {
    std::vector<int> sp;
    if (op == Oper::MATMUL) {
        if (in.size() != 2) {
            std::cout << "Error: input for MatMul is only 1 node\n";
            return nullptr;
        }

        if (in[0]->shape[1] != in[1]->shape[0]) {
            std::cout << "Error: Dimensions don't match for Matrix Multiplication\n";
            return nullptr;
        }

        sp = std::vector<int>({in[0]->shape[0], in[1]->shape[1]});
    }

    if (op == Oper::ADD) {
        if (in.size() != 2) {
            std::cout << "Error: input for addition is given: " << in.size() << "\n";
            return nullptr;
        }
        if (in[0]->shape[0] != in[1]->shape[0] || in[0]->shape[1] != in[1]->shape[1]) {
            std::cout << "Error: ADD shape mismatch\n";
            return nullptr;
        }
        sp = in[0]->shape;
    }

    if (op == Oper::ReLU) {
        if (in.size() != 1) {
            std::cout << "Error: ReLU is given input: " << in.size() << "\n";
            return nullptr;
        }
        sp = std::vector<int>({in[0]->shape[0], in[0]->shape[1]});
    }

    std::unique_ptr<Node> newNode = std::make_unique<Node>();
    newNode->id = nodes.size();
    newNode->name = nm;
    newNode->operation = op;
    newNode->shape = sp;
    newNode->inputs = in;

    Node* raw = newNode.get();
    nodes.push_back(std::move(newNode));

    return raw;
}

void Graph::printGraph() {
    for (auto& item : nodes) {
        std::cout << "Node Name: " << item->name << "\n";
        std::cout << "ID: " << item->id << "\n";
        std::cout << "Operation: " << op2String(item->operation) << "\n";
        std::cout << "Shape: (";
        for (int i = 0; i < item->shape.size(); i++) {
            std::cout << item->shape[i];
            if (i < item->shape.size() - 1) std::cout << ", ";
        }
        std::cout << ")\n";
        if (!item->inputs.empty()) {
            std::cout << "Input Nodes: ";
            for (Node* node : item->inputs) {
                std::cout << node->name << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        std::cout << std::endl;
        std::cout << std::endl;
    }
}

std::vector<Node*> Graph::topoSort() {
    int nodesNum = nodes.size();
    std::vector<int> indegree(nodesNum);
    std::unordered_map<int, std::vector<Node*>> adjList;

    for (int i = 0; i < nodesNum; i++) {
        Node* node = nodes[i].get();
        indegree[i] = node->inputs.size();
        for (Node* item : node->inputs) {
            adjList[item->id].push_back(node);
        }
    }

    std::queue<Node*> que;
    for (int i = 0; i < nodesNum; i++) {
        if (indegree[i] == 0) {
            Node* node = nodes[i].get();
            que.push(node);
        }
    }

    std::vector<Node*> topo;
    while (!que.empty()) {
        Node* node = que.front();
        que.pop();
        topo.push_back(node);

        for (Node* item : adjList[node->id]) {
            indegree[item->id]--;
            if (indegree[item->id] == 0) {
                que.push(item);
            }
        }
    }

    if (topo.size() != nodesNum) {
        std::cout << "Error: cycle detected in graph, topological sort incomplete\n";
        return {};
    }

    for (Node* item : topo) {
        sorted.push_back(item);
    }

    return topo;
}

