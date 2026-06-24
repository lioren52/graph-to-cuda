#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <queue>
#include <fstream>
#include <unordered_map>
#include "kernel.cu"
#include <random>

enum class Oper {
    INPUT, MATMUL, ADD, ReLU
};

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

    // Calculate how many complete floats fit into the requested byte count
    size_t num_floats = bytes_to_read / sizeof(float);

    // If the bytes requested are smaller than a single float, return empty
    if (num_floats == 0) {
        std::cerr << "Warning: Requested bytes (" << bytes_to_read 
                  << ") is smaller than the size of one float." << std::endl;
        return data;
    }

    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    // Pre-allocate the vector to avoid unnecessary reallocations
    data.resize(num_floats);

    // Read the data directly into the vector's underlying memory buffer
    file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));

    // Check how many bytes were *actually* read (in case the file is smaller than requested)
    std::streamsize bytes_read = file.gcount();
    size_t floats_read = bytes_read / sizeof(float);

    // If the file ended early, shrink the vector down to what we actually grabbed
    if (floats_read < num_floats) {
        data.resize(floats_read);
    }

    return data;
}

void writeVectorToFile(const float *data, int N, const std::string& filename) {
    // Open a file output stream
    std::ofstream outFile(filename);

    // Check if the file opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    // Write each element to a new line
    for (int i = 0; i < N; i++) {
        outFile << data[i];
    }

    // Close the file to free up system resources
    outFile.close();
    std::cout << "Successfully wrote " << N << " elements to " << filename << std::endl;
}

class Node {
public:
    int id;
    std::string name;
    Oper operation;
    std::vector<Node*> inputs;
    std::vector<int> shape;
};


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

class Graph {
    std::vector<std::unique_ptr<Node>> nodes;
    Node* outputNode = nullptr;
    std::vector<Node*> sorted;

public:
    float* bufferAlloc(Node* node) {
        float* address;
        size_t bytesNode = node->shape[0] * node->shape[1];

        cudaMalloc((void**)&address, bytesNode * sizeof(float));

        return address;
    }

    std::unordered_map<int, float*> nodeMem() {
        std::unordered_map<int, float*> mp;


        for (const std::unique_ptr<Node>& node : nodes) {
            Node* raw = node.get();
            mp[raw->id] = bufferAlloc(raw);
        }

        return mp;
    }

    void generator() {
        for (Node* item : sorted) {
            if (item->operation == Oper::INPUT) {
                generateAndSaveInput(item);
            }
        }
    }

    void execute() {
        std::unordered_map<int, float*> nodeMemMap = nodeMem();
        int inputTill = 0;

        for (int i = 0; i < sorted.size(); i++) {
            if (sorted[i]->operation == Oper::INPUT) {
                size_t byteSize = sorted[i]->shape[0] * sorted[i]->shape[1] * sizeof(float);
                std::vector<float> cont = readFloatsFromFile(sorted[i]->name, byteSize);

                cudaMemcpy(nodeMemMap[sorted[i]->id], cont.data(), byteSize, cudaMemcpyHostToDevice);
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
        }

        Node* node = getOutput();
        std::vector<float> output(node->shape[0] * node->shape[1]);
        cudaMemcpy(output.data(), nodeMemMap[node->id], node->shape[0] * node->shape[1] * sizeof(float), cudaMemcpyDeviceToHost);
        writeVectorToFile(output.data(), output.size(), node->name);
    }


    void setOutput(Node* node) {
        outputNode = node;
    }

    Node* getOutput() {
        return outputNode;
    }


    Node* addInput(std::string nm, std::vector<int> shp) {
        std::unique_ptr<Node> newNode = std::make_unique<Node>();
        newNode->id = nodes.size();
        newNode->name = nm;
        newNode->shape = shp;
        newNode->operation = Oper::INPUT;
        Node* raw = newNode.get();
        nodes.push_back(std::move(newNode));

        return raw;
    }

    Node* addNode(std::string nm, Oper op, std::vector<Node*> in) {
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

    void printGraph() {
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

    std::vector<Node*> topoSort() {
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
};

int main() {
    Graph graph;

    // Layer 1 inputs
    Node* weights1 = graph.addInput("Weights_Layer_1", {16, 784});
    Node* inputs   = graph.addInput("Inputs", {784, 1});
    Node* bias1    = graph.addInput("Bias_Layer_1", {16, 1});

    // Layer 2 inputs
    Node* weights2 = graph.addInput("Weights_Layer_2", {16, 16});
    Node* bias2    = graph.addInput("Bias_Layer_2", {16, 1});

    // Residual projection inputs
    Node* weights_proj = graph.addInput("Weights_Proj", {16, 16});
    Node* bias_proj    = graph.addInput("Bias_Proj", {16, 1});

    // Layer 1 forward
    Node* matmul1  = graph.addNode("MatMul_Layer_1", Oper::MATMUL, {weights1, inputs});
    Node* add1     = graph.addNode("Add_Layer_1", Oper::ADD, {matmul1, bias1});
    Node* relu1    = graph.addNode("ReLU_Layer_1", Oper::ReLU, {add1});

    // Main path: Layer 2
    Node* matmul2  = graph.addNode("MatMul_Layer_2", Oper::MATMUL, {weights2, relu1});
    Node* add2     = graph.addNode("Add_Layer_2", Oper::ADD, {matmul2, bias2});
    Node* relu2    = graph.addNode("ReLU_Layer_2", Oper::ReLU, {add2});

    // Residual path: separate projection of relu1 (branches off same node)
    Node* matmul_proj = graph.addNode("MatMul_Proj", Oper::MATMUL, {weights_proj, relu1});
    Node* add_proj    = graph.addNode("Add_Proj", Oper::ADD, {matmul_proj, bias_proj});

    // Merge: add main path + residual path back together
    Node* residual_add = graph.addNode("Residual_Add", Oper::ADD, {relu2, add_proj});
    Node* relu_out     = graph.addNode("ReLU_Out", Oper::ReLU, {residual_add});

    graph.setOutput(relu_out);
    std::vector<Node*> topoS = graph.topoSort();
    std::cout << "------------Topological Sort------------" << std::endl;
    std::vector<Node*> topo = graph.topoSort();
    for (Node* item : topo) {
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
    }

    std::cout << "Generator" << std::endl;
    graph.generator();
    std::cout << "Execution" << std::endl;
    graph.execute();



    return 0;
}