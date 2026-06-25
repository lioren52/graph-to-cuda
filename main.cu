#include <iostream>
#include <vector>
#include <fileio.h>
#include <graph.h>
#include <node.h>

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


    std::cout << "Generating values" << std::endl;
    graph.generator();
    std::cout << "Execution" << std::endl;
    graph.execute();



    return 0;
}