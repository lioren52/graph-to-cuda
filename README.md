# graph-to-cuda
A minimal C++ neural network graph compiler focusing on operator fusion and CUDA code generation. Builds a custom Intermediate Representation (IR) to perform shape inference, topological sorting, and pattern-based fusion (e.g., MatMul+Bias+ReLU). Generates raw CUDA kernels to demonstrate low-level GPU memory mechanics without heavy frameworks.
