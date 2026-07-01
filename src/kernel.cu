__global__ void matrixMul(float* A, float* B, float* C, int row_A, int N, int col_B) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    if (Row < row_A && Col < col_B) {
        float pValue = 0;
        for (int i = 0; i < N; i++) {
            pValue += A[(Row*N) + i] * B[Col + (i*col_B)];
        }
        C[(Row*col_B) + Col] = pValue;
    }
}

__global__ void matrixAdd(float *A, float *B, float *C, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {

        int index = row * width + col;
        
        C[index] = A[index] + B[index];
    }
}

__global__ void matrixReLU(const float *A, float *C, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        
        C[index] = fmaxf(0.0f, A[index]); 
    }
}

void matMul(float* A, float* B, float* C, int row_A, int N, int col_B) {
    dim3 threadPerBlock(16, 16);
    dim3 blocks((col_B + 15) / 16, (row_A + 15) / 16);

    matrixMul<<<blocks, threadPerBlock>>>(A, B, C, row_A, N, col_B);
}

void matAdd(float *A, float *B, float *C, int height, int width) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, height, width);
}

void matReLU(float *A, float *C, int height, int width) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(A, C, height, width);
}

// =====================================================================
// 1. FUSED: Matrix Multiplication + ReLU (MR)
// =====================================================================
__global__ void matrixMulReLU(float* A, float* B, float* C, int row_A, int N, int col_B) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    if (Row < row_A && Col < col_B) {
        float pValue = 0;
        for (int i = 0; i < N; i++) {
            pValue += A[(Row*N) + i] * B[Col + (i*col_B)];
        }
        // FUSE: Apply ReLU immediately before writing to VRAM
        C[(Row*col_B) + Col] = fmaxf(0.0f, pValue);
    }
}

void matMulReLU(float* A, float* B, float* C, int row_A, int N, int col_B) {
    dim3 threadPerBlock(16, 16);
    dim3 blocks((col_B + 15) / 16, (row_A + 15) / 16);

    matrixMulReLU<<<blocks, threadPerBlock>>>(A, B, C, row_A, N, col_B);
}

// =====================================================================
// 2. FUSED: Matrix Addition + ReLU (AR)
// =====================================================================
__global__ void matrixAddReLU(float *A, float *B, float *C, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        
        // FUSE: Add and apply ReLU in one memory operation
        float val = A[index] + B[index];
        C[index] = fmaxf(0.0f, val);
    }
}

void matAddReLU(float *A, float *B, float *C, int height, int width) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    matrixAddReLU<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, height, width);
}

// =====================================================================
// 3. FUSED: Matrix Multiplication + Addition (MA)
// =====================================================================
__global__ void matrixMulAdd(float* A, float* B, float* Bias, float* C, int row_A, int N, int col_B) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    if (Row < row_A && Col < col_B) {
        float pValue = 0;
        for (int i = 0; i < N; i++) {
            pValue += A[(Row*N) + i] * B[Col + (i*col_B)];
        }
        
        // FUSE: Add the bias/matrix immediately.
        // Assuming Bias matches the output dimensions [row_A, col_B] exactly.
        C[(Row*col_B) + Col] = pValue + Bias[(Row*col_B) + Col];
    }
}

void matMulAdd(float* A, float* B, float* Bias, float* C, int row_A, int N, int col_B) {
    dim3 threadPerBlock(16, 16);
    dim3 blocks((col_B + 15) / 16, (row_A + 15) / 16);

    matrixMulAdd<<<blocks, threadPerBlock>>>(A, B, Bias, C, row_A, N, col_B);
}

// =====================================================================
// 4. FUSED: Matrix Multiplication + Addition + ReLU (MAR)
// =====================================================================
__global__ void matrixMulAddReLU(float* A, float* B, float* Bias, float* C, int row_A, int N, int col_B) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    if (Row < row_A && Col < col_B) {
        float pValue = 0;
        
        // 1. Execute Matrix Multiplication
        for (int i = 0; i < N; i++) {
            pValue += A[(Row*N) + i] * B[Col + (i*col_B)];
        }
        
        // 2. Execute Addition
        float valWithBias = pValue + Bias[(Row*col_B) + Col];
        
        // 3. Execute ReLU & Write Back to VRAM
        C[(Row*col_B) + Col] = fmaxf(0.0f, valWithBias);
    }
}

void matMulAddReLU(float* A, float* B, float* Bias, float* C, int row_A, int N, int col_B) {
    dim3 threadPerBlock(16, 16);
    dim3 blocks((col_B + 15) / 16, (row_A + 15) / 16);

    matrixMulAddReLU<<<blocks, threadPerBlock>>>(A, B, Bias, C, row_A, N, col_B);
}