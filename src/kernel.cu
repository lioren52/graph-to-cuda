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
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, B, height, width);
}

void matReLU(float *A, float *C, int height, int width) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(A, C, height, width);
}