void matMul(float* A, float* B, float* C, int row_A, int N, int col_B);

void matAdd(float *A, float *B, float *C, int height, int width);

void matReLU(float *A, float *C, int height, int width);

void matMulReLU(float* A, float* B, float* C, int row_A, int N, int col_B);

void matAddReLU(float *A, float *B, float *C, int height, int width);

void matMulAdd(float* A, float* B, float* Bias, float* C, int row_A, int N, int col_B);

void matMulAddReLU(float* A, float* B, float* Bias, float* C, int row_A, int N, int col_B);