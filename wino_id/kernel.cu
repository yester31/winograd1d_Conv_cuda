#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>
#include <cublas_v2.h>
using namespace std;

#define d(input, i, Inz) ( input[Inz + i * C ] )
__global__ void kernel_winograd_Btd_1x3( float* __restrict__ pOutputs, float* __restrict__ pInputs, int N, int C, int H, int W, int left, int top, int TileW, int TileH)
{
	int tw_idx = blockIdx.x % TileW; // tile 수평 방향 인덱스
	int th_idx = blockIdx.x / TileW; // tile 수직 방향 인덱스 
	int b_idx = blockIdx.y; // N
	int p_idx = blockIdx.z;

	extern __shared__ float input[];

	int c_idx = threadIdx.x + p_idx * blockDim.x; // C
	int h_idx = th_idx ;						  // H
	int w_idx = tw_idx * 4 + threadIdx.y - left;  // W

	float v = 0.f; // padding

	if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
		v = pInputs[b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx];
	}

	input[c_idx + threadIdx.y * C] = v;

	__syncthreads();

	float BTd[6];

	BTd[0] = d(input, 0, c_idx) * 4 - d(input, 2, c_idx) * 5 + d(input, 4, c_idx);
	BTd[1] = -d(input, 1, c_idx) * 4 - d(input, 2, c_idx) * 4 + d(input, 3, c_idx) + d(input, 4, c_idx);
	BTd[2] = d(input, 1, c_idx) * 4 - d(input, 2, c_idx) * 4 - d(input, 3, c_idx) + d(input, 4, c_idx);
	BTd[3] = - d(input, 1, c_idx) * 2 - d(input, 2, c_idx) + d(input, 3, c_idx) * 2 + d(input, 4, c_idx);
	BTd[4] = d(input, 1, c_idx) * 2 - d(input, 2, c_idx) - d(input, 3, c_idx) * 2 + d(input, 4, c_idx);
	BTd[5] = d(input, 1, c_idx) * 4 - d(input, 3, c_idx) * 5 + d(input, 5, c_idx);

	//__syncthreads();

	int o_base = threadIdx.y * TileH * TileW * C * N + b_idx * TileH * TileW * C + blockIdx.x * C + threadIdx.x + p_idx * blockDim.x;
		
	pOutputs[o_base] = BTd[threadIdx.y];

}


void winograd_Btd_1x3( float* output, float* data, int N, int C, int H, int W, int left, int top, int TileW, int TileH, cudaStream_t stream)
{
		
		int threadX;
		int threadY = 6; // parted channel related

		int blockX = TileW * TileH;  // Total Tile related
		int blockY = N;// equal batch
		int blockZ;  // parted channel related

		if (C <= 170) {
			threadX = C; blockZ = 1;
		}
		else if (C == 224 || C == 256 || C == 192) {
			threadX = C / 2;
			blockZ = 2;
		}
		else if (C == 512) {
			threadX = C / 4;
			blockZ = 4;
		}
		else if (C == 1024) {
			threadX = C / 8;
			blockZ = 8;
		}
		else {
			printf("Input Data Channel=%d is not supported for WinoGrad transform\n", C);
			exit(-1);
		}

		dim3 grid(blockX, blockY, blockZ);
		dim3 block(threadX, threadY);
		int shared = (threadY * C) * sizeof(float);//??
		kernel_winograd_Btd_1x3 << < grid, block, shared, stream >> > ( output, data, N, C, H, W, left, top, TileW, TileH); // pH = H + top + bottom, pW = W + left + right
}


__global__ void kernel_winograd_Atgd_1x3(float* __restrict__ pOutputs, float* __restrict__ pInputs, float* __restrict__ pBiases, int P, int Q, int TileH, int TileW, float scale, int activation)
{
	int tw_idx = blockIdx.x % TileW; // 출력값의 수평 방향 인덱스
	int th_idx = blockIdx.x / TileW; // 출력값의 수직 방향 인덱스

	extern __shared__ float input[];

	int s_idx = (threadIdx.x) * gridDim.z * gridDim.x * gridDim.y + blockIdx.y * TileH * TileW * gridDim.z + (tw_idx + th_idx * TileW) *  gridDim.z + blockIdx.z;

	input[threadIdx.x] = pInputs[s_idx];

	__syncthreads();

	float tmp[4];

		tmp[0] = input[0] + input[1] +	    input[2] +     input[3] + input[4];
		tmp[1] = input[1] - input[2] + 2 * input[3] - 2 * input[4];
		tmp[2] = input[1] + input[2] + 4 * input[3] + 4 * input[4];
		tmp[3] = input[1] - input[2] + 8 * input[3] - 8 * input[4] + input[5];
	
	__syncthreads();


	tmp[threadIdx.y] *= scale; // scale
	float b = pBiases == nullptr ? 0.f : pBiases[blockIdx.z];
	tmp[threadIdx.y] += b; // bias
	tmp[threadIdx.y] = (activation == 1 && tmp[threadIdx.y] < 0) ? 0 : tmp[threadIdx.y]; // relu

	int nkpg_idx = blockIdx.y * gridDim.z * P * Q + blockIdx.z * P * Q + th_idx * Q + tw_idx * 4 + threadIdx.y;

	pOutputs[nkpg_idx] = tmp[threadIdx.y];
}

cublasStatus_t SgemmStridedBatched(
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp,
	const float* dev_A, int WidthA, int HeightA, long long strideA,
	const float* dev_B, int WidthB, int HeightB, long long strideB,
	float *dev_C, long long strideC, int batchCount, float Alpha = 1.0f, float Beta = 0.0f)
{
	int lda = WidthA;
	int ldb = WidthB;

	if (AOp != CUBLAS_OP_N) {
		int tmp = WidthA;
		WidthA = HeightA;
		HeightA = tmp;
	}
	if (BOp != CUBLAS_OP_N) {
		int tmp = WidthB;
		WidthB = HeightB;
		HeightB = tmp;
	}

	int m = WidthB;
	int n = HeightA;
	int k = WidthA;

	return cublasSgemmStridedBatched(Blas, BOp, AOp, m, n, k, &Alpha, dev_B, ldb, strideB, dev_A, lda, strideA, &Beta, dev_C, m, strideC, batchCount);
}

void winograd_Atgd_1x3(float* output, float* data, float* bias, int N, int K, int P, int Q, int TileW, int TileH, float scale, int activation, cudaStream_t stream)
{
	int threadx = 6; 
	int blockX = TileH * TileW;  // Total Tile related
	int blockY = N; // equal batch
	int blockZ = K; // output channel

	dim3 grid(blockX, blockY, blockZ);
	dim3 block(threadx,4);
	int shared = (6) * sizeof(float);

	kernel_winograd_Atgd_1x3 << < grid, block, shared, stream >> > (output, data, bias, P, Q, TileH, TileW, scale, activation);
}

__global__ void kernel_winograd_Btd_3x1 (float* __restrict__ pOutputs, float* __restrict__ pInputs, int N, int C, int H, int W, int left, int top, int TileW, int TileH)
{
	int tw_idx = blockIdx.x % TileW; // tile 수평 방향 인덱스
	int th_idx = blockIdx.x / TileW; // tile 수직 방향 인덱스 
	int b_idx = blockIdx.y; // N
	int p_idx = blockIdx.z;

	extern __shared__ float input[];

	int c_idx = threadIdx.x + p_idx * blockDim.x;	// C
	int h_idx = (th_idx * 4 - top) + threadIdx.y;	// H
	int w_idx = tw_idx;					// W

	float v = 0.f; // padding

	if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
		v = pInputs[b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx];
	}

	input[c_idx + threadIdx.y * C] = v;

	__syncthreads();

	float BTd[6];

	BTd[0] = d(input, 0, c_idx) * 4 - d(input, 2, c_idx) * 5 + d(input, 4, c_idx);
	BTd[1] = -d(input, 1, c_idx) * 4 - d(input, 2, c_idx) * 4 + d(input, 3, c_idx) + d(input, 4, c_idx);
	BTd[2] = d(input, 1, c_idx) * 4 - d(input, 2, c_idx) * 4 - d(input, 3, c_idx) + d(input, 4, c_idx);
	BTd[3] = -d(input, 1, c_idx) * 2 - d(input, 2, c_idx) + d(input, 3, c_idx) * 2 + d(input, 4, c_idx);
	BTd[4] = d(input, 1, c_idx) * 2 - d(input, 2, c_idx) - d(input, 3, c_idx) * 2 + d(input, 4, c_idx);
	BTd[5] = d(input, 1, c_idx) * 4 - d(input, 3, c_idx) * 5 + d(input, 5, c_idx);

	int o_base = threadIdx.y * TileH * TileW * C * N + b_idx * TileH * TileW * C + blockIdx.x * C + threadIdx.x + p_idx * blockDim.x;

	pOutputs[o_base] = BTd[threadIdx.y];

}


void winograd_Btd_3x1(float* output, float* data, int N, int C, int H, int W, int left, int top, int TileW, int TileH, cudaStream_t stream)
{
	int threadX;
	int threadY = 6; // parted channel related
	int blockX = TileW * TileH;  // Total Tile related
	int blockY = N;// equal batch
	int blockZ;  // parted channel related

	if (C <= 170) {
		threadX = C; blockZ = 1;
	}
	else if (C == 224 || C == 256 || C == 192) {
		threadX = C / 2;
		blockZ = 2;
	}
	else if (C == 512) {
		threadX = C / 4;
		blockZ = 4;
	}
	else if (C == 1024) {
		threadX = C / 8;
		blockZ = 8;
	}
	else {
		printf("Input Data Channel=%d is not supported for WinoGrad transform\n", C);
		exit(-1);
	}

	dim3 grid(blockX, blockY, blockZ);
	dim3 block(threadX, threadY);
	int shared = (threadY * C) * sizeof(float);

	kernel_winograd_Btd_3x1 << < grid, block, shared, stream >> > (output, data, N, C, H, W, left, top, TileW, TileH); // pH = H + top + bottom, pW = W + left + right
}


// [TileH*TileW, N, K] [6,6] [6*6*sizeof(float)]
__global__ void kernel_winograd_Atgd_3x1(float* __restrict__ pOutputs, float* __restrict__ pInputs, float* __restrict__ pBiases, int P, int Q, int TileH, int TileW, float scale, int activation)
{
	int tw_idx = blockIdx.x % TileW; // 출력값의 수평 방향 인덱스
	int th_idx = blockIdx.x / TileW; // 출력값의 수직 방향 인덱스

	extern __shared__ float input[];

	int s_idx = (threadIdx.x) * gridDim.z * gridDim.x * gridDim.y + blockIdx.y * TileH * TileW * gridDim.z + (tw_idx + th_idx * TileW) *  gridDim.z + blockIdx.z;

	input[threadIdx.x] = pInputs[s_idx];

	__syncthreads();

	float tmp[4];

	tmp[0] = input[0] + input[1] + input[2] + input[3] + input[4];
	tmp[1] = input[1] - input[2] + 2 * input[3] - 2 * input[4];
	tmp[2] = input[1] + input[2] + 4 * input[3] + 4 * input[4];
	tmp[3] = input[1] - input[2] + 8 * input[3] - 8 * input[4] + input[5];

	__syncthreads();


	tmp[threadIdx.y] *= scale; // scale
	float b = pBiases == nullptr ? 0.f : pBiases[blockIdx.z];
	tmp[threadIdx.y] += b; // bias
	tmp[threadIdx.y] = (activation == 1 && tmp[threadIdx.y] < 0) ? 0 : tmp[threadIdx.y]; // relu

	int q_idx = tw_idx;
	int p_idx = th_idx * 4;

	int nkpg_idx = blockIdx.y * gridDim.z * P * Q + blockIdx.z * P * Q + (p_idx + threadIdx.y) * Q + q_idx;

	pOutputs[nkpg_idx] = tmp[threadIdx.y];
}


void winograd_Atgd_3x1(float* output, float* data, float* bias, int N, int K, int P, int Q, int TileW, int TileH, float scale, int activation, cudaStream_t stream)
{

	int threadx = 6;
	int blockX = TileH * TileW;  // Total Tile related
	int blockY = N; // equal batch
	int blockZ = K; // output channel

	dim3 grid(blockX, blockY, blockZ);
	dim3 block(threadx, 4);
	int shared = (6) * sizeof(float);

	kernel_winograd_Atgd_3x1 << < grid, block, shared, stream >> > (output, data, bias, P, Q, TileH, TileW, scale, activation);
}


void winograd_1d_1x3_fucn(float* output_Btd, float* data_Btd, float* output_Atgd, float* data_Atgd, float* bias, int N, int C, int H, int W, int left, int top, int TileW, int TileH, int K, int P, int Q, float scale, int activation, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, const float* dev_A, int WidthA, int HeightA, long long strideA, const float* dev_B, int WidthB, int HeightB, long long strideB,
	float *dev_C, long long strideC, int batchCount) {

	// 2. data transform
	winograd_Btd_1x3(output_Btd, data_Btd, N, C, H, W, left, top, TileW, TileH, stream);
	// 3. Point wise matrix multiplication
	SgemmStridedBatched(Blas, AOp, BOp, dev_A, WidthA, HeightA, strideA, dev_B, WidthB, HeightB, strideB, dev_C, strideC, batchCount);
	// 4. Result transform
	winograd_Atgd_1x3(output_Atgd, data_Atgd, bias, N, K, P, Q, TileW, TileH, scale, activation, stream);
};

void winograd_1d_3x1_fucn(float* output_Btd, float* data_Btd, float* output_Atgd, float* data_Atgd, float* bias, int N, int C, int H, int W, int left, int top, int TileW, int TileH, int K, int P, int Q, float scale, int activation, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, const float* dev_A, int WidthA, int HeightA, long long strideA, const float* dev_B, int WidthB, int HeightB, long long strideB,
	float *dev_C, long long strideC, int batchCount) {

	// 2. data transform
	winograd_Btd_3x1(output_Btd, data_Btd, N, C, H, W, left, top, TileW, TileH, stream);
	// 3. Point wise matrix multiplication
	SgemmStridedBatched(Blas, AOp, BOp, dev_A, WidthA, HeightA, strideA, dev_B, WidthB, HeightB, strideB, dev_C, strideC, batchCount);
	// 4. Result transform
	winograd_Atgd_3x1(output_Atgd, data_Atgd, bias, N, K, P, Q, TileW, TileH, scale, activation, stream);
};
