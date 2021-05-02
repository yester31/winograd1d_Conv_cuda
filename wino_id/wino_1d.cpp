#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h> // include path ===> %NVCUDASAMPLES_ROOT%/common/inc
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cudnn.h>
#include <iostream>
#include <iomanip>
#include <cublas_v2.h>

using namespace std;
using namespace chrono;


void winograd_1d_1x3_fucn(float* output_Btd, float* data_Btd, float* output_Atgd, float* data_Atgd, float* bias, int N, int C, int H, int W, int left, int top, int TileW, int TileH, int K, int P, int Q, float scale, int activation, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, const float* dev_A, int WidthA, int HeightA, long long strideA, const float* dev_B, int WidthB, int HeightB, long long strideB,
	float *dev_C, long long strideC, int batchCount);


void winograd_1d_3x1_fucn(float* output_Btd, float* data_Btd, float* output_Atgd, float* data_Atgd, float* bias, int N, int C, int H, int W, int left, int top, int TileW, int TileH, int K, int P, int Q, float scale, int activation, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, const float* dev_A, int WidthA, int HeightA, long long strideA, const float* dev_B, int WidthB, int HeightB, long long strideB,
	float *dev_C, long long strideC, int batchCount);

void deviceQuery()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("Device %d : \"%s\"\n", dev, deviceProp.name);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("  Multiprocessors (MP) :                         %d\n", deviceProp.multiProcessorCount);
		printf("  CUDA Cores/MP :                                %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
		printf("  CUDA Cores :                                   %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  Total amount of shared memory per block:       %llu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	}
	printf("\n");
}

// input : [K,C,3,3] cuDNN-default format
// output : [6,1,C,K] TensorFlow-like format

void  wino_transform_weight_f_1x3_cpu(vector<float> &output, vector<float> input, int K, int C)
{
	static const double G[6][3] = { {1 / 4.,0,0}, {-1 / 6.,-1 / 6.,-1 / 6.}, {-1 / 6.,1 / 6.,-1 / 6.}, {1 / 24., 1 / 12., 1 / 6.}, {1 / 24., -1 / 12., 1 / 6.}, {0,0,1.} };

	for (int oc = 0; oc < K; oc++) { // out_channel
		for (int ic = 0; ic < C; ic++) { // in_channel
			int input_idx = oc * C * 3 + ic * 3;				// [K, C, 3]	
			//int output_idx = oc * C * 6 + ic * 6;				// [K, C, 6]			
			double Gg[6][3] = { 0 };
			for (int m = 0; m < 6; m++) {
				double sum = 0.;
				for (int k = 0; k < 3; k++) {
					sum += G[m][k] * input[input_idx + k];
				}
				int o_idx = m * C * K  + ic * K + oc;			// [6, C, K]
				output[o_idx] = (float)sum;
				//output[output_idx + m] = (float)sum;
			}
		}
	}
}

// 1씩 증가하는 등차수열
void inititalizedData(vector<float>& container)
{
	int count = 0;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
		count++;
	}
}

// 모든 값을 1로 초기화
void inititalizedDataOne(vector<float>& container)
{
	int count = 1;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
	}
}

void valueCheck(vector<float> valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				cout << "  ";
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;
					//cout.setf(ios::fixed);
					//cout.precision(6);
					cout << setw(8) << valueCheckInput[g_idx] << " ";

				}cout << endl;
			}cout << endl; cout << endl;
		}cout << endl; cout << endl;
	}cout << endl;
}

typedef struct {
	int N;
	int C, H, W; // data [N,C,H,W]
	int K, P, Q; // output [N,K,P,Q]
	int KH, KW; // weight height, width
	int left, right, top, bottom; // pad left, right, top, bottom
} Config;

const int ITER = 200;

int main() {
	int F = 6;
	deviceQuery();
	cudaSetDevice(0);
	cudaStream_t stream;
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;
	int status;
	status = (int)cudaStreamCreate(&stream);
	status |= (int)cudnnCreate(&cudnnHandle);
	status |= (int)cublasCreate(&cublasHandle);
	status |= (int)cudnnSetStream(cudnnHandle, stream);
	status |= (int)cublasSetStream(cublasHandle, stream);
	printf("status=%d CUDA Setup Done!\n\n", status);

	//Config c = { 3,1024,16,16,  512,16,16,  1,3,  1,1,0,0 };
	//Config c = { 1, 192, 8, 8, 224, 8, 8,  1, 3,  1, 1, 0, 0 };

	//Config c = { 1, 192, 8, 8, 224, 8, 8,  3, 1,  0, 0, 1, 1 };
	//Config c = { 3,16,112,112,  16,112,112,  1,3,  1,1,0,0 };
	//Config c = { 3,16,224,224,  16,224,224,  1,3,  1,1,0,0 };
	//Config c = { 3,256,256,256,  256,256,256,  1,3,  1,1,0,0 };
	//Config c = { 1,1,4,4,  1,4,4,  1,3,  1,1,0,0 };
	  Config c = { 1,1,4,4,  1,4,4,  3,1,  0,0,1,1 };


	vector<float> data(c.N*c.C*c.H*c.W);		// input data [N,C,H,W]
	vector<float> bias(c.K);
	vector<float> tWeight(F*c.C*c.K);		 // transformed weight [6,C,K]
	vector<float> weight(c.K*c.C*c.KH*c.KW); // weight [K,C,3]

	inititalizedData(weight);
	//valueCheck(weight, c.K, c.C, c.KH, c.KW);

	//////////////////////////////////////////////////////////////
	// 1. filter transform
	//////////////////////////////////////////////////////////////
	wino_transform_weight_f_1x3_cpu(tWeight, weight, c.K, c.C);

	inititalizedData(data);		//  1씩 증가하는 등차수열
	//cout << "최초 생성한 data 값 생성" << endl;
	//valueCheck(data, c.N, c.C, c.H, c.W);
	inititalizedDataOne(bias);		//  1

	float* d_data; // device input data
	status |= (int)cudaMalloc(&d_data, data.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	float* d_bias; // device bias
	status |= (int)cudaMalloc(&d_bias, bias.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);

	
	int pH = c.H + c.top + c.bottom;
	int pW = c.W + c.left + c.right;
	int block = F - 2; // 4
	int TileH;
	int TileW;

	if (c.KH == 1 && c.KW == 3) {
		 TileH = pH;
		 TileW = pW / block + int(pW % block > 2);
	}
	else if(c.KH == 3 && c.KW == 1) {
		 TileH = pH / block + int(pW % block > 2);
		 TileW = pW;	
	}
	else { //아마 33 일때 
		TileH = pH / block + int(pW % block > 2);
		TileW = pW / block + int(pW % block > 2);	
	}

	//////////////////////////////////////////////////////////////주의

	float* d_tWeight;
	status |= (int)cudaMalloc(&d_tWeight, tWeight.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_tWeight, tWeight.data(), tWeight.size() * sizeof(float), cudaMemcpyHostToDevice);

	vector<float> output_Btd(c.N * c.C * TileH * TileW  * F); // Btd output
	float* d_output_Btd; // device output data
	status |= (int)cudaMalloc(&d_output_Btd, output_Btd.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_output_Btd, output_Btd.data(), output_Btd.size() * sizeof(float), cudaMemcpyHostToDevice);

	vector<float> output_pointwise(c.N * c.K * TileH * TileW * 6); // output_pointwise
	float* d_output_pointwise; // device output data
	status |= (int)cudaMalloc(&d_output_pointwise, output_pointwise.size() * sizeof(float));

	vector<float> finallOut(c.N * c.K * c.P * c.Q); // 최종결과 가져오기
	float* d_finallOut; // device output data
	status |= (int)cudaMalloc(&d_finallOut, finallOut.size() * sizeof(float));

	float scale = 1.0f;  //scale
	int actiavter = 1; // relu=1, linear=0

	//////////////////////////////////////////////////////////////
	if (c.KH == 1 && c.KW == 3) {
		uint64_t total_time = 0;
		for (int iIdx = 0; iIdx < ITER; iIdx++) {
			uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

			winograd_1d_1x3_fucn(d_output_Btd, d_data, d_finallOut, d_output_pointwise, d_bias, c.N, c.C, c.H, c.W, c.left, c.top, TileW, TileH, c.K, c.P, c.Q, scale, actiavter, stream,
				cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, d_output_Btd, c.C, c.N * TileH*TileW, c.N * c.C*TileH*TileW, d_tWeight, c.K, c.C, c.K * c.C, d_output_pointwise, c.N * c.K*TileH*TileW, 6);

			status |= (int)cudaStreamSynchronize(stream);
			total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
		}

		status |= (int)cudaMemcpy(finallOut.data(), d_finallOut, finallOut.size() * sizeof(float), cudaMemcpyDeviceToHost);

		double checksum = 0;
		for (auto d : finallOut) checksum += fabs((double)d);
		printf(" winograd1x3  status=%d avg_dur_time=%6.3f[msec] checksum=%.6f\n", status, total_time / 1000.f / ITER, checksum);


	}else {		
		uint64_t total_time = 0;
		for (int iIdx = 0; iIdx < ITER; iIdx++) {
			uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

			winograd_1d_3x1_fucn(d_output_Btd, d_data, d_finallOut, d_output_pointwise, d_bias, c.N, c.C, c.H, c.W, c.left, c.top, TileW, TileH, c.K, c.P, c.Q, scale, actiavter, stream,
				cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, d_output_Btd, c.C, c.N * TileH*TileW, c.N * c.C*TileH*TileW, d_tWeight, c.K, c.C, c.K * c.C, d_output_pointwise, c.N * c.K*TileH*TileW, 6);

			status |= (int)cudaStreamSynchronize(stream);
			total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
		}

		status |= (int)cudaMemcpy(finallOut.data(), d_finallOut, finallOut.size() * sizeof(float), cudaMemcpyDeviceToHost);

		double checksum = 0;
		for (auto d : finallOut) checksum += fabs((double)d);
		printf(" winograd3x1  status=%d avg_dur_time=%6.3f[msec] checksum=%.6f\n", status, total_time / 1000.f / ITER, checksum);

	}

	cout << "Final output" << endl;
	valueCheck(finallOut, c.N, c.K, c.P, c.Q);	// 출력값 확인

	status |= (int)cudaFree(d_output_pointwise);
	status |= (int)cudaFree(d_finallOut);
	status |= (int)cudaFree(d_tWeight); 
	status |= (int)cudaFree(d_output_Btd); 

	//////////////////////////////////
	///////////// cuDNN //////////////
	//////////////////////////////////

	float* d_weight; // device weight
	status |= (int)cudaMalloc(&d_weight, weight.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);

	//cout << "after Device" << endl;
	vector<float> weight_test(weight.size());
	status |= (int)cudaMemcpy(weight_test.data(), d_weight, weight.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//valueCheck(weight_test, c.K, c.C, c.KH, c.KW);	// weight 확인
	vector<float> data_test(data.size());
	status |= (int)cudaMemcpy(data_test.data(), d_data, data_test.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//valueCheck(data_test, c.N, c.C, c.H, c.W );	// 출력값 확인

	vector<float> output_cudnn(c.N*c.K*c.P*c.Q); // cudnn_output

	float* d_output_cudnn; // device output data
	status |= (int)cudaMalloc(&d_output_cudnn, output_cudnn.size() * sizeof(float));
	status |= (int)cudaMemset(d_output_cudnn, 0, output_cudnn.size() * sizeof(float)); // initialize ZERO

	cudnnTensorDescriptor_t xdesc;
	status |= (int)cudnnCreateTensorDescriptor(&xdesc);
	status |= (int)cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, c.N, c.C, c.H, c.W);
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status |= (int)cudnnCreateFilterDescriptor(&wdesc);
	status |= (int)cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, c.K, c.C, c.KH, c.KW);
	cudnnConvolutionDescriptor_t conv_desc;
	status |= (int)cudnnCreateConvolutionDescriptor(&conv_desc);
	status |= (int)cudnnSetConvolution2dDescriptor(conv_desc, c.top, c.left, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
	cudnnTensorDescriptor_t ydesc;
	status |= (int)cudnnCreateTensorDescriptor(&ydesc);
	status |= (int)cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, c.N, c.K, c.P, c.Q);
	cudnnTensorDescriptor_t bias_desc;
	status |= (int)cudnnCreateTensorDescriptor(&bias_desc);
	status |= (int)cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c.K, 1, 1);
	cudnnActivationDescriptor_t act_desc;
	status |= (int)cudnnCreateActivationDescriptor(&act_desc);
	status |= (int)cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);

	cudnnConvolutionFwdAlgo_t cudnn_algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
	float one = 1.f, zero = 0.f;
	status |= (int)cudnnGetConvolutionForwardAlgorithm(cudnnHandle, xdesc, wdesc, conv_desc, ydesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &cudnn_algo);

	for (int al = 0; al < CUDNN_CONVOLUTION_FWD_ALGO_COUNT; al++) {
		cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)al;
		if (algo == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT || algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT || algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) continue;
		if (algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) continue; // only ACTIVATION==LINEAR, not RELU

		size_t workspace_size;
		status |= (int)cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xdesc, wdesc, conv_desc, ydesc, algo, &workspace_size);
		float* d_workspace = nullptr;
		status |= (int)cudaMalloc(&d_workspace, workspace_size);


		uint64_t total_time = 0;
		for (int idx = 0; idx < ITER; idx++) {
			uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
			status |= (int)cudnnConvolutionBiasActivationForward(
				cudnnHandle,
				&one, xdesc, d_data,
				wdesc, d_weight,
				conv_desc, algo,
				d_workspace, workspace_size,
				&zero, ydesc, d_output_cudnn,
				bias_desc, d_bias,
				act_desc,
				ydesc, d_output_cudnn);
			status |= (int)cudaStreamSynchronize(stream);
			total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
		}
		status |= (int)cudaMemcpy(output_cudnn.data(), d_output_cudnn, output_cudnn.size() * sizeof(float), cudaMemcpyDeviceToHost);

		//vector<float> output_test(c.K*c.C*c.KH*c.KW);
		//status |= (int)cudaMemcpy(output_test.data(), d_weight, weight.size() * sizeof(float), cudaMemcpyDeviceToHost);

		double checksum = 0;
		for (auto d : output_cudnn) checksum += fabs((double)d);
		printf("   cuDNN(%d/%d) status=%d avg_dur_time=%6.3f[msec] checksum=%.6f\n", al, (int)cudnn_algo, status, total_time / 1000.f / ITER, checksum);

		if (d_workspace)
			status |= (int)cudaFree(d_workspace);

	}
	printf("\n");

	//cout << "Cudnn Winograd output" << endl;
	//valueCheck(output_cudnn, c.N, c.K, c.P, c.Q);	// 출력값 확인

	status |= (int)cudaFree(d_weight); 
	status |= (int)cudaFree(d_output_cudnn); 
	status |= (int)cudaFree(d_data);
	status |= (int)cudaFree(d_bias);

	return 0;
}