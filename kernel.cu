
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#define cyclelength 100
#define checksize 10000000.0

using namespace std;

int CPU(int N)
{
	float* vectA = new float[N];
	float* vectB = new float[N];
	float* vectC = new float[N];
	float* vectY = new float[N];

	float checksumm = 0;
	//��������� ������� �������
	for (int i = 0; i < N; i++) {
		vectA[i] = static_cast<float>(i * 1.1);
		vectB[i] = static_cast<float>(i * 2.7) + 1;
		vectC[i] = static_cast<float>(i * 3.14);
	}

	auto startc = std::chrono::high_resolution_clock::now();

	//����������
	for (int j = 0; j < cyclelength; j++)
		for (int i = 0; i < N; i++)
			vectY[i] = ((vectA[i] * vectB[i]) + ((vectA[i] * vectA[i] * vectA[i]) * vectC[i] / vectB[i]));

	auto finishc = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finishc - startc).count();
	cout << "Chrono: N = " << N << "; CPU Time = " << duration / 1000.0 << " ms\n";

	for (int i = 0; i < N; i++) {
		checksumm += vectY[i] / checksize;
		//cout << "Result[" << i << "]:=" << vectY[i] << endl;
	}

	printf("checksumm = %f\n\n\n", checksumm);

	delete[] vectA;
	delete[] vectB;
	delete[] vectC;
	delete[] vectY;
	return 0;
}

__global__ void addKernel(float* vectA, float* vectB, float* vectC, float* vectY, int S)
{
	int k;
	//����� ����� � ������ * ������ ����� + ����� ������ � ������� �����
	int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
	int m = 0;
	for (int j = 0; j < cyclelength; j++)
		for (k = 0; k < S; k++)
		{
			m = idx_thread * S + k;
			vectY[m] = ((vectA[m] * vectB[m]) + ((vectA[m] * vectA[m] * vectA[m]) * vectC[m] / vectB[m]));
		}

}

__global__ void fun_kernel(float* a, int s) {
	
}

int GPU(int N, int blocks, int blocksize)
{
	float* vectA = new float[N];
	float* vectB = new float[N];
	float* vectC = new float[N];
	float* vectY = new float[N];


	float* devA;
	float* devB; //��������� �� ������ ��� ���
	float* devC;
	float* devY; //��������� �� ������ ��� ���

	float elapsedTime; //��� �������� ������� ���������� CUDA

	//float checksumm = 0;

	cudaEvent_t start, stop; //�������������� �������
	cudaEventCreate(&start); //������������� ������� start
	cudaEventCreate(&stop); //������������� ������� stop

	float m = 3;
	float checksumm = 0;
	//��������� ������� �������
	for (int i = 0; i < N; i++) {
		vectA[i] = static_cast<float>(i * 1.1);
		vectB[i] = static_cast<float>(i * 2.7) + 1;
		vectC[i] = static_cast<float>(i * 3.14);
	}
	////��������� ������ �� ��� ��� ��������
	cudaMalloc((void**)&devA, N * sizeof(float));
	cudaMalloc((void**)&devB, N * sizeof(float));
	cudaMalloc((void**)&devC, N * sizeof(float));
	cudaMalloc((void**)&devY, N * sizeof(float));

	//blocks = 16; //���������� ������ �������
	//blocksize = 512; //������� �� ���� ����

	int steps = static_cast<int>(N / (blocks * blocksize));  //���������� ��������� ��������, ������� ������������ ������ �����

	//��������� ����� ������ ����������
	cudaEventRecord(start, 0);
	auto startc = std::chrono::high_resolution_clock::now();

	//�������� ������ 1 �� ����� devA, �������� �����-��, � � ������ ��� �������� � ����� -> ���
	cudaMemcpy(devA, vectA, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, vectB, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devC, vectC, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, vectY, N * sizeof(float), cudaMemcpyHostToDevice);

	//��������� �� ���
	addKernel <<<blocks, blocksize>>> (devA, devB, devC, devY, steps);

	//�������� ���� ����� ��������� ��������
	cudaMemcpy(vectY, devY, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(devY);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); //������������� host � device �� ������� stop

	cudaEventElapsedTime(&elapsedTime, start, stop); //����� = stop - start
	auto finishc = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finishc - startc).count();
	cout << "Chrono: N = " << N << ";"<<" Blocks = "<<blocks<<"; Blocksize = "<<blocksize<<"; GPU Time = " << duration / 1000.0 << " ms\n";
	//cout << duration / 1000.0 << endl;
	printf("CUDA: GPU Time = %f\n", elapsedTime);

	for (int i = 0; i < N; i++) {
		checksumm += vectY[i] / checksize;
		//cout << "Result[" << i << "]:=" << vectY[i] << endl;
	}

	//printf("checksumm = %f\n\n\n", checksumm);

	//printf("vect1 = %5.f\n", vect1[1023]);

	//tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

	delete[] vectA;
	delete[] vectB;
	delete[] vectC;
	delete[] vectY;

	return 0;
}

int main(){
	int count[3] = { pow(2,14),pow(2,16) };
	int blocks[5] = {1,2,4,8,16};
	int blocksize[6] = {1,4,32,64,256,512};

	for (int i = 0; i < 2; i++)
		{
			CPU(count[i]);
		}
	system("Pause");
	for (int i = 0; i < 2; i++) 
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 6; k++) 
				GPU(count[i], blocks[j], blocksize[k]);

	system("Pause");

	return 0;
}