#include <iostream>
#include <chrono>
#include <cstdlib>
#define SIZE 69212
#define THREADS 96

using namespace std;

float sequentialSum = 0;
float parallelSum = 0;

int* getArray(int n) {
    int* res= new int[n];
    for(int i = 0; i < n; i++) {
        res[i] = rand();
    }
    return res;
}

__global__ void reductionMax(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < SIZE) {
        unsigned int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < SIZE) {
            int index = i + k * gridDim.x * blockDim.x;
            if (in[2 * index] < in[2 * index + step]) in[2 * index] = in[2 * index + step];
            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

__global__ void reductionMin(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < SIZE) {
        int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < SIZE) {
            int index = i + k * gridDim.x * blockDim.x;
            if(in[2*index] > in[2*index + step]) in[2*index] = in[2*index + step];

            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

__global__ void reductionSum(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < SIZE) {
        int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < SIZE) {
            int index = i + k * gridDim.x * blockDim.x;
            in[2*index] += in[2*index + step];
            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

__global__ void reductionMultiply(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < SIZE) {
        int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < SIZE) {
            int index = i + k * gridDim.x * blockDim.x;
            in[2*index] *= in[2*index + step];
            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

float procesSequentially() {
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    const auto startc = std::chrono::steady_clock::now();

    int* sum = getArray(SIZE);


    int* sum_cuda;
    cudaMalloc(&sum_cuda, SIZE * sizeof(int));
    cudaMemcpy( sum_cuda, sum, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    reductionSum<<<1, THREADS>>>(sum_cuda);

    int* prod = getArray(SIZE);
    int* prod_cuda;
    cudaMalloc(&prod_cuda, SIZE * sizeof(int));
    cudaMemcpy( prod_cuda, prod, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionMultiply<<<1, THREADS>>>(prod_cuda);

    cudaDeviceSynchronize();

    int* min = getArray(SIZE);
    int* min_cuda;
    cudaMalloc(&min_cuda, SIZE * sizeof(int));
    cudaMemcpy( min_cuda, min, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionMin<<<1, THREADS>>>(min_cuda);

    cudaDeviceSynchronize();

    int* max = getArray(SIZE);
    int* max_cuda;
    cudaMalloc(&max_cuda, SIZE * sizeof(int));
    cudaMemcpy( max_cuda, max, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionMax<<<1, THREADS>>>(max_cuda);

    cudaDeviceSynchronize();

    const auto end = std :: chrono :: steady_clock :: now () ;
    const std :: chrono :: duration<double> elapsed_seconds{end - startc};
    std::cout << elapsed_seconds.count()*1000 << ";" << endl;


    delete(sum);
    delete(prod);
    delete(min);
    delete(max);
    cudaFree(sum_cuda);
    cudaFree(prod_cuda);
    cudaFree(min_cuda);
    cudaFree(max_cuda);
    return elapsed_seconds.count() * 1000;
}

float procesParallel() {
    const auto startc = std::chrono::steady_clock::now();

    int* sum = getArray(SIZE);
    int* sum_cuda;
    cudaMalloc(&sum_cuda, SIZE * sizeof(int));
    cudaMemcpy( sum_cuda, sum, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionSum<<<1, THREADS>>>(sum_cuda);

    int* prod = getArray(SIZE);
    int* prod_cuda;
    cudaMalloc(&prod_cuda, SIZE * sizeof(int));
    cudaMemcpy( prod_cuda, prod, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionMultiply<<<1, THREADS>>>(prod_cuda);

    int* min = getArray(SIZE);
    int* min_cuda;
    cudaMalloc(&min_cuda, SIZE * sizeof(int));
    cudaMemcpy( min_cuda, min, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionMin<<<1, THREADS>>>(min_cuda);

    int* max = getArray(SIZE);
    int* max_cuda;
    cudaMalloc(&max_cuda, SIZE * sizeof(int));
    cudaMemcpy( max_cuda, max, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    reductionMax<<<1, THREADS>>>(max_cuda);

    cudaDeviceSynchronize();

    const auto end = std :: chrono :: steady_clock :: now () ;
    const std :: chrono :: duration<double> elapsed_seconds{end - startc};
    std::cout << elapsed_seconds.count()*1000 << ";" << endl;

    delete(sum);
    delete(prod);
    delete(min);
    delete(max);
    cudaFree(sum_cuda);
    cudaFree(prod_cuda);
    cudaFree(min_cuda);
    cudaFree(max_cuda);
    return elapsed_seconds.count() * 1000;
}

int main() {
    float arrayGenerationAverage = 0;
    for(int i = 0; i < 10; i++){
        const auto startc = std::chrono::steady_clock::now();
        getArray(SIZE);
        const auto end = std :: chrono :: steady_clock :: now () ;
        const std :: chrono :: duration<double> elapsed_seconds{end - startc};
        arrayGenerationAverage += elapsed_seconds.count()*1000;
    }
    std::cout << "Array calc time: " << arrayGenerationAverage / 10 << endl;

    cout << "Sequential" << endl;
    for(int i = 0; i < 10; i++)
        procesSequentially();
    for(int i = 0; i < 2000; i++)
        sequentialSum += procesSequentially();
    cout << "Parallel" << endl;
    for(int i = 0; i < 2000; i++)
        parallelSum += procesParallel();

    cout << sequentialSum/2000 << "; " << parallelSum/2000  << endl;
    return 0;
}
