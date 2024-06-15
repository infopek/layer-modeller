#include "kriging_gpu.cuh"
#include "kriging_utilities.cuh"

__global__ void calculateCovarianceMatrix(const DataPoint* observedData,int n,  double* covMatrix, TheoreticalParam param) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;
    if (i<n && j<n ) {
        double h = std::sqrt(std::pow(observedData[i].x - observedData[j].x, 2) + std::pow(observedData[i].y - observedData[j].y, 2));
        covMatrix[i*n+j] = param.nugget + param.sill * (1.0 - exp(-(h * h) / (pow(param.range,2))));
        
    } 
}
__device__ void runFunctionOnData(size_t n, double* K, const DataPoint* observedData, TheoreticalParam param, double targetX, double targetY) {
    size_t j = threadIdx.x;

}
__device__ void calculateVariance(const DataPoint* observedData, size_t row, size_t col, double* calculations, int dataSize, TheoreticalParam param) {
    for (size_t i = 0; i < dataSize; ++i) {
        double h = std::sqrt(std::pow(observedData[i].x -(int)row, 2) + std::pow(observedData[i].y - (int)col, 2));
        calculations[i]=param.nugget + param.sill * (1.0 - exp(-((h * h) / (param.range * param.range))));
    }
}
__device__ void solveForVari(double* covMxDecomposed, double* calculations, int dataSize) {
    for (int i = 0; i < dataSize; i++) {
        double s = 0;
        for (int j = 0; j < i; j++)
            s += covMxDecomposed[i * dataSize+ j] * calculations[j];
        calculations[i] = (calculations[i] - s) / covMxDecomposed[i*dataSize+i];
    }
}
__device__ void estimation(double* d_krigingOutput,const DataPoint* observedData, size_t row, size_t col, double* calculations, int dataSize,size_t pitch) {
    for (size_t i = 0; i < dataSize; ++i)
        ((double*)((char*)d_krigingOutput + row * pitch))[col]  += calculations[i] * observedData[i].value;
}

__global__ void estimateValue(double* d_krigingOutput, const size_t size,size_t outputPitch, const DataPoint* observedData, double* covMatrixLLT, const size_t dataSize, TheoreticalParam param, cudaPitchedPtr d_calculationsMx) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / size;
    int col = idx % size;
    size_t pitch = d_calculationsMx.pitch;
    size_t slicePitch = pitch * size;

    char* slice = (char*)d_calculationsMx.ptr + row * slicePitch;
    double* calculations = (double*)(slice + col * pitch);

    if (row < size && col < size) {
        calculateVariance(observedData,row,col,calculations,dataSize,param);
        solveForVari(covMatrixLLT,calculations,dataSize);
        estimation(d_krigingOutput,observedData,row,col,calculations,dataSize,outputPitch);
    }

}

void createInterpolationGPU(const std::vector<DataPoint>* observedData, TheoreticalParam param, double** krigingOutput, size_t maxX, size_t maxY, size_t size) {
    cudaError_t cudaError;
    size_t observedDataSize = observedData->size();

    DataPoint* d_observedData = nullptr;
    cudaMalloc((void**)&d_observedData, sizeof(DataPoint) * observedDataSize);
    checkCudaErrors(cudaMemcpy(d_observedData, observedData->data(), sizeof(DataPoint) * observedDataSize, cudaMemcpyHostToDevice));
    double* d_covMatrix = nullptr;
    size_t cpitch=0;
    checkCudaErrors(cudaMalloc((void**)&d_covMatrix, observedDataSize * sizeof(double)*observedDataSize));
    calculateCovarianceMatrix<<<(int)(pow(observedDataSize,2)+511)/512,512>>>(d_observedData,observedDataSize, d_covMatrix,param);
    
    double* cmx = new double[observedDataSize * observedDataSize];

    checkCudaErrors(cudaMemcpy(cmx, d_covMatrix, sizeof(double) * observedDataSize*observedDataSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < observedDataSize; i++) {
        for (int j = 0; j < observedDataSize; j++) {
            printf("%4.2f\t", cmx[i * observedDataSize + j]);
        }
        printf("\n");
    }

    printf("\n");
    Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(cmx,observedDataSize,observedDataSize).fullPivLu().matrixLU();
    // mat = mat.covMxDecomposed();
    checkCudaErrors(cudaMemcpy(d_covMatrix, mat.data(), sizeof(double) * observedDataSize*observedDataSize, cudaMemcpyHostToDevice));

     //cusolverDnHandle_t solver_handle;
     //cusolverDnCreate(&solver_handle);
     //int work_size = 0;
     //int *devInfo;           
     //cudaMalloc(&devInfo, sizeof(int));
     //cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, observedDataSize, d_covMatrix,observedDataSize, &work_size);
     //double *work;
     //checkCudaErrors(cudaMalloc(&work, work_size * sizeof(double)));
     //cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, observedDataSize, d_covMatrix, observedDataSize, work, work_size, devInfo);
     //cusolverDnDestroy(solver_handle);
     //cudaFree(devInfo);
     //cudaFree(work);
    
    checkCudaErrors(cudaMemcpy(cmx, d_covMatrix, sizeof(double) * observedDataSize*observedDataSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < observedDataSize; i++) {
        for (int j = 0; j < observedDataSize; j++) {
            printf("%4.2f\t", cmx[i * observedDataSize + j]);
        }
        printf("\n");
    }

    printf("\n");
    cudaExtent extent = make_cudaExtent(observedDataSize * sizeof(double), size, size);
    cudaPitchedPtr calculationsMx;
    checkCudaErrors(cudaMalloc3D(&calculationsMx, extent));

    size_t blockSize = 1024;
    size_t gridSize = ((size*size)+blockSize-1) / blockSize;

    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    double* d_krigingOutput = nullptr;
    size_t pitch=0;
    checkCudaErrors(cudaMallocPitch((void**)&d_krigingOutput,&pitch, size * sizeof(double),size));
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    estimateValue<<<gridDim, blockDim>>>(d_krigingOutput, size,pitch, d_observedData, d_covMatrix, observedDataSize, param,calculationsMx);
    checkLastCudaError();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy2D(*krigingOutput, sizeof(double)*size, d_krigingOutput,pitch, sizeof(double)*size,size, cudaMemcpyDeviceToHost));
    //cudaFree(d_weights3d.ptr);
    cudaFree(calculationsMx.ptr);
    cudaFree(d_krigingOutput);

    cudaFree(d_observedData);
    cudaFree(d_covMatrix);
    cudaFree(d_krigingOutput);
}
