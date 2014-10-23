// -*- C++ -*-
#include <cstdio>

#include <cuda_runtime.h>
// These come from the cublas matrix multiplication example
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#include <cublas_v2.h>
#pragma GCC diagnostic pop


#include "MatrixMultiplication_cuda.cuh"

__global__
void
cudaDoNaiveMatrixMultiplication_kernel(const unsigned int matrixSize,
                                                   const double * leftMatrix,
                                                   const double * rightMatrix,
                                                   double * resultMatrix) {
  // TODO: something!
}

void
cudaDoMatrixMultiplication(const unsigned int maxNumberOfBlocks,
                           const unsigned int numberOfThreadsPerBlock,
                           const unsigned int matrixSize) {

  // TODO: something!
}


void
multiplyMatricesUsingCublas(const unsigned int matrixSize,
                            const double * leftMatrix,
                            const double * rightMatrix,
                            double * resultMatrix) {

  const unsigned int numberOfEntries = matrixSize * matrixSize;

  // allocate device memory
  double * dev_leftMatrix;
  double * dev_rightMatrix;
  double * dev_resultMatrix;
  (cudaMalloc((void **) &dev_leftMatrix, numberOfEntries * sizeof(double)));
  (cudaMalloc((void **) &dev_rightMatrix, numberOfEntries * sizeof(double)));
  (cudaMalloc((void **) &dev_resultMatrix, numberOfEntries * sizeof(double)));
  // copy matrices to the device
  (cudaMemcpy(dev_leftMatrix, leftMatrix, numberOfEntries * sizeof(double), cudaMemcpyHostToDevice));
  (cudaMemcpy(dev_rightMatrix, rightMatrix, numberOfEntries * sizeof(double), cudaMemcpyHostToDevice));

  const double alpha = 1.0f;
  const double beta  = 0.0f;
  cublasHandle_t handle;

  (cublasCreate(&handle));

  // perform the multiply
  (cublasDgemm(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N, // don't transpose
                              matrixSize, matrixSize, matrixSize, // sizes
                              &alpha, // no scalar premultiply
                              dev_rightMatrix, matrixSize, // left matrix
                              dev_leftMatrix, matrixSize, // right matrix
                              &beta, // don't premultiply result by anything
                              dev_resultMatrix, matrixSize));

  // copy result from device to host
  (cudaMemcpy(resultMatrix, dev_resultMatrix,
                             numberOfEntries * sizeof(double),
                             cudaMemcpyDeviceToHost));

  // Destroy the handle
  (cublasDestroy(handle));

  // clean up memory
  (cudaFree(dev_leftMatrix));
  (cudaFree(dev_rightMatrix));
  (cudaFree(dev_resultMatrix));
}
