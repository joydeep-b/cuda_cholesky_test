#include <stdio.h>

#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  const int num_cols = 4;
  google::InitGoogleLogging(argv[0]);
  float A_host[] = {
      4,  12, -16, 0,
     12,  37, -43, 0,
    -16, -43,  98, 0,
      0,   0,   0, 1
  };
  float b_host[] = { 1, 1, 1, 1};
  float x_expected[] = {
    113.75 / 3.0,
    -31.0 / 3.0,
    5.0 / 3.0,
    1.0
  };
  CHECK_EQ(sizeof(A_host), num_cols * num_cols * sizeof(float));
  CHECK_EQ(sizeof(b_host), num_cols * sizeof(float));
  float* A_device;
  float* b_device;

  // Initialize CuSolver.
  cusolverDnHandle_t cusolver_handle = nullptr;
  CHECK_EQ(cusolverDnCreate(&cusolver_handle), CUSOLVER_STATUS_SUCCESS);

  CHECK_EQ(cudaMalloc(
      &A_device, num_cols * num_cols * sizeof(float)), cudaSuccess);
  CHECK_EQ(cudaMemcpy(
      A_device, A_host, num_cols * num_cols * sizeof(float),
      cudaMemcpyHostToDevice), cudaSuccess);
  CHECK_EQ(cudaMalloc(&b_device, num_cols * sizeof(float)), cudaSuccess);
  CHECK_EQ(cudaMemcpy(
      b_device, b_host, num_cols * sizeof(float),
      cudaMemcpyHostToDevice), cudaSuccess);

  // Allocate buffer space.
  int device_workspace_size = 0;
  CHECK_EQ(cusolverDnSpotrf_bufferSize(cusolver_handle,
                                       CUBLAS_FILL_MODE_LOWER,
                                       num_cols,
                                       A_device,
                                       num_cols,
                                       &device_workspace_size),
      CUSOLVER_STATUS_SUCCESS);
  float* device_workspace = nullptr;
  CHECK_EQ(cudaMalloc(&device_workspace,
                      device_workspace_size * sizeof(float)),
      cudaSuccess);
  int* error_device = nullptr;
  CHECK_EQ(cudaMalloc(&error_device, sizeof(int)), cudaSuccess);

  // Perform Cholesky factorization.
  CHECK_EQ(cusolverDnSpotrf(cusolver_handle,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols,
                            A_device,
                            num_cols,
                            device_workspace,
                            device_workspace_size,
                            error_device),
      CUSOLVER_STATUS_SUCCESS);
  int error_host = 0;
  CHECK_EQ(cudaMemcpy(&error_host,
                      error_device,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
      cudaSuccess);
  CHECK_EQ(error_host, 0);

  // Solve the system.
  CHECK_EQ(cusolverDnSpotrs(cusolver_handle,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols,
                            1,
                            A_device,
                            num_cols,
                            b_device,
                            num_cols,
                            error_device),
      CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaMemcpy(&error_host,
                      error_device,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
      cudaSuccess);
  CHECK_EQ(error_host, 0);

  // Copy the result back to the host.
  float x_host[num_cols];
  CHECK_EQ(cudaMemcpy(x_host,
                      b_device,
                      num_cols * sizeof(float),
                      cudaMemcpyDeviceToHost),
      cudaSuccess);

  // Check the result.
  for (int i = 0; i < num_cols; ++i) {
    printf("%d: %e\n", i, fabs(x_host[i] - x_expected[i]));
  }
  return 0;
}