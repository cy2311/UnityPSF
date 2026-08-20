// Linux host API for the official GlobLoc v1.0 CUDA kernels.
// The fitting kernel and parameterization remain from the upstream source.

#include <cuda_runtime.h>

#include "definitions.h"

extern void kernel_splineMLEFit_z_EMCCD_multi_wrapper(
    dim3 dimGrid,
    dim3 dimBlock,
    const float* d_data,
    const float* d_coeff,
    const float* dTAll,
    int spline_xsize,
    int spline_ysize,
    int spline_zsize,
    int sz,
    int iterations,
    int NV,
    int noChannels,
    float* d_Parameters,
    float* d_CRLBs,
    float* d_LogLikelihood,
    float* d_initZall,
    int Nfits,
    const int* d_shared);

namespace {

int check_cuda(cudaError_t status) {
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

}  // namespace

extern "C" int globloc_fit_multichannel_emccd_spline(
    const float* data,
    const int* shared,
    int iterations,
    const float* coeff,
    const float* dTAll,
    const float* initZ,
    int sz,
    int spline_xsize,
    int spline_ysize,
    int spline_zsize,
    int Nfits,
    int noChannels,
    float* parameters,
    float* crlbs,
    float* log_likelihood) {
    if (data == nullptr || shared == nullptr || coeff == nullptr || dTAll == nullptr || initZ == nullptr ||
        parameters == nullptr || crlbs == nullptr || log_likelihood == nullptr) {
        return -1;
    }
    if (sz <= 0 || sz > IMSZBIG || spline_xsize <= 0 || spline_ysize <= 0 || spline_zsize <= 0 ||
        Nfits <= 0 || noChannels <= 0 || noChannels > Max_No_Channel || iterations <= 0) {
        return -2;
    }

    const int shared_sum = shared[0] + shared[1] + shared[2] + shared[3] + shared[4];
    const int no_parameters = 5 * noChannels - shared_sum * (noChannels - 1);
    const size_t data_count = static_cast<size_t>(sz) * sz * Nfits * noChannels;
    const size_t coeff_count = static_cast<size_t>(spline_xsize) * spline_ysize * spline_zsize * 64 * noChannels;
    const size_t transform_count = static_cast<size_t>(Nfits) * noChannels * 2 * 5;
    const size_t shared_count = static_cast<size_t>(Nfits) * 5;

    float* d_data = nullptr;
    float* d_coeff = nullptr;
    float* d_transform = nullptr;
    float* d_init_z = nullptr;
    int* d_shared = nullptr;
    float* d_parameters = nullptr;
    float* d_crlbs = nullptr;
    float* d_log_likelihood = nullptr;
    int status = 0;

    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_data), data_count * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_coeff), coeff_count * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_transform), transform_count * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_init_z), static_cast<size_t>(Nfits) * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_shared), shared_count * sizeof(int)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_parameters), static_cast<size_t>(no_parameters + 1) * Nfits * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_crlbs), static_cast<size_t>(no_parameters) * Nfits * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_log_likelihood), static_cast<size_t>(Nfits) * sizeof(float)));
    if (status != 0) goto cleanup;

    status = check_cuda(cudaMemcpy(d_data, data, data_count * sizeof(float), cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(d_coeff, coeff, coeff_count * sizeof(float), cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(d_transform, dTAll, transform_count * sizeof(float), cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(d_init_z, initZ, static_cast<size_t>(Nfits) * sizeof(float), cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(d_shared, shared, shared_count * sizeof(int), cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;

    {
        const dim3 block(BSZ);
        const dim3 grid((Nfits + BSZ - 1) / BSZ);
        kernel_splineMLEFit_z_EMCCD_multi_wrapper(
            grid,
            block,
            d_data,
            d_coeff,
            d_transform,
            spline_xsize,
            spline_ysize,
            spline_zsize,
            sz,
            iterations,
            no_parameters,
            noChannels,
            d_parameters,
            d_crlbs,
            d_log_likelihood,
            d_init_z,
            Nfits,
            d_shared);
    }
    status = check_cuda(cudaGetLastError());
    if (status != 0) goto cleanup;
    status = check_cuda(cudaDeviceSynchronize());
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(parameters, d_parameters, static_cast<size_t>(no_parameters + 1) * Nfits * sizeof(float), cudaMemcpyDeviceToHost));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(crlbs, d_crlbs, static_cast<size_t>(no_parameters) * Nfits * sizeof(float), cudaMemcpyDeviceToHost));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMemcpy(log_likelihood, d_log_likelihood, static_cast<size_t>(Nfits) * sizeof(float), cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(d_data);
    cudaFree(d_coeff);
    cudaFree(d_transform);
    cudaFree(d_init_z);
    cudaFree(d_shared);
    cudaFree(d_parameters);
    cudaFree(d_crlbs);
    cudaFree(d_log_likelihood);
    return status;
}
