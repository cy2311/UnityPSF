#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

using cx = c10::complex<float>;

__device__ __forceinline__ cx cx_mul(const cx& a, const cx& b) {
  return cx(a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real());
}

__device__ __forceinline__ cx cx_add(const cx& a, const cx& b) {
  return cx(a.real() + b.real(), a.imag() + b.imag());
}

__device__ __forceinline__ cx cx_scale(const cx& a, float s) {
  return cx(a.real() * s, a.imag() * s);
}

__device__ __forceinline__ cx cx_exp_i(float phase) {
  float s, c;
  sincosf(phase, &s, &c);
  return cx(c, s);
}

__device__ __forceinline__ float cx_abs2(const cx& a) {
  return a.real() * a.real() + a.imag() * a.imag();
}

// Evaluate the pupil field once per emitter/pupil pixel.  The previous kernel
// repeated this phase calculation for every polarization and output column.
__global__ void pupil_field_kernel(
    const float* __restrict__ coeffs, const float* __restrict__ z_nm,
    const float* __restrict__ phase_basis, const float* __restrict__ amplitude,
    const cx* __restrict__ carrier, const float* __restrict__ kz_medium,
    const float* __restrict__ kz_immersion, cx* __restrict__ field,
    int n_emitters, int n_modes, int pupil,
    float objstage0_nm, float zemit0_nm) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_emitters * pupil * pupil;
  if (q >= total) return;
  int pidx = q % (pupil * pupil);
  int emitter = q / (pupil * pupil);

  float phase = 0.0f;
  for (int mode = 0; mode < n_modes; ++mode) {
    phase += coeffs[emitter * n_modes + mode] * phase_basis[mode * pupil * pupil + pidx];
  }
  float ztotal = z_nm[emitter] + zemit0_nm;
  if (ztotal >= 0.0f) {
    phase += ztotal * kz_medium[pidx] + objstage0_nm * kz_immersion[pidx];
  } else {
    phase += (objstage0_nm + ztotal) * kz_immersion[pidx];
  }
  field[q] = cx_scale(cx_mul(cx_exp_i(phase), carrier[pidx]), amplitude[pidx]);
}

// First separable transform: pupil columns (pupil) -> camera columns (psf).
__global__ void first_transform_kernel(
    const cx* __restrict__ field, const cx* __restrict__ polarization,
    const cx* __restrict__ transform_y, cx* __restrict__ intermediate,
    int n_emitters, int pupil, int psf) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_emitters * 6 * psf * pupil;
  if (q >= total) return;
  int out_col = q % psf;
  int t = q / psf;
  int row = t % pupil;
  t /= pupil;
  int component = t % 6;
  int emitter = t / 6;

  cx accum(0.0f, 0.0f);
  for (int col = 0; col < pupil; ++col) {
    int pidx = row * pupil + col;
    cx value = field[emitter * pupil * pupil + pidx];
    value = cx_mul(value, polarization[component * pupil * pupil + pidx]);
    accum = cx_add(accum, cx_mul(value, transform_y[out_col * pupil + col]));
  }
  intermediate[((emitter * 6 + component) * pupil + row) * psf + out_col] = accum;
}

// Second separable transform and vector-intensity reduction.
__global__ void second_transform_kernel(
    const cx* __restrict__ intermediate, const cx* __restrict__ transform_x,
    float* __restrict__ output, int n_emitters, int pupil, int psf,
    const float* __restrict__ norm_intensity) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_emitters * psf * psf;
  if (q >= total) return;
  int x = q % psf;
  int t = q / psf;
  int y = t % psf;
  int emitter = t / psf;
  float intensity = 0.0f;
  for (int component = 0; component < 6; ++component) {
    cx accum(0.0f, 0.0f);
    for (int col = 0; col < pupil; ++col) {
      cx value = intermediate[((emitter * 6 + component) * pupil + col) * psf + x];
      accum = cx_add(accum, cx_mul(transform_x[y * pupil + col], value));
    }
    intensity += cx_abs2(accum);
  }
  output[q] = fmaxf(0.0f, intensity / (3.0f * fmaxf(*norm_intensity, 1.0e-20f)));
}

__global__ void normalize_kernel(float* __restrict__ output, int n_emitters, int psf) {
  int emitter = blockIdx.x;
  if (emitter >= n_emitters) return;
  __shared__ float sum;
  if (threadIdx.x == 0) {
    sum = 0.0f;
    for (int i = 0; i < psf * psf; ++i) sum += output[emitter * psf * psf + i];
    sum = fmaxf(sum, 1.0e-20f);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < psf * psf; i += blockDim.x) {
    output[emitter * psf * psf + i] /= sum;
  }
}

__global__ void reduce_frames_kernel(
    const float* __restrict__ patches,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ center_x,
    const int64_t* __restrict__ center_y,
    float* __restrict__ output,
    int n,
    int frame_count,
    int height,
    int width,
    int psf) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * psf * psf;
  if (q >= total) return;
  int px = q % psf;
  int t = q / psf;
  int py = t % psf;
  int emitter = t / psf;
  int x = static_cast<int>(center_x[emitter]) + px - psf / 2;
  int y = static_cast<int>(center_y[emitter]) + py - psf / 2;
  int frame = static_cast<int>(frame_indices[emitter]);
  if (frame >= 0 && frame < frame_count && x >= 0 && x < width && y >= 0 && y < height) {
    atomicAdd(output + (frame * height + y) * width + x, patches[q]);
  }
}

}  // namespace

torch::Tensor render_psf_bank_cuda(
    torch::Tensor coeffs_rad, torch::Tensor z_nm, torch::Tensor phase_basis,
    torch::Tensor amplitude, torch::Tensor polarization, torch::Tensor carrier,
    torch::Tensor kz_medium, torch::Tensor kz_immersion,
    torch::Tensor transform_y, torch::Tensor transform_x,
    torch::Tensor norm_intensity, double objstage0_nm, double zemit0_nm) {
  TORCH_CHECK(coeffs_rad.is_cuda(), "coeffs_rad must be CUDA");
  TORCH_CHECK(coeffs_rad.scalar_type() == torch::kFloat32, "coeffs_rad must be float32");
  TORCH_CHECK(z_nm.scalar_type() == torch::kFloat32, "z_nm must be float32");
  TORCH_CHECK(phase_basis.scalar_type() == torch::kFloat32, "phase_basis must be float32");
  TORCH_CHECK(amplitude.scalar_type() == torch::kFloat32, "amplitude must be float32");
  TORCH_CHECK(kz_medium.scalar_type() == torch::kFloat32, "kz_medium must be float32");
  TORCH_CHECK(kz_immersion.scalar_type() == torch::kFloat32, "kz_immersion must be float32");
  TORCH_CHECK(polarization.scalar_type() == torch::kComplexFloat, "polarization must be complex64");
  TORCH_CHECK(carrier.scalar_type() == torch::kComplexFloat, "carrier must be complex64");
  TORCH_CHECK(transform_y.scalar_type() == torch::kComplexFloat, "transform_y must be complex64");
  TORCH_CHECK(transform_x.scalar_type() == torch::kComplexFloat, "transform_x must be complex64");
  TORCH_CHECK(coeffs_rad.dim() == 2, "coeffs_rad must have shape [N,K]");
  TORCH_CHECK(z_nm.dim() == 1 && z_nm.size(0) == coeffs_rad.size(0), "z_nm shape mismatch");
  const int n = static_cast<int>(coeffs_rad.size(0));
  const int k = static_cast<int>(coeffs_rad.size(1));
  const int pupil = static_cast<int>(amplitude.size(0));
  const int psf = static_cast<int>(transform_y.size(0));
  TORCH_CHECK(amplitude.dim() == 2 && amplitude.size(0) == pupil && amplitude.size(1) == pupil, "amplitude must be [P,P]");
  TORCH_CHECK(phase_basis.dim() == 3 && phase_basis.size(0) == k && phase_basis.size(1) == pupil && phase_basis.size(2) == pupil, "phase_basis shape mismatch");
  TORCH_CHECK(polarization.dim() == 3 && polarization.size(0) == 6 && polarization.size(1) == pupil && polarization.size(2) == pupil, "polarization must be [6,P,P]");
  TORCH_CHECK(carrier.dim() == 2 && carrier.size(0) == pupil && carrier.size(1) == pupil, "carrier shape mismatch");
  TORCH_CHECK(transform_y.dim() == 2 && transform_y.size(0) == psf && transform_y.size(1) == pupil, "transform_y shape mismatch");
  TORCH_CHECK(transform_x.dim() == 2 && transform_x.size(0) == psf && transform_x.size(1) == pupil, "transform_x shape mismatch");
  TORCH_CHECK(kz_medium.dim() == 2 && kz_medium.size(0) == pupil && kz_medium.size(1) == pupil, "kz_medium shape mismatch");
  TORCH_CHECK(kz_immersion.dim() == 2 && kz_immersion.size(0) == pupil && kz_immersion.size(1) == pupil, "kz_immersion shape mismatch");
  TORCH_CHECK(norm_intensity.numel() == 1, "norm_intensity must be scalar");
  TORCH_CHECK(coeffs_rad.is_contiguous() && z_nm.is_contiguous() && phase_basis.is_contiguous() &&
                  amplitude.is_contiguous() && polarization.is_contiguous() && carrier.is_contiguous() &&
                  kz_medium.is_contiguous() && kz_immersion.is_contiguous() && transform_y.is_contiguous() &&
                  transform_x.is_contiguous(),
              "all renderer inputs must be contiguous");

  c10::cuda::CUDAGuard device_guard(coeffs_rad.device());
  auto field = torch::empty({n, pupil, pupil}, coeffs_rad.options().dtype(torch::kComplexFloat));
  auto intermediate = torch::empty({n, 6, pupil, psf}, coeffs_rad.options().dtype(torch::kComplexFloat));
  auto output = torch::empty({n, psf, psf}, coeffs_rad.options().dtype(torch::kFloat32));
  if (n == 0) return output;
  const int threads = 256;
  const int field_total = n * pupil * pupil;
  const int first_total = n * 6 * psf * pupil;
  const int second_total = n * psf * psf;
  auto stream = at::cuda::getCurrentCUDAStream(coeffs_rad.device().index());
  pupil_field_kernel<<<(field_total + threads - 1) / threads, threads, 0, stream>>>(
      coeffs_rad.data_ptr<float>(), z_nm.data_ptr<float>(), phase_basis.data_ptr<float>(),
      amplitude.data_ptr<float>(), carrier.data_ptr<c10::complex<float>>(),
      kz_medium.data_ptr<float>(), kz_immersion.data_ptr<float>(),
      field.data_ptr<c10::complex<float>>(), n, k, pupil,
      static_cast<float>(objstage0_nm), static_cast<float>(zemit0_nm));
  first_transform_kernel<<<(first_total + threads - 1) / threads, threads, 0, stream>>>(
      field.data_ptr<c10::complex<float>>(), polarization.data_ptr<c10::complex<float>>(),
      transform_y.data_ptr<c10::complex<float>>(), intermediate.data_ptr<c10::complex<float>>(),
      n, pupil, psf);
  second_transform_kernel<<<(second_total + threads - 1) / threads, threads, 0, stream>>>(
      intermediate.data_ptr<c10::complex<float>>(), transform_x.data_ptr<c10::complex<float>>(),
      output.data_ptr<float>(), n, pupil, psf, norm_intensity.data_ptr<float>());
  normalize_kernel<<<n, 128, 0, stream>>>(output.data_ptr<float>(), n, psf);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor reduce_frames_cuda(
    torch::Tensor patches,
    torch::Tensor frame_indices,
    torch::Tensor center_x,
    torch::Tensor center_y,
    int64_t frame_count,
    int64_t height,
    int64_t width,
    double background) {
  TORCH_CHECK(patches.is_cuda() && patches.scalar_type() == torch::kFloat32,
              "patches must be CUDA float32");
  TORCH_CHECK(frame_indices.is_cuda() && center_x.is_cuda() && center_y.is_cuda(),
              "frame metadata must be CUDA");
  TORCH_CHECK(frame_indices.scalar_type() == torch::kLong &&
                  center_x.scalar_type() == torch::kLong && center_y.scalar_type() == torch::kLong,
              "frame metadata must be int64");
  TORCH_CHECK(patches.dim() == 3, "patches must be [N,S,S]");
  const int n = static_cast<int>(patches.size(0));
  const int psf = static_cast<int>(patches.size(1));
  TORCH_CHECK(patches.size(2) == psf && frame_indices.numel() == n &&
                  center_x.numel() == n && center_y.numel() == n,
              "frame metadata shape mismatch");
  TORCH_CHECK(frame_count > 0 && height > 0 && width > 0,
              "frame dimensions must be positive");
  auto output = torch::full({frame_count, height, width}, static_cast<float>(background), patches.options());
  if (n == 0) return output;
  c10::cuda::CUDAGuard device_guard(patches.device());
  auto stream = at::cuda::getCurrentCUDAStream(patches.device().index());
  const int threads = 256;
  reduce_frames_kernel<<<(n * psf * psf + threads - 1) / threads, threads, 0, stream>>>(
      patches.data_ptr<float>(), frame_indices.data_ptr<int64_t>(), center_x.data_ptr<int64_t>(),
      center_y.data_ptr<int64_t>(), output.data_ptr<float>(), n, static_cast<int>(frame_count),
      static_cast<int>(height), static_cast<int>(width), psf);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
