#include <torch/extension.h>

#include <vector>

torch::Tensor render_psf_bank_cuda(
    torch::Tensor coeffs_rad,
    torch::Tensor z_nm,
    torch::Tensor phase_basis,
    torch::Tensor amplitude,
    torch::Tensor polarization,
    torch::Tensor carrier,
    torch::Tensor kz_medium,
    torch::Tensor kz_immersion,
    torch::Tensor transform_y,
    torch::Tensor transform_x,
    torch::Tensor norm_intensity,
    double objstage0_nm,
    double zemit0_nm);

torch::Tensor reduce_frames_cuda(
    torch::Tensor patches,
    torch::Tensor frame_indices,
    torch::Tensor center_x,
    torch::Tensor center_y,
    int64_t frame_count,
    int64_t height,
    int64_t width,
    double background);

torch::Tensor render_psf_bank(
    torch::Tensor coeffs_rad,
    torch::Tensor z_nm,
    torch::Tensor phase_basis,
    torch::Tensor amplitude,
    torch::Tensor polarization,
    torch::Tensor carrier,
    torch::Tensor kz_medium,
    torch::Tensor kz_immersion,
    torch::Tensor transform_y,
    torch::Tensor transform_x,
    torch::Tensor norm_intensity,
    double objstage0_nm,
    double zemit0_nm) {
  TORCH_CHECK(coeffs_rad.is_cuda(), "cuda_gemm renderer requires CUDA tensors");
  return render_psf_bank_cuda(
      coeffs_rad, z_nm, phase_basis, amplitude, polarization, carrier,
      kz_medium, kz_immersion, transform_y, transform_x, norm_intensity,
      objstage0_nm, zemit0_nm);
}

torch::Tensor reduce_frames(
    torch::Tensor patches,
    torch::Tensor frame_indices,
    torch::Tensor center_x,
    torch::Tensor center_y,
    int64_t frame_count,
    int64_t height,
    int64_t width,
    double background) {
  TORCH_CHECK(patches.is_cuda(), "reduce_frames requires CUDA tensors");
  return reduce_frames_cuda(patches, frame_indices, center_x, center_y,
                            frame_count, height, width, background);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_psf_bank", &render_psf_bank, "Render vector PSF bank (CUDA)");
  m.def("reduce_frames", &reduce_frames, "Reduce PSF patches into frames (CUDA)");
}
