#include "GaussianRasterizer.h"

// Standard library includes
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>

// CUDA headers
#include <cuda_runtime.h>

// Forward declare only what we need - avoid including Magnum headers
namespace esp {
namespace assets {
class GaussianSplattingData;

// Simple POD struct to pass Gaussian data without Magnum types
struct GaussianSplatSimple {
  float position[3];
  float normal[3];
  float f_dc[3];
  float opacity;
  float scale[3];
  float rotation[4];  // quaternion (x, y, z, w)
};
}  // namespace assets
}  // namespace esp

// Include GL types after checking they're available
#ifdef __CUDACC__
// When compiling with nvcc, use CUDA's GL interop which defines GL types
#include <cuda_gl_interop.h>
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0DE1
#endif
#endif

// Include rasterizer
#include "esp/gfx/gaussian_rasterizer/rasterizer.h"

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - "  \
                << cudaGetErrorString(err) << std::endl;                      \
      assert(false && "CUDA error occurred");                                 \
    }                                                                         \
  } while (0)

namespace esp {
namespace gfx {

// Helper function to allocate memory via lambda
char* allocateBuffer(void*& bufferPtr, size_t size) {
  CUDA_CHECK(cudaMalloc(&bufferPtr, size));
  return reinterpret_cast<char*>(bufferPtr);
}

// Implementation struct that holds CUDA resources
struct GaussianRasterizer::Impl {
  // CUDA memory for Gaussian data
  void* d_positions = nullptr;
  void* d_normals = nullptr;
  void* d_sh_dc = nullptr;
  void* d_sh_rest = nullptr;
  void* d_opacities = nullptr;
  void* d_scales = nullptr;
  void* d_rotations = nullptr;

  // CUDA-OpenGL interop resources
  cudaGraphicsResource* colorTexResource = nullptr;
  cudaGraphicsResource* depthTexResource = nullptr;

  // Persistent CUDA buffers for rasterization
  void* geomBuffer = nullptr;
  void* binningBuffer = nullptr;
  void* imageBuffer = nullptr;

  int lastGaussianCount = 0;
  int lastWidth = 0;
  int lastHeight = 0;

  ~Impl() {
    cleanup();
  }

  void cleanup() {
    // Unregister OpenGL resources
    if (colorTexResource) {
      cudaGraphicsUnregisterResource(colorTexResource);
      colorTexResource = nullptr;
    }
    if (depthTexResource) {
      cudaGraphicsUnregisterResource(depthTexResource);
      depthTexResource = nullptr;
    }

    // Free CUDA memory
    if (d_positions) cudaFree(d_positions);
    if (d_normals) cudaFree(d_normals);
    if (d_sh_dc) cudaFree(d_sh_dc);
    if (d_sh_rest) cudaFree(d_sh_rest);
    if (d_opacities) cudaFree(d_opacities);
    if (d_scales) cudaFree(d_scales);
    if (d_rotations) cudaFree(d_rotations);
    if (geomBuffer) cudaFree(geomBuffer);
    if (binningBuffer) cudaFree(binningBuffer);
    if (imageBuffer) cudaFree(imageBuffer);

    d_positions = d_normals = d_sh_dc = d_sh_rest = nullptr;
    d_opacities = d_scales = d_rotations = nullptr;
    geomBuffer = binningBuffer = imageBuffer = nullptr;
  }

  void uploadGaussianData(const GaussianSplatSimple* gaussians, int P) {
    // Only reallocate if size changed
    if (P != lastGaussianCount) {
      if (d_positions) cudaFree(d_positions);
      if (d_normals) cudaFree(d_normals);
      if (d_sh_dc) cudaFree(d_sh_dc);
      if (d_opacities) cudaFree(d_opacities);
      if (d_scales) cudaFree(d_scales);
      if (d_rotations) cudaFree(d_rotations);

      CUDA_CHECK(cudaMalloc(&d_positions, P * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_normals, P * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_sh_dc, P * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_opacities, P * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_scales, P * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_rotations, P * 4 * sizeof(float)));

      lastGaussianCount = P;
    }

    // Prepare CPU buffers
    std::vector<float> positions(P * 3);
    std::vector<float> normals(P * 3);
    std::vector<float> sh_dc(P * 3);
    std::vector<float> opacities(P);
    std::vector<float> scales(P * 3);
    std::vector<float> rotations(P * 4);

    for (int i = 0; i < P; ++i) {
      const auto& g = gaussians[i];
      positions[i * 3 + 0] = g.position[0];
      positions[i * 3 + 1] = g.position[1];
      positions[i * 3 + 2] = g.position[2];

      normals[i * 3 + 0] = g.normal[0];
      normals[i * 3 + 1] = g.normal[1];
      normals[i * 3 + 2] = g.normal[2];

      sh_dc[i * 3 + 0] = g.f_dc[0];
      sh_dc[i * 3 + 1] = g.f_dc[1];
      sh_dc[i * 3 + 2] = g.f_dc[2];

      opacities[i] = g.opacity;

      scales[i * 3 + 0] = g.scale[0];
      scales[i * 3 + 1] = g.scale[1];
      scales[i * 3 + 2] = g.scale[2];

      rotations[i * 4 + 0] = g.rotation[0];
      rotations[i * 4 + 1] = g.rotation[1];
      rotations[i * 4 + 2] = g.rotation[2];
      rotations[i * 4 + 3] = g.rotation[3];
    }

    // Upload to GPU
    CUDA_CHECK(cudaMemcpy(d_positions, positions.data(), P * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, normals.data(), P * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_dc, sh_dc.data(), P * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities, opacities.data(), P * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), P * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotations, rotations.data(), P * 4 * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
};

GaussianRasterizer::GaussianRasterizer() : impl_(std::make_unique<Impl>()) {}

GaussianRasterizer::~GaussianRasterizer() = default;

void GaussianRasterizer::render(
    const GaussianSplatSimple* gaussians,
    int numGaussians,
    const float* viewMatrix,
    const float* projMatrix,
    int width,
    int height,
    unsigned int colorTextureId,
    unsigned int depthTextureId,
    float backgroundR,
    float backgroundG,
    float backgroundB) {
  const int W = width;
  const int H = height;
  const int P = numGaussians;

  if (P == 0) {
    std::cerr << "Warning: No Gaussians to render" << std::endl;
    return;
  }

  // Upload Gaussian data to CUDA
  impl_->uploadGaussianData(gaussians, P);

  // Register OpenGL textures with CUDA (if not already registered)
  if (!impl_->colorTexResource) {
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &impl_->colorTexResource, colorTextureId,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
  }
  if (!impl_->depthTexResource) {
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &impl_->depthTexResource, depthTextureId,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
  }

  // Map OpenGL textures to CUDA
  CUDA_CHECK(cudaGraphicsMapResources(1, &impl_->colorTexResource, 0));
  CUDA_CHECK(cudaGraphicsMapResources(1, &impl_->depthTexResource, 0));

  cudaArray* colorArray = nullptr;
  cudaArray* depthArray = nullptr;
  CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
      &colorArray, impl_->colorTexResource, 0, 0));
  CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
      &depthArray, impl_->depthTexResource, 0, 0));

  // Allocate output buffers in device memory
  float* d_colorOutput = nullptr;
  float* d_depthOutput = nullptr;
  CUDA_CHECK(cudaMalloc(&d_colorOutput, W * H * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_depthOutput, W * H * sizeof(float)));

  // Matrices are already in float* format (row-major)
  const float* viewmat = viewMatrix;
  const float* projmat = projMatrix;

  // Compute camera parameters
  float focal_x = projmat[0] * W / 2.0f;
  float focal_y = projmat[5] * H / 2.0f;
  float tan_fovx = 1.0f / projmat[0];
  float tan_fovy = 1.0f / projmat[5];

  // Camera position - extract from view matrix
  // For simplicity, assuming camera is at origin (can be improved)
  float cam_pos[3] = {-viewmat[12], -viewmat[13], -viewmat[14]};

  // Background color
  float bg_color[3] = {backgroundR, backgroundG, backgroundB};

  // Allocate persistent buffers using lambdas
  auto geometryBufferFunc = [this](size_t size) {
    return allocateBuffer(impl_->geomBuffer, size);
  };
  auto binningBufferFunc = [this](size_t size) {
    return allocateBuffer(impl_->binningBuffer, size);
  };
  auto imageBufferFunc = [this](size_t size) {
    return allocateBuffer(impl_->imageBuffer, size);
  };

  // Call CUDA rasterizer
  int D = 3;  // SH degree 0 (only DC component)
  int M = 0;  // No higher order SH

  CudaRasterizer::Rasterizer::forward(
      geometryBufferFunc, binningBufferFunc, imageBufferFunc,
      P, D, M, bg_color, W, H,
      reinterpret_cast<float*>(impl_->d_positions),    // means3D
      nullptr,                                         // shs (use precomp colors)
      reinterpret_cast<float*>(impl_->d_sh_dc),        // colors_precomp
      reinterpret_cast<float*>(impl_->d_opacities),    // opacities
      reinterpret_cast<float*>(impl_->d_scales),       // scales
      1.0f,                                            // scale_modifier
      reinterpret_cast<float*>(impl_->d_rotations),    // rotations
      nullptr,                                         // cov3D_precomp
      viewmat, projmat, cam_pos, tan_fovx, tan_fovy,
      false,                                           // prefiltered
      d_colorOutput,                                   // out_color
      d_depthOutput,                                   // depth (now supported!)
      false,                                           // antialiasing
      nullptr,                                         // radii
      false);                                          // debug

  // Copy color output to OpenGL texture via CUDA array
  CUDA_CHECK(cudaMemcpy2DToArray(
      colorArray, 0, 0, d_colorOutput, W * 3 * sizeof(float),
      W * 3 * sizeof(float), H, cudaMemcpyDeviceToDevice));

  // Copy depth output to OpenGL texture via CUDA array
  CUDA_CHECK(cudaMemcpy2DToArray(
      depthArray, 0, 0, d_depthOutput, W * sizeof(float),
      W * sizeof(float), H, cudaMemcpyDeviceToDevice));

  // Cleanup
  cudaFree(d_colorOutput);
  cudaFree(d_depthOutput);

  // Unmap resources
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &impl_->colorTexResource, 0));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &impl_->depthTexResource, 0));

  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace gfx
}  // namespace esp

