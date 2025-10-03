#ifndef ESP_GFX_GAUSSIANRASTERIZER_H_
#define ESP_GFX_GAUSSIANRASTERIZER_H_

#include <memory>

namespace esp {
namespace gfx {

// Simple POD struct to pass Gaussian data without Magnum types
struct GaussianSplatSimple {
  float position[3];
  float normal[3];
  float f_dc[3];
  float f_rest[45];  // Higher-order SH coefficients (degree 3: 15 coefficients per channel, 3 channels)
  float opacity;
  float scale[3];
  float rotation[4];  // quaternion (x, y, z, w)
};

/**
 * @brief C++ wrapper for CUDA-based Gaussian Splatting rasterization
 * 
 * This class encapsulates:
 * - CUDA memory management
 * - CUDA-OpenGL interoperability (buffer/texture mapping)
 * - Calling the CUDA rasterization kernels
 */
class GaussianRasterizer {
 public:
  explicit GaussianRasterizer();
  ~GaussianRasterizer();

  // Non-copyable
  GaussianRasterizer(const GaussianRasterizer&) = delete;
  GaussianRasterizer& operator=(const GaussianRasterizer&) = delete;

  /**
   * @brief Render Gaussian Splatting to output textures/renderbuffers
   * 
   * @param gaussians Array of Gaussian splats
   * @param numGaussians Number of Gaussians
   * @param viewMatrix The view matrix (camera extrinsics) - 16 float array, row-major
   * @param projMatrix The projection matrix (camera intrinsics) - 16 float array, row-major
   * @param width Viewport width
   * @param height Viewport height
   * @param colorResourceId OpenGL renderbuffer/texture ID for RGB output
   * @param colorResourceType GL_RENDERBUFFER or GL_TEXTURE_2D
   * @param depthTextureId OpenGL texture ID for depth output (GL_TEXTURE_2D, DepthComponent32F)
   * @param backgroundR Background color R component
   * @param backgroundG Background color G component
   * @param backgroundB Background color B component
   */
  void render(const GaussianSplatSimple* gaussians,
              int numGaussians,
              const float* viewMatrix,
              const float* projMatrix,
              int width,
              int height,
              unsigned int colorResourceId,
              unsigned int colorResourceType,
              unsigned int depthTextureId,
              float backgroundR = 0.0f,
              float backgroundG = 0.0f,
              float backgroundB = 0.0f);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANRASTERIZER_H_

