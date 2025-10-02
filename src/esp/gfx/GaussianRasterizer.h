#ifndef ESP_GFX_GAUSSIANRASTERIZER_H_
#define ESP_GFX_GAUSSIANRASTERIZER_H_

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector3.h>
#include <memory>

#include "esp/core/Esp.h"

namespace esp {
namespace assets {
class GaussianSplattingData;
}
namespace gfx {

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
   * @brief Render Gaussian Splatting to output textures
   * 
   * @param gaussianData The Gaussian Splatting data to render
   * @param viewMatrix The view matrix (camera extrinsics)
   * @param projMatrix The projection matrix (camera intrinsics)
   * @param viewport The viewport size (width, height)
   * @param colorTexture OpenGL texture for RGB output (GL_RGBA32F)
   * @param depthTexture OpenGL texture for depth output (GL_R32F)
   * @param background Background color (RGB)
   */
  void render(const assets::GaussianSplattingData& gaussianData,
              const Mn::Matrix4& viewMatrix,
              const Mn::Matrix4& projMatrix,
              const Mn::Vector2i& viewport,
              Mn::GL::Texture2D& colorTexture,
              Mn::GL::Texture2D& depthTexture,
              const Mn::Vector3& background = Mn::Vector3{0.0f, 0.0f, 0.0f});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANRASTERIZER_H_

