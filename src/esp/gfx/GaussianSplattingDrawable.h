#ifndef ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_
#define ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_

#include <memory>

#include "esp/gfx/Drawable.h"
#include "esp/gfx/DrawableConfiguration.h"
#include "esp/gfx/GaussianRasterizer.h"
#include "esp/gfx/ShaderManager.h"

namespace esp {
namespace assets {
class GaussianSplattingData;
}
namespace gfx {

class RenderTarget;

/**
 * @brief Drawable for rendering 3D Gaussian Splatting via CUDA
 * 
 * Unlike traditional drawables that use OpenGL shaders, this drawable
 * invokes CUDA kernels to rasterize Gaussian splats directly.
 */
class GaussianSplattingDrawable : public Drawable {
 public:
  /**
   * @brief Constructor
   * 
   * @param node Scene node to which this drawable is attached
   * @param gaussianData Pointer to the Gaussian Splatting data
   * @param shaderManager Reference to shader manager (for light setup)
   * @param cfg Drawable configuration
   * @param renderTarget Pointer to the RenderTarget (can be null, set later)
   */
  explicit GaussianSplattingDrawable(
      scene::SceneNode& node,
      assets::GaussianSplattingData* gaussianData,
      ShaderManager& shaderManager,
      DrawableConfiguration& cfg,
      RenderTarget* renderTarget = nullptr);

  /**
   * @brief Set the RenderTarget for this drawable
   * 
   * @param renderTarget Pointer to the RenderTarget to use for rendering
   */
  void setRenderTarget(RenderTarget* renderTarget) { renderTarget_ = renderTarget; }

  ~GaussianSplattingDrawable() override = default;

 protected:
  /**
   * @brief Draw the Gaussian Splatting scene using CUDA
   * 
   * @param transformationMatrix Transformation matrix from object to world space
   * @param camera The rendering camera
   */
  void draw(const Mn::Matrix4& transformationMatrix,
            Mn::SceneGraph::Camera3D& camera) override;

 private:
  assets::GaussianSplattingData* gaussianData_;
  std::shared_ptr<GaussianRasterizer> rasterizer_;
  RenderTarget* renderTarget_;
  
  // Cache converted Gaussian data to avoid per-frame conversion
  std::vector<GaussianSplatSimple> gaussiansCache_;
  bool cacheValid_ = false;
  
  void updateGaussiansCache();
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_

