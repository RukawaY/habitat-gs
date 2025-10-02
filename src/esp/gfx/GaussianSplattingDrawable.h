#ifndef ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_
#define ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_

#include <memory>

#include "esp/gfx/Drawable.h"
#include "esp/gfx/DrawableConfiguration.h"
#include "esp/gfx/GaussianRasterizer.h"

namespace esp {
namespace assets {
class GaussianSplattingData;
}
namespace gfx {

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
   * @param cfg Drawable configuration
   */
  explicit GaussianSplattingDrawable(
      scene::SceneNode& node,
      assets::GaussianSplattingData* gaussianData,
      DrawableConfiguration& cfg);

  ~GaussianSplattingDrawable() override = default;

  /**
   * @brief Get the drawable type
   */
  DrawableType getDrawableType() const override {
    return DrawableType::None;  // Custom type
  }

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
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_

