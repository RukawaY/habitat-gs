#include "GaussianSplattingDrawable.h"

#include "esp/assets/GaussianSplattingData.h"
#include "esp/gfx/GaussianRasterizerWrapper.h"
#include "esp/gfx/LightSetup.h"
#include "esp/gfx/RenderCamera.h"
#include "esp/gfx/RenderTarget.h"
#include "esp/gfx/ShaderManager.h"
#include "esp/scene/SceneNode.h"

namespace esp {
namespace gfx {

GaussianSplattingDrawable::GaussianSplattingDrawable(
    scene::SceneNode& node,
    assets::GaussianSplattingData* gaussianData,
    ShaderManager& shaderManager,
    DrawableConfiguration& cfg,
    RenderTarget* renderTarget)
    : Drawable{node, nullptr, DrawableType::None, cfg,
               shaderManager.get<LightSetup>(cfg.lightSetupKey_)},
      gaussianData_(gaussianData),
      rasterizer_(std::make_shared<GaussianRasterizer>()),
      renderTarget_(renderTarget) {
  CORRADE_INTERNAL_ASSERT(gaussianData_);
}

void GaussianSplattingDrawable::draw(
    const Mn::Matrix4& transformationMatrix,
    Mn::SceneGraph::Camera3D& camera) {
  
  if (!gaussianData_ || gaussianData_->getGaussianCount() == 0) {
    ESP_DEBUG() << "No Gaussian data to render";
    return;
  }

  if (!renderTarget_) {
    ESP_WARNING() << "GaussianSplattingDrawable::draw() - No RenderTarget set, skipping rendering";
    return;
  }

  // Compute view and projection matrices
  // View matrix: from world space to camera space
  Mn::Matrix4 cameraMatrix = camera.cameraMatrix();
  // Combine with object transformation
  Mn::Matrix4 viewMatrix = cameraMatrix * transformationMatrix;

  // Projection matrix
  Mn::Matrix4 projMatrix = camera.projectionMatrix();

  // Get viewport size
  Mn::Vector2i viewport = camera.viewport();

  // Background color - use transparent black to allow blending with existing content
  Mn::Vector3 background{0.0f, 0.0f, 0.0f};

  // Render using the wrapper function (clean call hierarchy)
  renderGaussiansToRenderTarget(
      *rasterizer_,
      *gaussianData_,
      viewMatrix,
      projMatrix,
      viewport,
      *renderTarget_,
      background
  );

  ESP_DEBUG() << "GaussianSplattingDrawable rendered " 
              << gaussianData_->getGaussianCount() << " Gaussians to RenderTarget";
}

}  // namespace gfx
}  // namespace esp

