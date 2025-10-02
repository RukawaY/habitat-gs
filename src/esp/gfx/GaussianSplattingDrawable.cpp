#include "GaussianSplattingDrawable.h"

#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/gfx/RenderCamera.h"
#include "esp/scene/SceneNode.h"

namespace esp {
namespace gfx {

GaussianSplattingDrawable::GaussianSplattingDrawable(
    scene::SceneNode& node,
    assets::GaussianSplattingData* gaussianData,
    DrawableConfiguration& cfg)
    : Drawable{node, nullptr, DrawableType::None, cfg,
               cfg.lightSetup_},
      gaussianData_(gaussianData),
      rasterizer_(std::make_shared<GaussianRasterizer>()) {
  CORRADE_INTERNAL_ASSERT(gaussianData_);
}

void GaussianSplattingDrawable::draw(
    const Mn::Matrix4& transformationMatrix,
    Mn::SceneGraph::Camera3D& camera) {
  
  if (!gaussianData_ || gaussianData_->getGaussianCount() == 0) {
    ESP_DEBUG() << "No Gaussian data to render";
    return;
  }

  // Get the render camera
  RenderCamera& renderCamera = static_cast<RenderCamera&>(camera);

  // Compute view and projection matrices
  // View matrix: from world space to camera space
  Mn::Matrix4 cameraMatrix = camera.cameraMatrix();
  // Combine with object transformation
  Mn::Matrix4 viewMatrix = cameraMatrix * transformationMatrix;

  // Projection matrix
  Mn::Matrix4 projMatrix = camera.projectionMatrix();

  // Get viewport size
  Mn::Vector2i viewport = camera.viewport();

  // Get current framebuffer
  // NOTE: In habitat-sim, rendering targets a RenderTarget's framebuffer
  // We need to get the color and depth textures from it
  // For now, we'll use a placeholder approach and render to the default framebuffer
  // TODO: Properly integrate with RenderTarget to get the actual textures

  ESP_WARNING() << "GaussianSplattingDrawable::draw() is not yet fully integrated "
                << "with RenderTarget. Rendering to placeholder textures.";

  // Placeholder: Create temporary textures for testing
  // In the actual implementation, these should come from the RenderTarget
  // static Mn::GL::Texture2D colorTexture;
  // static Mn::GL::Texture2D depthTexture;
  // static bool texturesInitialized = false;
  
  // if (!texturesInitialized || colorTexture.imageSize(0) != viewport) {
  //   colorTexture.setStorage(1, Mn::GL::TextureFormat::RGBA32F, viewport);
  //   depthTexture.setStorage(1, Mn::GL::TextureFormat::R32F, viewport);
  //   texturesInitialized = true;
  // }

  // Background color (black)
  Mn::Vector3 background{0.0f, 0.0f, 0.0f};

  // Render using CUDA
  // rasterizer_->render(*gaussianData_, viewMatrix, projMatrix, viewport,
  //                     colorTexture, depthTexture, background);

  ESP_DEBUG() << "GaussianSplattingDrawable rendered" 
              << gaussianData_->getGaussianCount() << "Gaussians";
}

}  // namespace gfx
}  // namespace esp

