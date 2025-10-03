#include "GaussianSplattingDrawable.h"

#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderbuffer.h>

#include "esp/assets/GaussianSplattingData.h"
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
      renderTarget_(renderTarget),
      cacheValid_(false) {
  CORRADE_INTERNAL_ASSERT(gaussianData_);
  // Pre-convert Gaussian data for efficiency
  updateGaussiansCache();
}

void GaussianSplattingDrawable::updateGaussiansCache() {
  if (!gaussianData_) {
    return;
  }

  const auto& gaussians = gaussianData_->getGaussians();
  gaussiansCache_.clear();
  gaussiansCache_.reserve(gaussians.size());
  
  // Convert GaussianSplattingData to GaussianSplatSimple format
  for (size_t i = 0; i < gaussians.size(); ++i) {
    GaussianSplatSimple g;
    const auto& splat = gaussians[i];
    
    // Fill simple struct
    g.position[0] = splat.position.x();
    g.position[1] = splat.position.y();
    g.position[2] = splat.position.z();
    
    g.normal[0] = splat.normal.x();
    g.normal[1] = splat.normal.y();
    g.normal[2] = splat.normal.z();
    
    // DC component (base color)
    g.f_dc[0] = splat.f_dc.x();
    g.f_dc[1] = splat.f_dc.y();
    g.f_dc[2] = splat.f_dc.z();
    
    // Higher-order SH coefficients
    size_t restCount = splat.f_rest.size();
    for (size_t j = 0; j < 45; ++j) {
      g.f_rest[j] = (j < restCount) ? splat.f_rest[j] : 0.0f;
    }
    
    g.opacity = splat.opacity;
    
    g.scale[0] = splat.scale.x();
    g.scale[1] = splat.scale.y();
    g.scale[2] = splat.scale.z();
    
    g.rotation[0] = splat.rotation.vector().x();
    g.rotation[1] = splat.rotation.vector().y();
    g.rotation[2] = splat.rotation.vector().z();
    g.rotation[3] = splat.rotation.scalar();
    
    gaussiansCache_.push_back(g);
  }
  
  cacheValid_ = true;
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

  // Get OpenGL resources from RenderTarget
  Mn::GL::Renderbuffer& colorBuffer = renderTarget_->getColorRenderbuffer();
  Mn::GL::Texture2D& depthTexture = renderTarget_->getDepthTexture();

  // Extract OpenGL IDs
  unsigned int colorBufferId = colorBuffer.id();
  unsigned int depthTextureId = depthTexture.id();

  // Background color - use clear color from the RenderTarget if available
  // For now, use transparent black to allow blending with existing content
  Mn::Vector3 background{0.0f, 0.0f, 0.0f};

  // Update cache if needed
  if (!cacheValid_) {
    updateGaussiansCache();
  }

  int gaussianCount = gaussiansCache_.size();

  // Prepare view and projection matrices as float arrays (row-major)
  float viewMatArray[16];
  float projMatArray[16];
  
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      viewMatArray[i * 4 + j] = viewMatrix[i][j];
      projMatArray[i * 4 + j] = projMatrix[i][j];
    }
  }

  // Render using CUDA to the RenderTarget's resources
  // Use GL_RENDERBUFFER for color and GL_TEXTURE_2D for depth
  #ifndef GL_RENDERBUFFER
  #define GL_RENDERBUFFER 0x8D41
  #endif
  #ifndef GL_TEXTURE_2D
  #define GL_TEXTURE_2D 0x0DE1
  #endif
  
  rasterizer_->render(
      gaussiansCache_.data(),
      gaussianCount,
      viewMatArray,
      projMatArray,
      viewport.x(),
      viewport.y(),
      colorBufferId,
      GL_RENDERBUFFER,
      depthTextureId,
      background.r(),
      background.g(),
      background.b()
  );

  ESP_DEBUG() << "GaussianSplattingDrawable rendered " 
              << gaussianCount << " Gaussians to RenderTarget";
}

}  // namespace gfx
}  // namespace esp

