#include "GaussianRasterizerWrapper.h"

#include <vector>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/Renderbuffer.h>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/gfx/RenderTarget.h"

namespace esp {
namespace gfx {

namespace {

// Helper function to convert GaussianSplattingData to simple POD structs
std::vector<GaussianSplatSimple> convertGaussianData(
    const assets::GaussianSplattingData& gaussianData) {
  const auto& gaussians = gaussianData.getGaussians();
  std::vector<GaussianSplatSimple> simpleGaussians;
  simpleGaussians.reserve(gaussians.size());
  
  for (size_t i = 0; i < gaussians.size(); ++i) {
    GaussianSplatSimple simple;
    const auto& g = gaussians[i];
    
    simple.position[0] = g.position.x();
    simple.position[1] = g.position.y();
    simple.position[2] = g.position.z();
    
    simple.normal[0] = g.normal.x();
    simple.normal[1] = g.normal.y();
    simple.normal[2] = g.normal.z();
    
    simple.f_dc[0] = g.f_dc.x();
    simple.f_dc[1] = g.f_dc.y();
    simple.f_dc[2] = g.f_dc.z();
    
    // Copy higher-order SH coefficients
    for (size_t j = 0; j < 45 && j < g.f_rest.size(); ++j) {
      simple.f_rest[j] = g.f_rest[j];
    }
    // Zero-fill if less than 45 coefficients
    for (size_t j = g.f_rest.size(); j < 45; ++j) {
      simple.f_rest[j] = 0.0f;
    }
    
    simple.opacity = g.opacity;
    
    simple.scale[0] = g.scale.x();
    simple.scale[1] = g.scale.y();
    simple.scale[2] = g.scale.z();
    
    simple.rotation[0] = g.rotation.vector().x();
    simple.rotation[1] = g.rotation.vector().y();
    simple.rotation[2] = g.rotation.vector().z();
    simple.rotation[3] = g.rotation.scalar();
    
    simpleGaussians.push_back(simple);
  }
  
  return simpleGaussians;
}

// Helper function to convert Magnum matrices to float arrays (row-major)
void convertMatrix(const Magnum::Matrix4& matrix, float* output) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      output[i * 4 + j] = matrix[j][i];  // Transpose (column-major to row-major)
    }
  }
}

}  // anonymous namespace

void renderGaussiansToRenderTarget(
    GaussianRasterizer& rasterizer,
    const assets::GaussianSplattingData& gaussianData,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& viewport,
    RenderTarget& renderTarget,
    const Magnum::Vector3& background) {
  
  // Convert matrices
  float viewMat[16];
  float projMat[16];
  convertMatrix(viewMatrix, viewMat);
  convertMatrix(projMatrix, projMat);
  
  // Convert Gaussian data
  std::vector<GaussianSplatSimple> simpleGaussians = convertGaussianData(gaussianData);
  
  // Get OpenGL resources from RenderTarget
  Magnum::GL::Renderbuffer& colorBuffer = renderTarget.getColorRenderbuffer();
  Magnum::GL::Texture2D& depthTexture = renderTarget.getDepthTexture();
  
  // GL constants
  #ifndef GL_RENDERBUFFER
  #define GL_RENDERBUFFER 0x8D41
  #endif
  #ifndef GL_TEXTURE_2D
  #define GL_TEXTURE_2D 0x0DE1
  #endif
  
  // Call CUDA implementation with RenderTarget resources
  rasterizer.render(
      simpleGaussians.data(),
      static_cast<int>(simpleGaussians.size()),
      viewMat,
      projMat,
      viewport.x(),
      viewport.y(),
      colorBuffer.id(),
      GL_RENDERBUFFER,
      depthTexture.id(),
      background.r(),
      background.g(),
      background.b());
}

void renderGaussians(
    GaussianRasterizer& rasterizer,
    const assets::GaussianSplattingData& gaussianData,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& viewport,
    Magnum::GL::Texture2D& colorTexture,
    Magnum::GL::Texture2D& depthTexture,
    const Magnum::Vector3& background) {
  
  // Convert matrices
  float viewMat[16];
  float projMat[16];
  convertMatrix(viewMatrix, viewMat);
  convertMatrix(projMatrix, projMat);
  
  // Convert Gaussian data
  std::vector<GaussianSplatSimple> simpleGaussians = convertGaussianData(gaussianData);
  
  #ifndef GL_TEXTURE_2D
  #define GL_TEXTURE_2D 0x0DE1
  #endif
  
  // Call CUDA implementation with textures
  rasterizer.render(
      simpleGaussians.data(),
      static_cast<int>(simpleGaussians.size()),
      viewMat,
      projMat,
      viewport.x(),
      viewport.y(),
      colorTexture.id(),
      GL_TEXTURE_2D,
      depthTexture.id(),
      background.r(),
      background.g(),
      background.b());
}

}  // namespace gfx
}  // namespace esp

