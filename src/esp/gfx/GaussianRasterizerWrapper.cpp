#include "GaussianRasterizerWrapper.h"

#include <vector>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/GL/Texture.h>

#include "esp/assets/GaussianSplattingData.h"

namespace esp {
namespace gfx {

void renderGaussians(
    GaussianRasterizer& rasterizer,
    const assets::GaussianSplattingData& gaussianData,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& viewport,
    Magnum::GL::Texture2D& colorTexture,
    Magnum::GL::Texture2D& depthTexture,
    const Magnum::Vector3& background) {
  
  // Convert Magnum matrices to float arrays (row-major)
  float viewMat[16];
  float projMat[16];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      viewMat[i * 4 + j] = viewMatrix[j][i];  // Transpose (column-major to row-major)
      projMat[i * 4 + j] = projMatrix[j][i];  // Transpose
    }
  }
  
  // Convert GaussianSplattingData to simple POD structs
  const auto& gaussians = gaussianData.getGaussians();
  std::vector<GaussianSplatSimple> simpleGaussians(gaussians.size());
  
  for (size_t i = 0; i < gaussians.size(); ++i) {
    const auto& g = gaussians[i];
    simpleGaussians[i].position[0] = g.position.x();
    simpleGaussians[i].position[1] = g.position.y();
    simpleGaussians[i].position[2] = g.position.z();
    
    simpleGaussians[i].normal[0] = g.normal.x();
    simpleGaussians[i].normal[1] = g.normal.y();
    simpleGaussians[i].normal[2] = g.normal.z();
    
    simpleGaussians[i].f_dc[0] = g.f_dc.x();
    simpleGaussians[i].f_dc[1] = g.f_dc.y();
    simpleGaussians[i].f_dc[2] = g.f_dc.z();
    
    // Copy higher-order SH coefficients
    for (size_t j = 0; j < 45 && j < g.f_rest.size(); ++j) {
      simpleGaussians[i].f_rest[j] = g.f_rest[j];
    }
    // Zero-fill if less than 45 coefficients
    for (size_t j = g.f_rest.size(); j < 45; ++j) {
      simpleGaussians[i].f_rest[j] = 0.0f;
    }
    
    simpleGaussians[i].opacity = g.opacity;
    
    simpleGaussians[i].scale[0] = g.scale.x();
    simpleGaussians[i].scale[1] = g.scale.y();
    simpleGaussians[i].scale[2] = g.scale.z();
    
    simpleGaussians[i].rotation[0] = g.rotation.vector().x();
    simpleGaussians[i].rotation[1] = g.rotation.vector().y();
    simpleGaussians[i].rotation[2] = g.rotation.vector().z();
    simpleGaussians[i].rotation[3] = g.rotation.scalar();
  }
  
  // Call CUDA implementation
  rasterizer.render(
      simpleGaussians.data(),
      static_cast<int>(simpleGaussians.size()),
      viewMat,
      projMat,
      viewport.x(),
      viewport.y(),
      colorTexture.id(),
      depthTexture.id(),
      background.r(),
      background.g(),
      background.b());
}

}  // namespace gfx
}  // namespace esp

