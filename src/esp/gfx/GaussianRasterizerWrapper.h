#ifndef ESP_GFX_GAUSSIANRASTERIZERWRAPPER_H_
#define ESP_GFX_GAUSSIANRASTERIZERWRAPPER_H_

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/GL/Texture.h>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/gfx/GaussianRasterizer.h"

namespace esp {
namespace gfx {

/**
 * @brief Wrapper function to call GaussianRasterizer from C++ code using Magnum types
 * 
 * This function converts Magnum types to plain C types before calling the CUDA implementation.
 * This is necessary because CUDA (nvcc) cannot compile code that includes Magnum headers.
 */
void renderGaussians(
    GaussianRasterizer& rasterizer,
    const assets::GaussianSplattingData& gaussianData,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& viewport,
    Magnum::GL::Texture2D& colorTexture,
    Magnum::GL::Texture2D& depthTexture,
    const Magnum::Vector3& background = Magnum::Vector3{0.0f, 0.0f, 0.0f});

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANRASTERIZERWRAPPER_H_

