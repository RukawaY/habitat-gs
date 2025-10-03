#ifndef ESP_GFX_GAUSSIANRASTERIZERWRAPPER_H_
#define ESP_GFX_GAUSSIANRASTERIZERWRAPPER_H_

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/Renderbuffer.h>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/gfx/GaussianRasterizer.h"

namespace esp {
namespace gfx {

class RenderTarget;

/**
 * @brief Wrapper to call GaussianRasterizer from C++ code using Magnum types
 * 
 * This wrapper converts Magnum types to plain C types before calling the CUDA implementation.
 * This is necessary because CUDA (nvcc) cannot compile code that includes Magnum headers.
 */

/**
 * @brief Render Gaussians directly to RenderTarget
 * 
 * @param rasterizer The CUDA rasterizer instance
 * @param gaussianData The Gaussian Splatting data to render
 * @param viewMatrix View matrix (world to camera)
 * @param projMatrix Projection matrix
 * @param viewport Viewport size
 * @param renderTarget RenderTarget to render into
 * @param background Background color
 */
void renderGaussiansToRenderTarget(
    GaussianRasterizer& rasterizer,
    const assets::GaussianSplattingData& gaussianData,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& viewport,
    RenderTarget& renderTarget,
    const Magnum::Vector3& background = Magnum::Vector3{0.0f, 0.0f, 0.0f});

/**
 * @brief Render Gaussians to separate color and depth textures
 * 
 * @deprecated Use renderGaussiansToRenderTarget instead for proper integration
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

