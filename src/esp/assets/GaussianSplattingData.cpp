#include "GaussianSplattingData.h"

#include <Corrade/Utility/Debug.h>
#include <Magnum/GL/Buffer.h>

namespace esp {
namespace assets {

void GaussianSplattingData::uploadBuffersToGPU(bool forceReload) {
  if (buffersOnGPU_ && !forceReload) {
    return;
  }

  if (gaussians_.empty()) {
    ESP_WARNING() << "No Gaussian data to upload to GPU";
    return;
  }

  // Create rendering buffer if it doesn't exist
  if (!renderingBuffer_) {
    renderingBuffer_ = std::make_unique<RenderingBuffer>();
  }

  const size_t gaussianCount = gaussians_.size();
  const size_t shRestCount = getSHRestCount();

  // Prepare data arrays for GPU upload
  Corrade::Containers::Array<Mn::Vector3> positions(gaussianCount);
  Corrade::Containers::Array<Mn::Vector3> normals(gaussianCount);
  Corrade::Containers::Array<Mn::Vector3> shDC(gaussianCount);
  Corrade::Containers::Array<float> shRest(gaussianCount * shRestCount);
  Corrade::Containers::Array<float> opacities(gaussianCount);
  Corrade::Containers::Array<Mn::Vector3> scales(gaussianCount);
  Corrade::Containers::Array<Mn::Quaternion> rotations(gaussianCount);

  // Copy data from Gaussians to contiguous arrays
  for (size_t i = 0; i < gaussianCount; ++i) {
    const GaussianSplat& gaussian = gaussians_[i];
    positions[i] = gaussian.position;
    normals[i] = gaussian.normal;
    shDC[i] = gaussian.f_dc;
    opacities[i] = gaussian.opacity;
    scales[i] = gaussian.scale;
    rotations[i] = gaussian.rotation;

    // Copy SH rest coefficients
    for (size_t j = 0; j < shRestCount; ++j) {
      shRest[i * shRestCount + j] = gaussian.f_rest[j];
    }
  }

  // Upload to GPU buffers
  renderingBuffer_->positionBuffer.setData(positions,
                                           Mn::GL::BufferUsage::StaticDraw);
  renderingBuffer_->normalBuffer.setData(normals,
                                         Mn::GL::BufferUsage::StaticDraw);
  renderingBuffer_->shDCBuffer.setData(shDC, Mn::GL::BufferUsage::StaticDraw);
  renderingBuffer_->shRestBuffer.setData(shRest,
                                         Mn::GL::BufferUsage::StaticDraw);
  renderingBuffer_->opacityBuffer.setData(opacities,
                                          Mn::GL::BufferUsage::StaticDraw);
  renderingBuffer_->scaleBuffer.setData(scales,
                                        Mn::GL::BufferUsage::StaticDraw);
  renderingBuffer_->rotationBuffer.setData(rotations,
                                           Mn::GL::BufferUsage::StaticDraw);

  buffersOnGPU_ = true;

  ESP_DEBUG() << "Uploaded" << gaussianCount
              << "Gaussians to GPU with" << shRestCount << "SH rest coefficients each";
}

void GaussianSplattingData::addGaussian(GaussianSplat&& splat) {
  gaussians_.push_back(std::move(splat));
}

}  // namespace assets
}  // namespace esp

