#ifndef ESP_ASSETS_GAUSSIANSPLATTINGDATA_H_
#define ESP_ASSETS_GAUSSIANSPLATTINGDATA_H_

/** @file
 * @brief Class @ref esp::assets::GaussianSplattingData
 */

#include <Corrade/Containers/Array.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector3.h>

#include "BaseMesh.h"
#include "esp/core/Esp.h"

namespace esp {
namespace assets {

/**
 * @brief Structure representing a single 3D Gaussian Splat.
 *
 * Each Gaussian is defined by its position, orientation, scale, color
 * (spherical harmonics coefficients), and opacity.
 */
struct GaussianSplat {
  //! Position of the Gaussian center
  Magnum::Vector3 position;

  //! Normal vector (nx, ny, nz)
  Magnum::Vector3 normal;

  //! Spherical harmonics DC coefficients (RGB base color)
  Magnum::Vector3 f_dc;  // f_dc_0, f_dc_1, f_dc_2

  //! Spherical harmonics higher-order coefficients (45 floats for degree 3)
  //! f_rest_0 to f_rest_44
  Corrade::Containers::Array<float> f_rest;

  //! Opacity of the Gaussian
  float opacity;

  //! Scale of the Gaussian in 3 axes
  Magnum::Vector3 scale;  // scale_0, scale_1, scale_2

  //! Rotation quaternion (rot_0, rot_1, rot_2, rot_3)
  Magnum::Quaternion rotation;

  /**
   * @brief Default constructor
   */
  GaussianSplat() : opacity(1.0f) {}
};

/**
 * @brief Data storage class for 3D Gaussian Splatting assets.
 *
 * This class stores and manages 3D Gaussian Splatting data loaded from PLY
 * files. Unlike traditional mesh data, Gaussian Splatting represents scenes
 * as a collection of oriented 3D Gaussians.
 */
class GaussianSplattingData : public BaseMesh {
 public:
  /**
   * @brief Rendering buffer structure for GPU upload.
   */
  struct RenderingBuffer {
    //! Buffer containing Gaussian positions
    Magnum::GL::Buffer positionBuffer;
    //! Buffer containing Gaussian normals
    Magnum::GL::Buffer normalBuffer;
    //! Buffer containing spherical harmonics DC coefficients
    Magnum::GL::Buffer shDCBuffer;
    //! Buffer containing spherical harmonics rest coefficients
    Magnum::GL::Buffer shRestBuffer;
    //! Buffer containing opacity values
    Magnum::GL::Buffer opacityBuffer;
    //! Buffer containing scale values
    Magnum::GL::Buffer scaleBuffer;
    //! Buffer containing rotation quaternions
    Magnum::GL::Buffer rotationBuffer;
  };

  /**
   * @brief Constructor. Sets asset type to GAUSSIAN_SPLATTING.
   */
  explicit GaussianSplattingData()
      : BaseMesh(SupportedMeshType::GAUSSIAN_SPLATTING) {}

  /**
   * @brief Destructor
   */
  ~GaussianSplattingData() override = default;

  /**
   * @brief Upload Gaussian data to GPU buffers.
   * @param forceReload If true, recompiles the buffers even if already uploaded.
   */
  void uploadBuffersToGPU(bool forceReload = false) override;

  /**
   * @brief Add a Gaussian splat to the data.
   * @param splat The Gaussian splat to add.
   */
  void addGaussian(const GaussianSplat& splat);

  /**
   * @brief Get the number of Gaussian splats.
   * @return Number of Gaussians stored.
   */
  size_t getGaussianCount() const { return gaussians_.size(); }

  /**
   * @brief Get read-only access to Gaussian data.
   * @return Const reference to the vector of Gaussians.
   */
  const std::vector<GaussianSplat>& getGaussians() const { return gaussians_; }

  /**
   * @brief Get the rendering buffer (for GPU rendering).
   * @return Pointer to the rendering buffer, or nullptr if not uploaded.
   */
  RenderingBuffer* getRenderingBuffer() {
    return renderingBuffer_.get();
  }

  /**
   * @brief Reserve space for a given number of Gaussians.
   * @param count Number of Gaussians to reserve space for.
   */
  void reserve(size_t count) { gaussians_.reserve(count); }

  /**
   * @brief Clear all Gaussian data.
   */
  void clear() {
    gaussians_.clear();
    buffersOnGPU_ = false;
    renderingBuffer_.reset();
  }

  /**
   * @brief Get the number of spherical harmonics coefficients per Gaussian.
   * @return Number of SH rest coefficients (typically 45 for degree 3).
   */
  size_t getSHRestCount() const {
    return gaussians_.empty() ? 0 : gaussians_[0].f_rest.size();
  }

 protected:
  //! Vector storing all Gaussian splats
  std::vector<GaussianSplat> gaussians_;

  //! Rendering buffer for GPU upload
  std::unique_ptr<RenderingBuffer> renderingBuffer_ = nullptr;
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_GAUSSIANSPLATTINGDATA_H_

