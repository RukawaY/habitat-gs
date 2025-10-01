#ifndef ESP_ASSETS_GAUSSIANSPLATTINGIMPORTER_H_
#define ESP_ASSETS_GAUSSIANSPLATTINGIMPORTER_H_

/** @file
 * @brief Class @ref esp::assets::GaussianSplattingImporter
 */

#include <Corrade/Containers/Optional.h>
#include <Magnum/Trade/AbstractImporter.h>

#include <fstream>
#include <string>
#include <vector>

namespace Mn = Magnum;

namespace esp {
namespace assets {

/**
 * @brief Custom importer for 3D Gaussian Splatting PLY files.
 *
 * This importer extends Magnum::Trade::AbstractImporter to support loading
 * 3DGS PLY files with the specific format containing Gaussian parameters
 * (position, normal, spherical harmonics, opacity, scale, rotation).
 */
class GaussianSplattingImporter : public Mn::Trade::AbstractImporter {
 public:
  /**
   * @brief Constructor
   */
  explicit GaussianSplattingImporter();

  /**
   * @brief Destructor
   */
  ~GaussianSplattingImporter() override = default;

  /**
   * @brief Get importer features
   */
  Mn::Trade::ImporterFeatures doFeatures() const override;

  /**
   * @brief Check if the file is a 3DGS PLY file
   */
  bool doIsOpened() const override;

  /**
   * @brief Open a 3DGS PLY file from the filesystem
   * @param filename Path to the PLY file
   */
  void doOpenFile(Corrade::Containers::StringView filename) override;

  /**
   * @brief Close the currently opened file
   */
  void doClose() override;

  /**
   * @brief Get the number of meshes (always 1 for 3DGS data)
   */
  Mn::UnsignedInt doMeshCount() const override;

  /**
   * @brief Load mesh data at given index (converts Gaussians to point cloud)
   * @param id Mesh index (should be 0)
   */
  Corrade::Containers::Optional<Mn::Trade::MeshData> doMesh(
      Mn::UnsignedInt id,
      Mn::UnsignedInt level) override;

  /**
   * @brief Get the loaded Gaussian count
   */
  size_t getGaussianCount() const { return gaussianCount_; }

  /**
   * @brief Get all loaded Gaussian positions
   */
  const std::vector<Magnum::Vector3>& getPositions() const { 
    return positions_; 
  }

  /**
   * @brief Get all loaded Gaussian normals
   */
  const std::vector<Magnum::Vector3>& getNormals() const { 
    return normals_; 
  }

  /**
   * @brief Get all loaded Gaussian SH DC coefficients
   */
  const std::vector<Magnum::Vector3>& getSHDC() const { 
    return sh_dc_; 
  }

  /**
   * @brief Get all loaded Gaussian SH rest coefficients
   */
  const std::vector<std::vector<float>>& getSHRest() const { 
    return sh_rest_; 
  }

  /**
   * @brief Get all loaded Gaussian opacities
   */
  const std::vector<float>& getOpacities() const { 
    return opacities_; 
  }

  /**
   * @brief Get all loaded Gaussian scales
   */
  const std::vector<Magnum::Vector3>& getScales() const { 
    return scales_; 
  }

  /**
   * @brief Get all loaded Gaussian rotations
   */
  const std::vector<Magnum::Quaternion>& getRotations() const { 
    return rotations_; 
  }

 private:
  /**
   * @brief Parse PLY header and determine property layout
   */
  bool parsePLYHeader(std::ifstream& file);

  /**
   * @brief Read binary PLY data
   */
  bool readBinaryData(std::ifstream& file);

  /**
   * @brief Convert little-endian bytes to float
   */
  float bytesToFloat(const unsigned char* bytes) const;

  //! Whether a file is currently opened
  bool opened_ = false;

  //! Number of Gaussians loaded
  size_t gaussianCount_ = 0;

  //! Number of SH rest coefficients per Gaussian
  size_t shRestCount_ = 0;

  //! Loaded Gaussian positions
  std::vector<Magnum::Vector3> positions_;

  //! Loaded Gaussian normals
  std::vector<Magnum::Vector3> normals_;

  //! Loaded Gaussian SH DC coefficients (base color)
  std::vector<Magnum::Vector3> sh_dc_;

  //! Loaded Gaussian SH rest coefficients
  std::vector<std::vector<float>> sh_rest_;

  //! Loaded Gaussian opacities
  std::vector<float> opacities_;

  //! Loaded Gaussian scales
  std::vector<Magnum::Vector3> scales_;

  //! Loaded Gaussian rotations (quaternions)
  std::vector<Magnum::Quaternion> rotations_;
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_GAUSSIANSPLATTINGIMPORTER_H_

