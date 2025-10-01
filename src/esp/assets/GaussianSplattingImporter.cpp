// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "GaussianSplattingImporter.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/MeshTools/Compile.h>

#include <fstream>
#include <sstream>
#include <cstring>

#include "esp/core/Esp.h"

namespace Cr = Corrade;

namespace esp {
namespace assets {

GaussianSplattingImporter::GaussianSplattingImporter() = default;

Mn::Trade::ImporterFeatures GaussianSplattingImporter::doFeatures() const {
  return Mn::Trade::ImporterFeature::OpenData |
         Mn::Trade::ImporterFeature::FileCallback;
}

bool GaussianSplattingImporter::doIsOpened() const {
  return opened_;
}

void GaussianSplattingImporter::doOpenFile(
    Corrade::Containers::StringView filename) {
  // Clear any previous data
  doClose();

  std::string filenameStr{filename};
  std::ifstream file(filenameStr, std::ios::binary);
  
  if (!file.is_open()) {
    ESP_ERROR() << "Failed to open 3DGS PLY file:" << filenameStr;
    return;
  }

  ESP_DEBUG() << "Opening 3DGS PLY file:" << filenameStr;

  // Parse PLY header
  if (!parsePLYHeader(file)) {
    ESP_ERROR() << "Failed to parse PLY header from:" << filenameStr;
    file.close();
    return;
  }

  // Read binary data
  if (!readBinaryData(file)) {
    ESP_ERROR() << "Failed to read binary data from:" << filenameStr;
    file.close();
    return;
  }

  file.close();
  opened_ = true;

  ESP_DEBUG() << "Successfully loaded" << gaussianCount_ 
              << "Gaussians from 3DGS PLY file";
}

void GaussianSplattingImporter::doClose() {
  opened_ = false;
  gaussianCount_ = 0;
  shRestCount_ = 0;
  positions_.clear();
  normals_.clear();
  sh_dc_.clear();
  sh_rest_.clear();
  opacities_.clear();
  scales_.clear();
  rotations_.clear();
}

Mn::UnsignedInt GaussianSplattingImporter::doMeshCount() const {
  return opened_ ? 1 : 0;
}

Corrade::Containers::Optional<Mn::Trade::MeshData>
GaussianSplattingImporter::doMesh(Mn::UnsignedInt id, Mn::UnsignedInt level) {
  if (!opened_ || id != 0) {
    ESP_ERROR() << "Invalid mesh request: opened=" << opened_ << ", id=" << id;
    return Corrade::Containers::NullOpt;
  }

  // Convert Gaussian positions to MeshData for compatibility
  // This creates a point cloud representation
  Corrade::Containers::Array<Magnum::Vector3> positionData(gaussianCount_);
  for (size_t i = 0; i < gaussianCount_; ++i) {
    positionData[i] = positions_[i];
  }

  // Create MeshData with positions only (as a point cloud)
  return Mn::Trade::MeshData{
      Mn::MeshPrimitive::Points,
      {},
      {Mn::Trade::MeshAttributeData{Mn::Trade::MeshAttribute::Position,
                                     Cr::Containers::arrayView(positionData)}},
      static_cast<Mn::UnsignedInt>(gaussianCount_)};
}

bool GaussianSplattingImporter::parsePLYHeader(std::ifstream& file) {
  std::string line;

  // Read magic number
  std::getline(file, line);
  if (line != "ply") {
    ESP_ERROR() << "Not a valid PLY file (missing 'ply' header)";
    return false;
  }

  // Read format
  std::getline(file, line);
  if (line.find("format binary_little_endian") == std::string::npos) {
    ESP_ERROR() << "Only binary_little_endian format is supported";
    return false;
  }

  // Parse header to find vertex count and properties
  bool foundVertexElement = false;
  shRestCount_ = 0;

  while (std::getline(file, line)) {
    if (line == "end_header") {
      break;
    }

    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;

    if (keyword == "element") {
      std::string elementType;
      iss >> elementType;
      if (elementType == "vertex") {
        iss >> gaussianCount_;
        foundVertexElement = true;
        ESP_DEBUG() << "Found" << gaussianCount_ << "vertices in PLY";
      }
    } else if (keyword == "property" && foundVertexElement) {
      std::string propType, propName;
      iss >> propType >> propName;
      
      // Count f_rest properties
      if (propName.find("f_rest_") == 0) {
        shRestCount_++;
      }
    }
  }

  if (!foundVertexElement || gaussianCount_ == 0) {
    ESP_ERROR() << "Invalid PLY header: no vertex element found";
    return false;
  }

  ESP_DEBUG() << "PLY header parsed: " << gaussianCount_ 
              << " Gaussians with " << shRestCount_ << " SH rest coefficients";

  return true;
}

float GaussianSplattingImporter::bytesToFloat(const unsigned char* bytes) const {
  float value;
  std::memcpy(&value, bytes, sizeof(float));
  return value;
}

bool GaussianSplattingImporter::readBinaryData(std::ifstream& file) {
  // Reserve space
  positions_.reserve(gaussianCount_);
  normals_.reserve(gaussianCount_);
  sh_dc_.reserve(gaussianCount_);
  sh_rest_.reserve(gaussianCount_);
  opacities_.reserve(gaussianCount_);
  scales_.reserve(gaussianCount_);
  rotations_.reserve(gaussianCount_);

  // Each vertex has:
  // - position (x, y, z): 3 floats = 12 bytes
  // - normal (nx, ny, nz): 3 floats = 12 bytes
  // - f_dc (f_dc_0, f_dc_1, f_dc_2): 3 floats = 12 bytes
  // - f_rest (f_rest_0 to f_rest_44): shRestCount_ floats
  // - opacity: 1 float = 4 bytes
  // - scale (scale_0, scale_1, scale_2): 3 floats = 12 bytes
  // - rotation (rot_0, rot_1, rot_2, rot_3): 4 floats = 16 bytes

  const size_t bytesPerVertex = 3 * 4 +  // position
                                3 * 4 +  // normal
                                3 * 4 +  // f_dc
                                shRestCount_ * 4 +  // f_rest
                                1 * 4 +  // opacity
                                3 * 4 +  // scale
                                4 * 4;   // rotation

  std::vector<unsigned char> buffer(bytesPerVertex);

  for (size_t i = 0; i < gaussianCount_; ++i) {
    file.read(reinterpret_cast<char*>(buffer.data()), bytesPerVertex);
    
    if (!file) {
      ESP_ERROR() << "Failed to read Gaussian" << i << "of" << gaussianCount_;
      return false;
    }

    size_t offset = 0;

    // Read position
    float x = bytesToFloat(&buffer[offset]); offset += 4;
    float y = bytesToFloat(&buffer[offset]); offset += 4;
    float z = bytesToFloat(&buffer[offset]); offset += 4;
    positions_.emplace_back(x, y, z);

    // Read normal
    float nx = bytesToFloat(&buffer[offset]); offset += 4;
    float ny = bytesToFloat(&buffer[offset]); offset += 4;
    float nz = bytesToFloat(&buffer[offset]); offset += 4;
    normals_.emplace_back(nx, ny, nz);

    // Read f_dc (SH DC coefficients)
    float f_dc_0 = bytesToFloat(&buffer[offset]); offset += 4;
    float f_dc_1 = bytesToFloat(&buffer[offset]); offset += 4;
    float f_dc_2 = bytesToFloat(&buffer[offset]); offset += 4;
    sh_dc_.emplace_back(f_dc_0, f_dc_1, f_dc_2);

    // Read f_rest (SH higher-order coefficients)
    std::vector<float> f_rest;
    f_rest.reserve(shRestCount_);
    for (size_t j = 0; j < shRestCount_; ++j) {
      f_rest.push_back(bytesToFloat(&buffer[offset]));
      offset += 4;
    }
    sh_rest_.push_back(std::move(f_rest));

    // Read opacity
    float opacity = bytesToFloat(&buffer[offset]); offset += 4;
    opacities_.push_back(opacity);

    // Read scale
    float scale_0 = bytesToFloat(&buffer[offset]); offset += 4;
    float scale_1 = bytesToFloat(&buffer[offset]); offset += 4;
    float scale_2 = bytesToFloat(&buffer[offset]); offset += 4;
    scales_.emplace_back(scale_0, scale_1, scale_2);

    // Read rotation (quaternion)
    float rot_0 = bytesToFloat(&buffer[offset]); offset += 4;
    float rot_1 = bytesToFloat(&buffer[offset]); offset += 4;
    float rot_2 = bytesToFloat(&buffer[offset]); offset += 4;
    float rot_3 = bytesToFloat(&buffer[offset]); offset += 4;
    rotations_.emplace_back(Magnum::Vector3(rot_1, rot_2, rot_3), rot_0);
  }

  ESP_DEBUG() << "Successfully read" << gaussianCount_ << "Gaussians";
  return true;
}

}  // namespace assets
}  // namespace esp

