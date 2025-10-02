// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/Path.h>
#include <string>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/assets/GaussianSplattingImporter.h"
#include "esp/assets/ResourceManager.h"
#include "esp/metadata/MetadataMediator.h"

#include "configure.h"

namespace Cr = Corrade;

using esp::assets::AssetInfo;
using esp::assets::AssetType;
using esp::assets::GaussianSplat;
using esp::assets::GaussianSplattingData;
using esp::assets::GaussianSplattingImporter;
using esp::assets::ResourceManager;
using esp::metadata::MetadataMediator;

namespace {

/**
 * @brief Test for 3D Gaussian Splatting import functionality
 */
struct GaussianImportTest : Cr::TestSuite::Tester {
  explicit GaussianImportTest();

  void testGaussianSplattingImporter();
  void testGaussianSplattingData();
  void testGaussianSplattingResourceManager();

  esp::logging::LoggingContext loggingContext_;
};

GaussianImportTest::GaussianImportTest() {
  addTests({&GaussianImportTest::testGaussianSplattingImporter,
            &GaussianImportTest::testGaussianSplattingData,
            &GaussianImportTest::testGaussianSplattingResourceManager});
}

void GaussianImportTest::testGaussianSplattingImporter() {
  ESP_DEBUG() << "=== Testing GaussianSplattingImporter ===";

  GaussianSplattingImporter importer;

  // Test that importer is initially closed
  CORRADE_VERIFY(!importer.doIsOpened());
  CORRADE_VERIFY(importer.doMeshCount() == 0);

  // Note: To actually test with a real file, you would need to:
  // 1. Provide a path to a valid 3DGS PLY file
  // 2. Uncomment and update the following code:
  
  
  std::string testFile = "/mnt/data/home/ziyuan/gaussian-splatting/output/d6858747-c/point_cloud/iteration_30000/point_cloud.ply";
  
  if (Cr::Utility::Path::exists(testFile)) {
    importer.doOpenFile(testFile);
    
    CORRADE_VERIFY(importer.doIsOpened());
    CORRADE_VERIFY(importer.doMeshCount() == 1);
    CORRADE_VERIFY(importer.getGaussianCount() > 0);
    
    ESP_DEBUG() << "Loaded" << importer.getGaussianCount() << "Gaussians";
    ESP_DEBUG() << "SH rest count:" << importer.getSHRest()[0].size();
    
    // Verify data integrity
    const auto& positions = importer.getPositions();
    const auto& normals = importer.getNormals();
    const auto& sh_dc = importer.getSHDC();
    
    CORRADE_VERIFY(positions.size() == importer.getGaussianCount());
    CORRADE_VERIFY(normals.size() == importer.getGaussianCount());
    CORRADE_VERIFY(sh_dc.size() == importer.getGaussianCount());
  } else {
    ESP_WARNING() << "Test 3DGS file not found, skipping file-based tests";
  }
  

  ESP_DEBUG() << "GaussianSplattingImporter test completed";
}

void GaussianImportTest::testGaussianSplattingData() {
  ESP_DEBUG() << "=== Testing GaussianSplattingData ===";

  GaussianSplattingData gsData;

  // Verify initial state
  CORRADE_VERIFY(gsData.getGaussianCount() == 0);
  CORRADE_VERIFY(gsData.getMeshType() ==
                 esp::assets::SupportedMeshType::GAUSSIAN_SPLATTING);

  // Create and add a test Gaussian
  GaussianSplat testSplat;
  testSplat.position = Magnum::Vector3{1.0f, 2.0f, 3.0f};
  testSplat.normal = Magnum::Vector3{0.0f, 1.0f, 0.0f};
  testSplat.f_dc = Magnum::Vector3{0.5f, 0.5f, 0.5f};
  testSplat.opacity = 0.8f;
  testSplat.scale = Magnum::Vector3{0.1f, 0.1f, 0.1f};
  testSplat.rotation = Magnum::Quaternion{{0.0f, 0.0f, 0.0f}, 1.0f};
  
  // Add some SH rest coefficients (45 for degree 3)
  testSplat.f_rest = Corrade::Containers::Array<float>(45);
  for (size_t i = 0; i < 45; ++i) {
    testSplat.f_rest[i] = 0.0f;
  }

  gsData.addGaussian(std::move(testSplat));

  // Verify Gaussian was added
  CORRADE_VERIFY(gsData.getGaussianCount() == 1);
  const auto& gaussians = gsData.getGaussians();
  CORRADE_VERIFY(gaussians.size() == 1);
  CORRADE_VERIFY(gaussians[0].position == testSplat.position);
  CORRADE_VERIFY(gaussians[0].opacity == testSplat.opacity);

  // Test clear
  gsData.clear();
  CORRADE_VERIFY(gsData.getGaussianCount() == 0);

  ESP_DEBUG() << "GaussianSplattingData test completed";
}

void GaussianImportTest::testGaussianSplattingResourceManager() {
  ESP_DEBUG() << "=== Testing ResourceManager with 3DGS ===";

  // Create MetadataMediator
  auto cfg = esp::sim::SimulatorConfiguration{};
  cfg.createRenderer = false;  // Don't need renderer for basic import test
  auto MM = MetadataMediator::create(cfg);

  // Create ResourceManager
  ResourceManager resourceManager(MM);

  // Note: To actually test loading a 3DGS file with ResourceManager:
  // 1. Provide a path to a valid 3DGS PLY file
  // 2. Uncomment and update the following code:
  
  
  std::string testFile = "/mnt/data/home/ziyuan/gaussian-splatting/output/d6858747-c/point_cloud/iteration_30000/point_cloud.ply";
  
  if (Cr::Utility::Path::exists(testFile)) {
    AssetInfo info{AssetType::GaussianSplatting, testFile};
    
    bool success = resourceManager.loadRenderAsset(info);
    CORRADE_VERIFY(success);
    
    // Verify the asset was loaded
    const auto& metadata = resourceManager.getMeshMetaData(testFile);
    CORRADE_VERIFY(metadata.meshIndex.first >= 0);
    
    ESP_DEBUG() << "Successfully loaded 3DGS asset through ResourceManager";
  } else {
    ESP_WARNING() << "Test 3DGS file not found, skipping ResourceManager test";
  }
  

  ESP_DEBUG() << "ResourceManager 3DGS test completed";
}

}  // namespace

CORRADE_TEST_MAIN(GaussianImportTest)

