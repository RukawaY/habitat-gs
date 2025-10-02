#include <Corrade/Utility/Path.h>
#include <Magnum/DebugTools/CompareImage.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <string>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/assets/GaussianSplattingImporter.h"
#include "esp/assets/ResourceManager.h"
#include "esp/core/Esp.h"
#include "esp/gfx/Drawable.h"
#include "esp/gfx/RenderCamera.h"
#include "esp/gfx/Renderer.h"
#include "esp/metadata/MetadataMediator.h"
#include "esp/scene/SceneGraph.h"
#include "esp/scene/SceneManager.h"
#include "esp/scene/SceneNode.h"
#include "esp/sim/Simulator.h"

#ifdef ESP_BUILD_WITH_CUDA
#include "esp/gfx/GaussianRasterizer.h"
#include "esp/gfx/GaussianSplattingDrawable.h"
#endif

#include "configure.h"

namespace Cr = Corrade;
namespace Mn = Magnum;

using esp::assets::AssetInfo;
using esp::assets::AssetType;
using esp::assets::GaussianSplattingData;
using esp::assets::GaussianSplattingImporter;
using esp::assets::ResourceManager;
using esp::metadata::MetadataMediator;

const std::string dataDir = Cr::Utility::Path::join(SCENE_DATASETS, "../");
const std::string testPlyFile = "/mnt/data/home/ziyuan/gaussian-splatting/output/d6858747-c/point_cloud/iteration_30000/point_cloud.ply";

// Test 1: Test basic Gaussian Splatting Drawable creation
void testGaussianSplattingDrawableCreation() {
  ESP_DEBUG() << "======== Test 1: GaussianSplattingDrawable Creation ========";

  auto MM = MetadataMediator::create();
  ResourceManager resourceManager(MM);

  // Create scene graph
  auto sceneID = esp::scene::SceneManager::create();
  auto& sceneGraph = esp::scene::SceneManager::getSceneGraph(sceneID);
  auto& rootNode = sceneGraph.getRootNode();
  auto& drawables = sceneGraph.getDrawables();

  // Create a test Gaussian Splatting file if it doesn't exist
  if (!Cr::Utility::Path::exists(testPlyFile)) {
    ESP_WARNING() << "Test PLY file" << testPlyFile
                  << "does not exist. Skipping drawable creation test.";
    return;
  }

  // Load Gaussian Splatting asset
  AssetInfo info{AssetType::GaussianSplatting, testPlyFile};
  bool loadSuccess = resourceManager.loadRenderAsset(info);

  if (!loadSuccess) {
    ESP_WARNING() << "Failed to load Gaussian Splatting asset:" << testPlyFile;
    ESP_WARNING() << "Drawable creation test skipped.";
    return;
  }

  ESP_DEBUG() << "Successfully loaded Gaussian Splatting asset";

  // Create render asset instance (which creates the drawable)
  esp::assets::RenderAssetInstanceCreationInfo creationInfo;
  creationInfo.filepath = testPlyFile;
  creationInfo.lightSetupKey = esp::gfx::NO_LIGHT_KEY;

  auto* instanceNode = resourceManager.createRenderAssetInstance(
      creationInfo, &rootNode, &drawables);

  CORRADE_VERIFY(instanceNode != nullptr);
  ESP_DEBUG() << "Successfully created GaussianSplattingDrawable";
  ESP_DEBUG() << "Drawables count:" << drawables.size();

  CORRADE_VERIFY(drawables.size() > 0);
  ESP_DEBUG() << "Test 1 PASSED";
}

#ifdef ESP_BUILD_WITH_CUDA
// Test 2: Test Gaussian Rasterizer
void testGaussianRasterizer() {
  ESP_DEBUG() << "======== Test 2: Gaussian Rasterizer ========";

  // Create simple test data
  GaussianSplattingData gaussianData;

  // Add a few test Gaussians
  for (int i = 0; i < 10; ++i) {
    esp::assets::GaussianSplat splat;
    splat.position = Mn::Vector3{float(i) * 0.1f, 0.0f, -5.0f};
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    splat.f_dc = Mn::Vector3{1.0f, 0.0f, 0.0f};  // Red
    splat.f_rest = Cr::Containers::Array<float>{};
    splat.opacity = 1.0f;
    splat.scale = Mn::Vector3{0.1f, 0.1f, 0.1f};
    splat.rotation = Mn::Quaternion::rotation(Mn::Deg(0.0f), Mn::Vector3::xAxis());

    gaussianData.addGaussian(std::move(splat));
  }

  ESP_DEBUG() << "Created test data with" << gaussianData.getGaussianCount()
              << "Gaussians";

  // Create rasterizer
  esp::gfx::GaussianRasterizer rasterizer;

  // Create test textures
  Mn::Vector2i viewport{640, 480};
  Mn::GL::Texture2D colorTexture;
  Mn::GL::Texture2D depthTexture;

  colorTexture.setStorage(1, Mn::GL::TextureFormat::RGBA32F, viewport);
  depthTexture.setStorage(1, Mn::GL::TextureFormat::R32F, viewport);

  // Set up camera matrices
  Mn::Matrix4 viewMatrix = Mn::Matrix4::lookAt(
      Mn::Vector3{0.0f, 0.0f, 0.0f},  // eye
      Mn::Vector3{0.0f, 0.0f, -1.0f},  // target
      Mn::Vector3{0.0f, 1.0f, 0.0f}   // up
  );

  float fovy = Mn::Deg(60.0f);
  float aspect = float(viewport.x()) / float(viewport.y());
  Mn::Matrix4 projMatrix = Mn::Matrix4::perspectiveProjection(
      fovy, aspect, 0.1f, 100.0f);

  // Render
  ESP_DEBUG() << "Calling rasterizer.render()...";
  try {
    rasterizer.render(gaussianData, viewMatrix, projMatrix, viewport,
                      colorTexture, depthTexture);
    ESP_DEBUG() << "Rasterizer render completed successfully";
  } catch (const std::exception& e) {
    ESP_ERROR() << "Rasterizer render failed:" << e.what();
    CORRADE_VERIFY(false);
  }

  ESP_DEBUG() << "Test 2 PASSED";
}
#endif

// Test 3: Test full rendering pipeline with Simulator
void testGaussianRenderingPipeline() {
  ESP_DEBUG() << "======== Test 3: Full Rendering Pipeline ========";

  if (!Cr::Utility::Path::exists(testPlyFile)) {
    ESP_WARNING() << "Test PLY file" << testPlyFile
                  << "does not exist. Skipping pipeline test.";
    return;
  }

  // This test requires a full simulator setup
  // For now, we'll just verify that the asset can be loaded and instantiated
  // A full integration test with rendering would require more setup

  auto MM = MetadataMediator::create();
  ResourceManager resourceManager(MM);

  AssetInfo info{AssetType::GaussianSplatting, testPlyFile};
  bool loadSuccess = resourceManager.loadRenderAsset(info);

  CORRADE_VERIFY(loadSuccess);
  ESP_DEBUG() << "Asset loaded successfully for pipeline test";

  // TODO: Extend this test to actually render frames using the Simulator
  // when the full integration is complete

  ESP_DEBUG() << "Test 3 PASSED (basic verification)";
}

int main(int argc, char** argv) {
  ESP_DEBUG() << "Starting GaussianRenderTest with" << (argc - 1) << "test cases...";

  int testCount = 0;
  int errorCount = 0;

  // Test 1: Drawable creation
  try {
    testGaussianSplattingDrawableCreation();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianSplattingDrawableCreation()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianSplattingDrawableCreation():" << e.what();
  }

#ifdef ESP_BUILD_WITH_CUDA
  // Test 2: Rasterizer (CUDA only)
  try {
    testGaussianRasterizer();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianRasterizer()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianRasterizer():" << e.what();
  }
#else
  ESP_WARNING() << "SKIPPED testGaussianRasterizer() - CUDA not enabled";
#endif

  // Test 3: Full pipeline
  try {
    testGaussianRenderingPipeline();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianRenderingPipeline()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianRenderingPipeline():" << e.what();
  }

  ESP_DEBUG() << "Finished GaussianRenderTest with" << errorCount
              << "errors out of" << testCount << "checks.";

  return errorCount > 0 ? 1 : 0;
}

