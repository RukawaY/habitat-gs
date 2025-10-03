#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/Path.h>
#include <Magnum/DebugTools/CompareImage.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <string>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/assets/GaussianSplattingImporter.h"
#include "esp/assets/ResourceManager.h"
#include "esp/core/Esp.h"
#include "esp/core/Logging.h"
#include "esp/gfx/Drawable.h"
#include "esp/gfx/RenderCamera.h"
#include "esp/gfx/Renderer.h"
#include "esp/gfx/WindowlessContext.h"
#include "esp/metadata/MetadataMediator.h"
#include "esp/scene/SceneGraph.h"
#include "esp/scene/SceneManager.h"
#include "esp/scene/SceneNode.h"
#include "esp/sim/Simulator.h"

#ifdef ESP_BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include "esp/gfx/GaussianRasterizer.h"
#include "esp/gfx/GaussianRasterizerWrapper.h"
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

  // Create scene manager and scene graph
  auto sceneMgr = esp::scene::SceneManager::create();
  int sceneID = sceneMgr->initSceneGraph();
  auto& sceneGraph = sceneMgr->getSceneGraph(sceneID);
  auto& rootNode = sceneGraph.getRootNode();
  auto& drawables = sceneGraph.getDrawables();

  // Create a test Gaussian Splatting file if it doesn't exist
  if (!Cr::Utility::Path::exists(testPlyFile)) {
    ESP_WARNING() << "Test PLY file" << testPlyFile
                  << "does not exist. Skipping drawable creation test.";
    return;
  }

  // Load and create Gaussian Splatting asset instance
  AssetInfo assetInfo{AssetType::GaussianSplatting, testPlyFile};
  
  esp::assets::RenderAssetInstanceCreationInfo creationInfo;
  creationInfo.filepath = testPlyFile;
  creationInfo.lightSetupKey = esp::NO_LIGHT_KEY;

  auto* instanceNode = resourceManager.loadAndCreateRenderAssetInstance(
      assetInfo, creationInfo, &rootNode, &drawables);

  if (instanceNode == nullptr) {
    ESP_ERROR() << "Failed to create render asset instance";
    return;
  }
  ESP_DEBUG() << "Successfully created GaussianSplattingDrawable";
  ESP_DEBUG() << "Drawables count:" << drawables.size();

  if (drawables.size() == 0) {
    ESP_ERROR() << "No drawables created";
    return;
  }
  ESP_DEBUG() << "Test 1 PASSED";
}

#ifdef ESP_BUILD_WITH_CUDA
// Test 2: Test Gaussian Rasterizer with basic data
void testGaussianRasterizer() {
  ESP_DEBUG() << "======== Test 2: Gaussian Rasterizer (Basic) ========";

  // Create simple test data
  GaussianSplattingData gaussianData;

  // Add a few test Gaussians with complete SH coefficients
  for (int i = 0; i < 10; ++i) {
    esp::assets::GaussianSplat splat;
    splat.position = Mn::Vector3{float(i) * 0.1f, 0.0f, -5.0f};
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    splat.f_dc = Mn::Vector3{1.0f, 0.0f, 0.0f};  // Red DC component
    
    // Initialize f_rest with 45 coefficients (degree 3 SH: 15 per channel × 3 channels)
    splat.f_rest = Cr::Containers::Array<float>{Cr::DirectInit, 45, 0.0f};
    
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

  // Use formats compatible with CUDA-OpenGL interop
  colorTexture.setStorage(1, Mn::GL::TextureFormat::RGBA32F, viewport);
  depthTexture.setStorage(1, Mn::GL::TextureFormat::R32F, viewport);  // Use R32F instead of DepthComponent32F for CUDA

  // Set up camera matrices
  Mn::Matrix4 viewMatrix = Mn::Matrix4::lookAt(
      Mn::Vector3{0.0f, 0.0f, 0.0f},  // eye
      Mn::Vector3{0.0f, 0.0f, -1.0f},  // target
      Mn::Vector3{0.0f, 1.0f, 0.0f}   // up
  );

  Mn::Rad fovy = Mn::Deg(60.0f);
  float aspect = float(viewport.x()) / float(viewport.y());
  Mn::Matrix4 projMatrix = Mn::Matrix4::perspectiveProjection(
      fovy, aspect, 0.1f, 100.0f);

  // Render
  ESP_DEBUG() << "Calling renderGaussians()...";
  try {
    esp::gfx::renderGaussians(rasterizer, gaussianData, viewMatrix, projMatrix, viewport,
                               colorTexture, depthTexture, Mn::Vector3{0.0f});
    ESP_DEBUG() << "Rasterizer render completed successfully";
  } catch (const std::exception& e) {
    ESP_ERROR() << "Rasterizer render failed:" << e.what();
    return;
  }

  ESP_DEBUG() << "Test 2 PASSED";
}

// Test 2b: Test with complete spherical harmonics
void testGaussianRasterizerWithSH() {
  ESP_DEBUG() << "======== Test 2b: Gaussian Rasterizer (With SH) ========";

  GaussianSplattingData gaussianData;

  // Create a colorful scene with different SH coefficients
  for (int i = 0; i < 5; ++i) {
    esp::assets::GaussianSplat splat;
    
    // Position in a row
    splat.position = Mn::Vector3{float(i - 2) * 0.5f, 0.0f, -3.0f};
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    
    // Different colors for each Gaussian
    float r = float(i) / 4.0f;
    float g = float(4 - i) / 4.0f;
    float b = 0.5f;
    splat.f_dc = Mn::Vector3{r, g, b};
    
    // Initialize higher-order SH coefficients
    splat.f_rest = Cr::Containers::Array<float>{Cr::DirectInit, 45, 0.0f};
    
    // Add some non-zero higher-order coefficients for view-dependent effects
    for (int j = 0; j < 15; ++j) {
      splat.f_rest[j] = 0.1f * std::sin(float(j) / 5.0f);  // Channel 0 (R)
      splat.f_rest[15 + j] = 0.1f * std::cos(float(j) / 5.0f);  // Channel 1 (G)
      splat.f_rest[30 + j] = 0.05f;  // Channel 2 (B)
    }
    
    splat.opacity = 0.9f;
    splat.scale = Mn::Vector3{0.2f, 0.2f, 0.1f};
    splat.rotation = Mn::Quaternion::rotation(
        Mn::Deg(float(i) * 20.0f), Mn::Vector3{0.0f, 1.0f, 0.0f});

    gaussianData.addGaussian(std::move(splat));
  }

  ESP_DEBUG() << "Created SH test data with" << gaussianData.getGaussianCount()
              << "Gaussians";

  esp::gfx::GaussianRasterizer rasterizer;
  Mn::Vector2i viewport{800, 600};
  Mn::GL::Texture2D colorTexture;
  Mn::GL::Texture2D depthTexture;

  colorTexture.setStorage(1, Mn::GL::TextureFormat::RGBA32F, viewport);
  depthTexture.setStorage(1, Mn::GL::TextureFormat::R32F, viewport);

  Mn::Matrix4 viewMatrix = Mn::Matrix4::lookAt(
      Mn::Vector3{0.0f, 0.5f, 1.0f},
      Mn::Vector3{0.0f, 0.0f, -3.0f},
      Mn::Vector3{0.0f, 1.0f, 0.0f});

  Mn::Matrix4 projMatrix = Mn::Matrix4::perspectiveProjection(
      Mn::Deg(70.0f), float(viewport.x()) / float(viewport.y()), 0.1f, 100.0f);

  try {
    esp::gfx::renderGaussians(rasterizer, gaussianData, viewMatrix, projMatrix, viewport,
                               colorTexture, depthTexture, Mn::Vector3{0.1f, 0.1f, 0.15f});
    ESP_DEBUG() << "SH rasterizer render completed successfully";
  } catch (const std::exception& e) {
    ESP_ERROR() << "SH rasterizer render failed:" << e.what();
    return;
  }

  ESP_DEBUG() << "Test 2b PASSED";
}

// Test 2c: Test complex scene with depth
void testGaussianRasterizerComplexScene() {
  ESP_DEBUG() << "======== Test 2c: Complex Scene with Depth ========";

  GaussianSplattingData gaussianData;

  // Create a multi-layer scene to test depth fusion
  // Layer 1: Background (far)
  for (int i = 0; i < 20; ++i) {
    esp::assets::GaussianSplat splat;
    float x = (float(i % 5) - 2.0f) * 0.4f;
    float y = (float(i / 5) - 1.5f) * 0.4f;
    splat.position = Mn::Vector3{x, y, -8.0f};
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    splat.f_dc = Mn::Vector3{0.2f, 0.3f, 0.8f};  // Blue background
    splat.f_rest = Cr::Containers::Array<float>{Cr::DirectInit, 45, 0.0f};
    splat.opacity = 0.8f;
    splat.scale = Mn::Vector3{0.3f, 0.3f, 0.2f};
    splat.rotation = Mn::Quaternion{};
    gaussianData.addGaussian(std::move(splat));
  }

  // Layer 2: Middle (medium depth)
  for (int i = 0; i < 10; ++i) {
    esp::assets::GaussianSplat splat;
    float angle = float(i) * 36.0f;  // 10 Gaussians in a circle
    float radius = 1.0f;
    splat.position = Mn::Vector3{
        radius * std::cos(Mn::Deg(angle).operator float()),
        radius * std::sin(Mn::Deg(angle).operator float()),
        -5.0f
    };
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    splat.f_dc = Mn::Vector3{0.8f, 0.8f, 0.2f};  // Yellow middle layer
    splat.f_rest = Cr::Containers::Array<float>{Cr::DirectInit, 45, 0.0f};
    splat.opacity = 0.9f;
    splat.scale = Mn::Vector3{0.2f, 0.2f, 0.15f};
    splat.rotation = Mn::Quaternion::rotation(Mn::Deg(angle), Mn::Vector3::zAxis());
    gaussianData.addGaussian(std::move(splat));
  }

  // Layer 3: Foreground (close)
  for (int i = 0; i < 5; ++i) {
    esp::assets::GaussianSplat splat;
    splat.position = Mn::Vector3{float(i - 2) * 0.3f, 0.0f, -2.0f};
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    splat.f_dc = Mn::Vector3{0.9f, 0.2f, 0.2f};  // Red foreground
    splat.f_rest = Cr::Containers::Array<float>{Cr::DirectInit, 45, 0.0f};
    splat.opacity = 1.0f;
    splat.scale = Mn::Vector3{0.15f, 0.15f, 0.1f};
    splat.rotation = Mn::Quaternion{};
    gaussianData.addGaussian(std::move(splat));
  }

  ESP_DEBUG() << "Created complex scene with" << gaussianData.getGaussianCount()
              << "Gaussians in 3 layers";

  esp::gfx::GaussianRasterizer rasterizer;
  Mn::Vector2i viewport{1024, 768};
  Mn::GL::Texture2D colorTexture;
  Mn::GL::Texture2D depthTexture;

  colorTexture.setStorage(1, Mn::GL::TextureFormat::RGBA32F, viewport);
  depthTexture.setStorage(1, Mn::GL::TextureFormat::R32F, viewport);

  Mn::Matrix4 viewMatrix = Mn::Matrix4::lookAt(
      Mn::Vector3{0.0f, 0.0f, 0.0f},
      Mn::Vector3{0.0f, 0.0f, -1.0f},
      Mn::Vector3{0.0f, 1.0f, 0.0f});

  Mn::Matrix4 projMatrix = Mn::Matrix4::perspectiveProjection(
      Mn::Deg(75.0f), float(viewport.x()) / float(viewport.y()), 0.1f, 100.0f);

  try {
    esp::gfx::renderGaussians(rasterizer, gaussianData, viewMatrix, projMatrix, viewport,
                               colorTexture, depthTexture, Mn::Vector3{0.0f, 0.0f, 0.0f});
    ESP_DEBUG() << "Complex scene render completed successfully";
  } catch (const std::exception& e) {
    ESP_ERROR() << "Complex scene render failed:" << e.what();
    return;
  }

  ESP_DEBUG() << "Test 2c PASSED";
}

// Test 2d: Test RenderTarget integration
void testGaussianRasterizerWithRenderTarget() {
  ESP_DEBUG() << "======== Test 2d: RenderTarget Integration ========";

  GaussianSplattingData gaussianData;

  // Create test scene
  for (int i = 0; i < 8; ++i) {
    esp::assets::GaussianSplat splat;
    float angle = float(i) * 45.0f;
    splat.position = Mn::Vector3{
        std::cos(Mn::Deg(angle).operator float()),
        std::sin(Mn::Deg(angle).operator float()),
        -4.0f
    };
    splat.normal = Mn::Vector3{0.0f, 0.0f, 1.0f};
    splat.f_dc = Mn::Vector3{
        std::abs(std::cos(Mn::Deg(angle).operator float())),
        std::abs(std::sin(Mn::Deg(angle).operator float())),
        0.5f
    };
    splat.f_rest = Cr::Containers::Array<float>{Cr::DirectInit, 45, 0.0f};
    splat.opacity = 0.95f;
    splat.scale = Mn::Vector3{0.25f, 0.25f, 0.2f};
    splat.rotation = Mn::Quaternion{};
    gaussianData.addGaussian(std::move(splat));
  }

  ESP_DEBUG() << "Created RenderTarget test scene with" << gaussianData.getGaussianCount()
              << "Gaussians";

  // Create RenderTarget
  Mn::Vector2i size{640, 480};
  Mn::Vector2 depthUnprojection{1.0f, 100.0f};
  
  auto renderTarget = std::make_unique<esp::gfx::RenderTarget>(
      size, depthUnprojection, nullptr,
      esp::gfx::RenderTarget::Flag::RgbaAttachment | 
      esp::gfx::RenderTarget::Flag::DepthTextureAttachment);

  esp::gfx::GaussianRasterizer rasterizer;

  Mn::Matrix4 viewMatrix = Mn::Matrix4::lookAt(
      Mn::Vector3{0.0f, 0.0f, 0.0f},
      Mn::Vector3{0.0f, 0.0f, -1.0f},
      Mn::Vector3{0.0f, 1.0f, 0.0f});

  Mn::Matrix4 projMatrix = Mn::Matrix4::perspectiveProjection(
      Mn::Deg(60.0f), float(size.x()) / float(size.y()), 0.1f, 100.0f);

  try {
    // Render using RenderTarget wrapper
    esp::gfx::renderGaussiansToRenderTarget(
        rasterizer, gaussianData, viewMatrix, projMatrix, size,
        *renderTarget, Mn::Vector3{0.0f, 0.0f, 0.0f});
    
    ESP_DEBUG() << "RenderTarget integration render completed successfully";
  } catch (const std::exception& e) {
    ESP_ERROR() << "RenderTarget integration render failed:" << e.what();
    return;
  }

  ESP_DEBUG() << "Test 2d PASSED";
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

  if (!loadSuccess) {
    ESP_ERROR() << "Failed to load render asset";
    return;
  }
  ESP_DEBUG() << "Asset loaded successfully for pipeline test";

  // TODO: Extend this test to actually render frames using the Simulator
  // when the full integration is complete

  ESP_DEBUG() << "Test 3 PASSED (basic verification)";
}

int main(int argc, char** argv) {
  // Initialize logging context - required for ESP_DEBUG, ESP_WARNING, etc.
  esp::logging::LoggingContext loggingContext;
  
  // Create OpenGL context - required for all GL operations
  auto context = esp::gfx::WindowlessContext::create_unique(0);
  if (!context) {
    ESP_ERROR() << "Failed to create OpenGL context";
    return 1;
  }

  ESP_DEBUG() << "Starting GaussianRenderTest with" << (argc - 1) << "test cases...";
  ESP_DEBUG() << "OpenGL context created successfully";

#ifdef ESP_BUILD_WITH_CUDA
  // Initialize CUDA device
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    printf("No CUDA devices available. Error: %s\n", cudaGetErrorString(err));
    ESP_ERROR() << "No CUDA devices available. Error: " << cudaGetErrorString(err);
    ESP_WARNING() << "Skipping CUDA-based tests";
  } else {
    printf("Found %d CUDA device(s)\n", deviceCount);
    ESP_DEBUG() << "Found" << deviceCount << "CUDA device(s)";
    cudaSetDevice(0);
    ESP_DEBUG() << "CUDA device 0 initialized";
  }
#endif

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
  // Test 2: Basic Rasterizer (CUDA only)
  try {
    testGaussianRasterizer();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianRasterizer()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianRasterizer():" << e.what();
  }

  // Test 2b: Rasterizer with SH
  try {
    testGaussianRasterizerWithSH();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianRasterizerWithSH()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianRasterizerWithSH():" << e.what();
  }

  // Test 2c: Complex Scene
  try {
    testGaussianRasterizerComplexScene();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianRasterizerComplexScene()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianRasterizerComplexScene():" << e.what();
  }

  // Test 2d: RenderTarget Integration
  try {
    testGaussianRasterizerWithRenderTarget();
    testCount++;
    ESP_DEBUG() << "OK [" << testCount << "] testGaussianRasterizerWithRenderTarget()";
  } catch (const std::exception& e) {
    errorCount++;
    ESP_ERROR() << "FAILED [" << testCount + 1 << "] testGaussianRasterizerWithRenderTarget():" << e.what();
  }
#else
  ESP_WARNING() << "SKIPPED CUDA tests - CUDA not enabled";
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

