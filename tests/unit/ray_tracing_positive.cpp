/*
 * Copyright (c) 2015-2023 The Khronos Group Inc.
 * Copyright (c) 2015-2023 Valve Corporation
 * Copyright (c) 2015-2023 LunarG, Inc.
 * Copyright (c) 2015-2023 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "../framework/layer_validation_tests.h"
#include "../framework/ray_tracing_objects.h"
#include "generated/vk_extension_helper.h"
#include "../layers/utils/vk_layer_utils.h"

bool RayTracingTest::InitFrameworkForRayTracingTest(VkRenderFramework *framework, bool is_khr,
                                                    VkPhysicalDeviceFeatures2KHR *features2,
                                                    VkValidationFeaturesEXT *enabled_features) {
    framework->AddRequiredExtensions(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    framework->AddRequiredExtensions(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    if (is_khr) {
        framework->AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        framework->AddRequiredExtensions(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    } else {
        framework->AddRequiredExtensions(VK_NV_RAY_TRACING_EXTENSION_NAME);
    }

    framework->InitFramework(&framework->Monitor(), enabled_features);
    if (!framework->AreRequiredExtensionsEnabled()) {
        printf("%s device extension not supported, skipping test\n", framework->RequiredExtensionsNotSupported().c_str());
        return false;
    }

    if (features2) {
        // extension enabled as dependency of RT extension
        auto vkGetPhysicalDeviceFeatures2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
            vk::GetInstanceProcAddr(framework->instance(), "vkGetPhysicalDeviceFeatures2KHR"));
        assert(vkGetPhysicalDeviceFeatures2KHR);
        vkGetPhysicalDeviceFeatures2KHR(framework->gpu(), features2);
    }
    return true;
}

TEST_F(PositiveRayTracing, GetAccelerationStructureBuildSizes) {
    TEST_DESCRIPTION("Test enabled features for GetAccelerationStructureBuildSizes");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

    ASSERT_NO_FATAL_FAILURE(InitFramework(m_errorMonitor));

    if (DeviceValidationVersion() < VK_API_VERSION_1_1) {
        GTEST_SKIP() << "At least Vulkan version 1.1 is required";
    }

    // Crashes without any warnings
    if (IsDriver(VK_DRIVER_ID_AMD_PROPRIETARY)) {
        GTEST_SKIP() << "Test does not run on AMD proprietary driver";
    }

    if (!AreRequiredExtensionsEnabled()) {
        GTEST_SKIP() << RequiredExtensionsNotSupported() << " not supported";
    }

    auto accel_struct_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
    GetPhysicalDeviceFeatures2(accel_struct_features);
    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &accel_struct_features));

    auto build_info = vku::InitStruct<VkAccelerationStructureBuildGeometryInfoKHR>();
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    uint32_t max_primitives_count;
    auto build_sizes_info = vku::InitStruct<VkAccelerationStructureBuildSizesInfoKHR>();
    vk::GetAccelerationStructureBuildSizesKHR(device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR, &build_info,
                                              &max_primitives_count, &build_sizes_info);
}

TEST_F(PositiveRayTracing, AccelerationStructureReference) {
    TEST_DESCRIPTION("Test device side accelerationStructureReference");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    ASSERT_NO_FATAL_FAILURE(InitFramework());
    if (DeviceValidationVersion() < VK_API_VERSION_1_1) {
        GTEST_SKIP() << "At least Vulkan version 1.1 is required";
    }
    if (!AreRequiredExtensionsEnabled()) {
        GTEST_SKIP() << RequiredExtensionsNotSupported() << " not supported";
    }

    auto bda_features = vku::InitStruct<VkPhysicalDeviceBufferDeviceAddressFeatures>();
    auto ray_query_features = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>(&bda_features);
    auto acc_structure_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(&ray_query_features);
    GetPhysicalDeviceFeatures2(acc_structure_features);
    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &acc_structure_features));

    m_commandBuffer->begin();
    // Build Bottom Level Acceleration Structure
    auto bot_level_build_geometry = std::make_shared<rt::as::BuildGeometryInfoKHR>(
        rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device));
    bot_level_build_geometry->BuildCmdBuffer(*m_device, m_commandBuffer->handle());

    // Build Top Level Acceleration Structure
    rt::as::BuildGeometryInfoKHR top_level_build_geometry =
        rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceTopLevel(DeviceValidationVersion(), *m_device, bot_level_build_geometry);
    top_level_build_geometry.BuildCmdBuffer(*m_device, m_commandBuffer->handle());

    m_commandBuffer->end();
}

TEST_F(PositiveRayTracing, HostAccelerationStructureReference) {
    TEST_DESCRIPTION("Test host side accelerationStructureReference");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    ASSERT_NO_FATAL_FAILURE(InitFramework());
    if (DeviceValidationVersion() < VK_API_VERSION_1_1) {
        GTEST_SKIP() << "At least Vulkan version 1.1 is required";
    }
    if (!AreRequiredExtensionsEnabled()) {
        GTEST_SKIP() << RequiredExtensionsNotSupported() << " not supported";
    }

    auto ray_query_features = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>();
    auto acc_structure_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(&ray_query_features);
    GetPhysicalDeviceFeatures2(acc_structure_features);

    if (acc_structure_features.accelerationStructureHostCommands == VK_FALSE) {
        GTEST_SKIP() << "accelerationStructureHostCommands feature not supported";
    }

    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &acc_structure_features));

    // Build Bottom Level Acceleration Structure
    auto bot_level_build_geometry = std::make_shared<rt::as::BuildGeometryInfoKHR>(
        rt::as::blueprint::BuildGeometryInfoSimpleOnHostBottomLevel(DeviceValidationVersion(), *m_device));
    bot_level_build_geometry->BuildHost(instance(), *m_device);

    // Build Top Level Acceleration Structure
    rt::as::BuildGeometryInfoKHR top_level_build_geometry =
        rt::as::blueprint::BuildGeometryInfoSimpleOnHostTopLevel(DeviceValidationVersion(), *m_device, bot_level_build_geometry);
    top_level_build_geometry.BuildHost(instance(), *m_device);
}

TEST_F(PositiveRayTracing, StridedDeviceAddressRegion) {
    TEST_DESCRIPTION("Test different valid VkStridedDeviceAddressRegionKHR");

    SetTargetApiVersion(VK_API_VERSION_1_2);

    auto bda_features = vku::InitStruct<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR>();
    bda_features.bufferDeviceAddress = VK_TRUE;
    auto ray_tracing_features = vku::InitStruct<VkPhysicalDeviceRayTracingPipelineFeaturesKHR>(&bda_features);
    auto features2 = vku::InitStruct<VkPhysicalDeviceFeatures2KHR>(&ray_tracing_features);
    if (!InitFrameworkForRayTracingTest(this, true, &features2)) {
        GTEST_SKIP() << "unable to init ray tracing test";
    }
    if (IsPlatform(kMockICD)) {
        GTEST_SKIP() << "Test not supported by MockICD";
    }
    // Needed for Ray Tracing
    if (DeviceValidationVersion() < VK_API_VERSION_1_2) {
        GTEST_SKIP() << "At least Vulkan version 1.2 is required";
    }

    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &features2));

    // Create ray tracing pipeline
    VkPipeline raytracing_pipeline = VK_NULL_HANDLE;
    {
        const VkPipelineLayoutObj empty_pipeline_layout(m_device, {});
        VkShaderObj rgen_shader(this, kRayTracingMinimalGlsl, VK_SHADER_STAGE_RAYGEN_BIT_KHR, SPV_ENV_VULKAN_1_2);
        VkShaderObj chit_shader(this, kRayTracingMinimalGlsl, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, SPV_ENV_VULKAN_1_2);

        const VkPipelineLayoutObj pipeline_layout(m_device, {});

        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages;
        shader_stages[0] = vku::InitStructHelper();
        shader_stages[0].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        shader_stages[0].module = chit_shader.handle();
        shader_stages[0].pName = "main";

        shader_stages[1] = vku::InitStructHelper();
        shader_stages[1].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        shader_stages[1].module = rgen_shader.handle();
        shader_stages[1].pName = "main";

        std::array<VkRayTracingShaderGroupCreateInfoKHR, 1> shader_groups;
        shader_groups[0] = vku::InitStructHelper();
        shader_groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shader_groups[0].generalShader = 1;
        shader_groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
        shader_groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
        shader_groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

        VkRayTracingPipelineCreateInfoKHR raytracing_pipeline_ci = vku::InitStructHelper();
        raytracing_pipeline_ci.flags = 0;
        raytracing_pipeline_ci.stageCount = static_cast<uint32_t>(shader_stages.size());
        raytracing_pipeline_ci.pStages = shader_stages.data();
        raytracing_pipeline_ci.pGroups = shader_groups.data();
        raytracing_pipeline_ci.groupCount = shader_groups.size();
        raytracing_pipeline_ci.layout = pipeline_layout.handle();

        const VkResult result = vk::CreateRayTracingPipelinesKHR(m_device->handle(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                                                 &raytracing_pipeline_ci, nullptr, &raytracing_pipeline);
        ASSERT_VK_SUCCESS(result);
    }

    VkBufferObj buffer;
    VkBufferCreateInfo buffer_ci = vku::InitStructHelper();
    buffer_ci.usage =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
    buffer_ci.size = 4096;
    buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer.init_no_mem(*m_device, buffer_ci);

    VkMemoryRequirements mem_reqs;
    vk::GetBufferMemoryRequirements(device(), buffer.handle(), &mem_reqs);

    auto alloc_flags = vku::InitStruct<VkMemoryAllocateFlagsInfo>();
    alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo alloc_info = vku::InitStructHelper(&alloc_flags);
    alloc_info.allocationSize = 4096;
    vkt::DeviceMemory mem(*m_device, alloc_info);
    vk::BindBufferMemory(device(), buffer.handle(), mem.handle(), 0);

    auto ray_tracing_properties = vku::InitStruct<VkPhysicalDeviceRayTracingPipelinePropertiesKHR>();
    GetPhysicalDeviceProperties2(ray_tracing_properties);

    const VkDeviceAddress device_address = buffer.address();

    VkStridedDeviceAddressRegionKHR stridebufregion = {};
    stridebufregion.deviceAddress = device_address;
    stridebufregion.stride = ray_tracing_properties.shaderGroupHandleAlignment;
    stridebufregion.size = stridebufregion.stride;

    m_commandBuffer->begin();

    vk::CmdBindPipeline(m_commandBuffer->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, raytracing_pipeline);

    vk::CmdTraceRaysKHR(m_commandBuffer->handle(), &stridebufregion, &stridebufregion, &stridebufregion, &stridebufregion, 100, 100,
                        1);

    // pRayGenShaderBindingTable->deviceAddress == 0
    {
        VkStridedDeviceAddressRegionKHR valid_region = stridebufregion;
        valid_region.deviceAddress = 0;
        vk::CmdTraceRaysKHR(m_commandBuffer->handle(), &valid_region, &stridebufregion, &stridebufregion, &stridebufregion, 100,
                            100, 1);
    }

    // pRayGenShaderBindingTable->size == 0, deviceAddress is invalid => region is considered unused so no error
    {
        VkStridedDeviceAddressRegionKHR empty_region = stridebufregion;
        empty_region.deviceAddress += buffer.create_info().size + 128;
        empty_region.size = 0;
        empty_region.stride = 0;
        vk::CmdTraceRaysKHR(m_commandBuffer->handle(), &empty_region, &stridebufregion, &stridebufregion, &stridebufregion, 100,
                            100, 1);
    }

    m_commandBuffer->end();

    vk::DestroyPipeline(device(), raytracing_pipeline, nullptr);
}

TEST_F(PositiveRayTracing, BarrierAccessMaskAccelerationStructureRayQueryEnabledRTXDisabled) {
    TEST_DESCRIPTION(
        "Test barrier with access ACCELERATION_STRUCTURE bit."
        "Ray query extension is enabled, as well as feature."
        "RTX extensions are disabled.");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    ASSERT_NO_FATAL_FAILURE(InitFramework());
    if (DeviceValidationVersion() < VK_API_VERSION_1_1) {
        GTEST_SKIP() << "Test requires at least Vulkan 1.1.";
    }
    if (!AreRequiredExtensionsEnabled()) {
        GTEST_SKIP() << RequiredExtensionsNotSupported() << " not supported";
    }
    auto ray_query_feature = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>();
    auto sync2_features = vku::InitStruct<VkPhysicalDeviceSynchronization2FeaturesKHR>(&ray_query_feature);
    GetPhysicalDeviceFeatures2(sync2_features);
    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &sync2_features, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));

    auto mem_barrier = vku::InitStruct<VkMemoryBarrier2>();
    mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

    VkBufferObj buffer(*m_device, 32);

    auto buffer_barrier = vku::InitStruct<VkBufferMemoryBarrier2>();
    buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    buffer_barrier.buffer = buffer.handle();
    buffer_barrier.size = 32;

    VkImageObj image(m_device);
    image.Init(128, 128, 1, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_TILING_OPTIMAL, 0);
    ASSERT_TRUE(image.initialized());

    auto image_barrier = vku::InitStruct<VkImageMemoryBarrier2>();
    image_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    image_barrier.image = image.handle();
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    auto dependency_info = vku::InitStruct<VkDependencyInfo>();
    dependency_info.memoryBarrierCount = 1;
    dependency_info.pMemoryBarriers = &mem_barrier;
    dependency_info.bufferMemoryBarrierCount = 1;
    dependency_info.pBufferMemoryBarriers = &buffer_barrier;
    dependency_info.imageMemoryBarrierCount = 1;
    dependency_info.pImageMemoryBarriers = &image_barrier;

    m_commandBuffer->begin();

    mem_barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    buffer_barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    image_barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    m_commandBuffer->PipelineBarrier2KHR(&dependency_info);

    m_commandBuffer->end();
}

TEST_F(PositiveRayTracing, BarrierAccessMaskAccelerationStructureRayQueryEnabledRTXEnabled) {
    TEST_DESCRIPTION(
        "Test barrier with access ACCELERATION_STRUCTURE bit."
        "Ray query extension is enabled, as well as feature."
        "RTX extensions are disabled.");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    ASSERT_NO_FATAL_FAILURE(InitFramework());
    if (DeviceValidationVersion() < VK_API_VERSION_1_1) {
        GTEST_SKIP() << "Test requires at least Vulkan 1.1.";
    }
    if (!AreRequiredExtensionsEnabled()) {
        GTEST_SKIP() << RequiredExtensionsNotSupported() << " not supported";
    }
    auto ray_query_feature = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>();
    auto sync2_features = vku::InitStruct<VkPhysicalDeviceSynchronization2FeaturesKHR>(&ray_query_feature);
    GetPhysicalDeviceFeatures2(sync2_features);
    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &sync2_features, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));

    auto mem_barrier = vku::InitStruct<VkMemoryBarrier2>();
    mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

    VkBufferObj buffer(*m_device, 32);

    auto buffer_barrier = vku::InitStruct<VkBufferMemoryBarrier2>();
    buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    buffer_barrier.buffer = buffer.handle();
    buffer_barrier.size = 32;

    VkImageObj image(m_device);
    image.Init(128, 128, 1, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_TILING_OPTIMAL, 0);
    ASSERT_TRUE(image.initialized());

    auto image_barrier = vku::InitStruct<VkImageMemoryBarrier2>();
    image_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    image_barrier.image = image.handle();
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    auto dependency_info = vku::InitStruct<VkDependencyInfo>();
    dependency_info.memoryBarrierCount = 1;
    dependency_info.pMemoryBarriers = &mem_barrier;
    dependency_info.bufferMemoryBarrierCount = 1;
    dependency_info.pBufferMemoryBarriers = &buffer_barrier;
    dependency_info.imageMemoryBarrierCount = 1;
    dependency_info.pImageMemoryBarriers = &image_barrier;

    m_commandBuffer->begin();

    // specify VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR as srcStageMask and dstStageMask
    mem_barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    buffer_barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    image_barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    mem_barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    buffer_barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    image_barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    m_commandBuffer->PipelineBarrier2KHR(&dependency_info);

    m_commandBuffer->end();
}

TEST_F(PositiveRayTracing, BarrierSync1NoCrash) {
    TEST_DESCRIPTION("Regression test for nullptr crash when Sync1 barrier API is used for acceleration structure accesses");
    ASSERT_NO_FATAL_FAILURE(Init());

    // This stage can not be used with ACCELERATION_STRUCTURE_READ access when ray query is disabled, but VVL also should not crash.
    constexpr VkPipelineStageFlags invalid_src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    auto barrier = vku::InitStruct<VkMemoryBarrier>();
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    m_errorMonitor->SetUnexpectedError("VUID-vkCmdPipelineBarrier-srcAccessMask-06257");
    m_commandBuffer->begin();
    m_commandBuffer->PipelineBarrier(invalid_src_stage, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    m_commandBuffer->end();
}

TEST_F(PositiveRayTracing, BuildAccelerationStructuresList) {
    TEST_DESCRIPTION("Build a list of destination acceleration structures, then do an update build on that same list");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    auto accel_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
    auto bda_features = vku::InitStruct<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR>(&accel_features);
    auto ray_query_features = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>(&bda_features);
    accel_features.accelerationStructure = VK_TRUE;
    bda_features.bufferDeviceAddress = VK_TRUE;
    ray_query_features.rayQuery = VK_TRUE;

    auto features2 = vku::InitStruct<VkPhysicalDeviceFeatures2KHR>(&ray_query_features);
    if (!InitFrameworkForRayTracingTest(this, true, &features2)) {
        GTEST_SKIP() << "unable to init ray tracing test";
    }
    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &features2, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));

    constexpr size_t build_info_count = 10;

    std::vector<rt::as::BuildGeometryInfoKHR> build_infos;
    for (size_t i = 0; i < build_info_count; ++i) {
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
        build_info.AddFlags(VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
        build_infos.emplace_back(std::move(build_info));
    }

    m_commandBuffer->begin();
    rt::as::BuildAccelerationStructuresKHR(*m_device, m_commandBuffer->handle(), build_infos);

    for (auto& build_info : build_infos) {
        build_info.SetSrcAS(build_info.GetDstAS());
        build_info.SetMode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR);
        build_info.SetDstAS(rt::as::blueprint::AccelStructSimpleOnDeviceBottomLevel(DeviceValidationVersion(), 4096));
    }

    rt::as::BuildAccelerationStructuresKHR(*m_device, m_commandBuffer->handle(), build_infos);
    m_commandBuffer->end();
}

TEST_F(PositiveRayTracing, AccelerationStructuresOverlappingMemory) {
    TEST_DESCRIPTION(
        "Validate acceleration structure building when source/destination acceleration structures and scratch buffers may "
        "overlap.");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    auto accel_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
    auto bda_features = vku::InitStruct<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR>(&accel_features);
    auto ray_query_features = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>(&bda_features);
    accel_features.accelerationStructure = VK_TRUE;
    bda_features.bufferDeviceAddress = VK_TRUE;
    ray_query_features.rayQuery = VK_TRUE;

    auto features2 = vku::InitStruct<VkPhysicalDeviceFeatures2KHR>(&ray_query_features);
    if (!InitFrameworkForRayTracingTest(this, true, &features2)) {
        GTEST_SKIP() << "unable to init ray tracing test";
    }

    if (ray_query_features.rayQuery == VK_FALSE) {
        GTEST_SKIP() << "rayQuery feature is not supported";
    }
    if (accel_features.accelerationStructure == VK_FALSE) {
        GTEST_SKIP() << "accelerationStructure feature is not supported";
    }
    if (bda_features.bufferDeviceAddress == VK_FALSE) {
        GTEST_SKIP() << "bufferDeviceAddress feature is not supported";
    }

    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &features2, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));

    constexpr size_t build_info_count = 3;

    auto alloc_flags = vku::InitStruct<VkMemoryAllocateFlagsInfo>();
    alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo alloc_info = vku::InitStructHelper(&alloc_flags);
    alloc_info.allocationSize = 8192 * build_info_count;
    vkt::DeviceMemory buffer_memory(*m_device, alloc_info);

    // Test using non overlapping memory chunks from the same buffer in multiple builds
    // The scratch buffer is used in multiple builds but bound at different offsets, so no validation error should be issued
    {
        auto scratch_buffer_ci = vku::InitStruct<VkBufferCreateInfo>();
        scratch_buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scratch_buffer_ci.size = 8192 * build_info_count;
        scratch_buffer_ci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        auto scratch_buffer = std::make_shared<VkBufferObj>();
        scratch_buffer->init_no_mem(*m_device, scratch_buffer_ci);
        vk::BindBufferMemory(m_device->device(), scratch_buffer->handle(), buffer_memory.handle(), 0);
        std::vector<rt::as::BuildGeometryInfoKHR> build_infos;
        for (size_t i = 0; i < build_info_count; ++i) {
            auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
            build_info.SetScratchBuffer(scratch_buffer);
            build_info.SetDeviceScratchOffset(i * 8192);
            build_infos.emplace_back(std::move(build_info));
        }

        m_commandBuffer->begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, m_commandBuffer->handle(), build_infos);
        m_commandBuffer->end();
    }
}

TEST_F(PositiveRayTracing, AccelerationStructuresReuseScratchMemory) {
    TEST_DESCRIPTION("Repro https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/6461");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    auto accel_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
    auto bda_features = vku::InitStruct<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR>(&accel_features);
    auto ray_query_features = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>(&bda_features);
    accel_features.accelerationStructure = VK_TRUE;
    bda_features.bufferDeviceAddress = VK_TRUE;
    ray_query_features.rayQuery = VK_TRUE;

    auto features2 = vku::InitStruct<VkPhysicalDeviceFeatures2KHR>(&ray_query_features);
    if (!InitFrameworkForRayTracingTest(this, true, &features2)) {
        GTEST_SKIP() << "unable to init ray tracing test";
    }

    if (ray_query_features.rayQuery == VK_FALSE) {
        GTEST_SKIP() << "rayQuery feature is not supported";
    }
    if (accel_features.accelerationStructure == VK_FALSE) {
        GTEST_SKIP() << "accelerationStructure feature is not supported";
    }
    if (bda_features.bufferDeviceAddress == VK_FALSE) {
        GTEST_SKIP() << "bufferDeviceAddress feature is not supported";
    }

    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &features2, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));

    // Allocate a memory chunk that will be used as backing memory for scratch buffer
    auto alloc_flags = vku::InitStruct<VkMemoryAllocateFlagsInfo>();
    alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo alloc_info = vku::InitStructHelper(&alloc_flags);
    alloc_info.allocationSize = 8192;
    vkt::DeviceMemory common_scratch_memory(*m_device, alloc_info);

    VkCommandBufferObj cmd_buffer_frame_0(m_device, m_commandPool);
    VkCommandBufferObj cmd_buffer_frame_1(m_device, m_commandPool);
    VkCommandBufferObj cmd_buffer_frame_2(m_device, m_commandPool);

    std::vector<rt::as::BuildGeometryInfoKHR> build_infos_frame_0;
    std::vector<rt::as::BuildGeometryInfoKHR> build_infos_frame_1;
    std::vector<rt::as::BuildGeometryInfoKHR> build_infos_frame_2;

    auto scratch_buffer_frame_0 = std::make_shared<VkBufferObj>();
    auto scratch_buffer_frame_1 = std::make_shared<VkBufferObj>();
    auto scratch_buffer_frame_2 = std::make_shared<VkBufferObj>();

    VkFenceObj fence_frame_0(*m_device);
    VkFenceObj fence_frame_1(*m_device);
    VkFenceObj fence_frame_2(*m_device);

    // Frame 0
    {
        // Nothing to wait for, resources used in frame 0 will be released in frame 2

        // Create scratch buffer
        auto scratch_buffer_ci = vku::InitStruct<VkBufferCreateInfo>();
        scratch_buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scratch_buffer_ci.size = 8192;
        scratch_buffer_ci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        scratch_buffer_frame_0->init_no_mem(*m_device, scratch_buffer_ci);

        // Bind memory to scratch buffer
        vk::BindBufferMemory(m_device->device(), scratch_buffer_frame_0->handle(), common_scratch_memory.handle(), 0);

        // Build a dummy acceleration structure
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
        build_info.SetScratchBuffer(scratch_buffer_frame_0);
        build_infos_frame_0.emplace_back(std::move(build_info));
        cmd_buffer_frame_0.begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, cmd_buffer_frame_0.handle(), build_infos_frame_0);

        // Synchronize accesses to scratch buffer memory: next op will be a new acceleration structure build
        auto barrier = vku::InitStruct<VkBufferMemoryBarrier>();
        barrier.buffer = scratch_buffer_frame_0->handle();
        barrier.size = scratch_buffer_ci.size;
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        cmd_buffer_frame_0.PipelineBarrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &barrier, 0,
                                           nullptr);
        cmd_buffer_frame_0.end();

        // Submit command buffer
        VkCommandBuffer cmd_buffer_handle = cmd_buffer_frame_0.handle();
        auto submit_info = vku::InitStruct<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_handle;
        vk::QueueSubmit(m_device->GetDefaultQueue()->handle(), 1, &submit_info, fence_frame_0);
    }

    // Frame 1
    {
        // Still nothing to wait for

        // Create scratch buffer
        auto scratch_buffer_ci = vku::InitStruct<VkBufferCreateInfo>();
        scratch_buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scratch_buffer_ci.size = 8192;
        scratch_buffer_ci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        scratch_buffer_frame_1->init_no_mem(*m_device, scratch_buffer_ci);

        // Bind memory to scratch buffer
        vk::BindBufferMemory(m_device->device(), scratch_buffer_frame_1->handle(), common_scratch_memory.handle(), 0);

        // Build a dummy acceleration structure
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
        build_info.SetScratchBuffer(scratch_buffer_frame_1);
        build_infos_frame_1.emplace_back(std::move(build_info));
        cmd_buffer_frame_1.begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, cmd_buffer_frame_1.handle(), build_infos_frame_1);

        // Synchronize accesses to scratch buffer memory: next op will be a new acceleration structure build
        auto barrier = vku::InitStruct<VkBufferMemoryBarrier>();
        barrier.buffer = scratch_buffer_frame_1->handle();
        barrier.size = scratch_buffer_ci.size;
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        cmd_buffer_frame_1.PipelineBarrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &barrier, 0,
                                           nullptr);
        cmd_buffer_frame_1.end();

        // Submit command buffer
        VkCommandBuffer cmd_buffer_handle = cmd_buffer_frame_1.handle();
        auto submit_info = vku::InitStruct<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_handle;
        vk::QueueSubmit(m_device->GetDefaultQueue()->handle(), 1, &submit_info, fence_frame_1);
    }

    // Frame 2
    {
        // Free resources from frame 0
        fence_frame_0.wait(kWaitTimeout);
        // Destroying buffer triggers VUID-vkDestroyBuffer-buffer-00922, it is still considered in use by cmd_buffer_frame_0 this
        // should not happen assuming synchronization is correct
        // Adding "fence_frame_1.wait(kWaitTimeout);" used to solve this issue.
        // Using a dedicated memory chunk for each scratch buffer also used to solve it.
        // The issue was that when recording a acceleration structure build command,
        // any buffer indirectly mentioned through a device address used to be added using a call to GetBuffersByAddress.
        // So when recording the build happening on frame 1, given that all scratch buffers have the same base device address,
        // scratch_buffer_frame_0 was *also* be added as a child to cmd_buffer_frame_1.
        // So when destroying it hereinafter, since frame 1 is still in flight, scratch_buffer_frame_0 is still
        // considered in use, so 00922 is triggered.
        // => Solution: buffers obtained through a call to GetBuffersByAddress should not get added as children,
        // since there is no 1 to 1 mapping between a device address and a buffer.
        scratch_buffer_frame_0 = nullptr;  // Remove reference
        build_infos_frame_0.clear();       // scratch_buffer_frame_0 will be destroyed in this call

        // Create scratch buffer
        auto scratch_buffer_ci = vku::InitStruct<VkBufferCreateInfo>();
        scratch_buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scratch_buffer_ci.size = 8192;
        scratch_buffer_ci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        scratch_buffer_frame_2->init_no_mem(*m_device, scratch_buffer_ci);

        // Bind memory to scratch buffer
        vk::BindBufferMemory(m_device->device(), scratch_buffer_frame_2->handle(), common_scratch_memory.handle(), 0);

        // Build a dummy acceleration structure
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
        build_info.SetScratchBuffer(scratch_buffer_frame_2);
        build_infos_frame_2.emplace_back(std::move(build_info));
        cmd_buffer_frame_2.begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, cmd_buffer_frame_2.handle(), build_infos_frame_2);

        // Synchronize accesses to scratch buffer memory: next op will be a new acceleration structure build
        auto barrier = vku::InitStruct<VkBufferMemoryBarrier>();
        barrier.buffer = scratch_buffer_frame_2->handle();
        barrier.size = scratch_buffer_ci.size;
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        cmd_buffer_frame_2.PipelineBarrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &barrier, 0,
                                           nullptr);
        cmd_buffer_frame_2.end();

        // Submit command buffer
        VkCommandBuffer cmd_buffer_handle = cmd_buffer_frame_2.handle();
        auto submit_info = vku::InitStruct<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_handle;
        vk::QueueSubmit(m_device->GetDefaultQueue()->handle(), 1, &submit_info, fence_frame_2);
    }

    fence_frame_1.wait(kWaitTimeout);
    fence_frame_2.wait(kWaitTimeout);
}

TEST_F(PositiveRayTracing, AccelerationStructuresDedicatedScratchMemory) {
    TEST_DESCRIPTION(
        "Repro https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/6461"
        "This time, each scratch buffer has its own memory");

    SetTargetApiVersion(VK_API_VERSION_1_1);
    auto accel_features = vku::InitStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
    auto bda_features = vku::InitStruct<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR>(&accel_features);
    auto ray_query_features = vku::InitStruct<VkPhysicalDeviceRayQueryFeaturesKHR>(&bda_features);
    accel_features.accelerationStructure = VK_TRUE;
    bda_features.bufferDeviceAddress = VK_TRUE;
    ray_query_features.rayQuery = VK_TRUE;

    auto features2 = vku::InitStruct<VkPhysicalDeviceFeatures2KHR>(&ray_query_features);
    if (!InitFrameworkForRayTracingTest(this, true, &features2)) {
        GTEST_SKIP() << "unable to init ray tracing test";
    }

    if (ray_query_features.rayQuery == VK_FALSE) {
        GTEST_SKIP() << "rayQuery feature is not supported";
    }
    if (accel_features.accelerationStructure == VK_FALSE) {
        GTEST_SKIP() << "accelerationStructure feature is not supported";
    }
    if (bda_features.bufferDeviceAddress == VK_FALSE) {
        GTEST_SKIP() << "bufferDeviceAddress feature is not supported";
    }

    ASSERT_NO_FATAL_FAILURE(InitState(nullptr, &features2, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));

    VkCommandBufferObj cmd_buffer_frame_0(m_device, m_commandPool);
    VkCommandBufferObj cmd_buffer_frame_1(m_device, m_commandPool);
    VkCommandBufferObj cmd_buffer_frame_2(m_device, m_commandPool);

    std::vector<rt::as::BuildGeometryInfoKHR> build_infos_frame_0;
    std::vector<rt::as::BuildGeometryInfoKHR> build_infos_frame_1;
    std::vector<rt::as::BuildGeometryInfoKHR> build_infos_frame_2;

    VkFenceObj fence_frame_0(*m_device);
    VkFenceObj fence_frame_1(*m_device);
    VkFenceObj fence_frame_2(*m_device);

    // Frame 0
    {
        // Nothing to wait for, resources used in frame 0 will be released in frame 2

        // Build a dummy acceleration structure
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);

        build_infos_frame_0.emplace_back(std::move(build_info));
        cmd_buffer_frame_0.begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, cmd_buffer_frame_0.handle(), build_infos_frame_0);

        // Synchronize accesses to scratch buffer memory: next op will be a new acceleration structure build
        auto barrier = vku::InitStruct<VkBufferMemoryBarrier>();
        barrier.buffer = build_infos_frame_0[0].GetScratchBuffer()->handle();
        barrier.size = build_infos_frame_0[0].GetScratchBuffer()->create_info().size;
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        cmd_buffer_frame_0.PipelineBarrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &barrier, 0,
                                           nullptr);
        cmd_buffer_frame_0.end();

        // Submit command buffer
        VkCommandBuffer cmd_buffer_handle = cmd_buffer_frame_0.handle();
        auto submit_info = vku::InitStruct<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_handle;
        vk::QueueSubmit(m_device->GetDefaultQueue()->handle(), 1, &submit_info, fence_frame_0);
    }

    // Frame 1
    {
        // Still nothing to wait for

        // Build a dummy acceleration structure
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
        build_infos_frame_1.emplace_back(std::move(build_info));
        cmd_buffer_frame_1.begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, cmd_buffer_frame_1.handle(), build_infos_frame_1);

        // Synchronize accesses to scratch buffer memory: next op will be a new acceleration structure build
        auto barrier = vku::InitStruct<VkBufferMemoryBarrier>();
        barrier.buffer = build_infos_frame_1[0].GetScratchBuffer()->handle();
        barrier.size = build_infos_frame_1[0].GetScratchBuffer()->create_info().size;
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        cmd_buffer_frame_1.PipelineBarrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &barrier, 0,
                                           nullptr);
        cmd_buffer_frame_1.end();

        // Submit command buffer
        VkCommandBuffer cmd_buffer_handle = cmd_buffer_frame_1.handle();
        auto submit_info = vku::InitStruct<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_handle;
        vk::QueueSubmit(m_device->GetDefaultQueue()->handle(), 1, &submit_info, fence_frame_1);
    }

    // Frame 2
    {
        // Free resources from frame 0
        fence_frame_0.wait(kWaitTimeout);
        build_infos_frame_0.clear();  // No validation error

        // Build a dummy acceleration structure
        auto build_info = rt::as::blueprint::BuildGeometryInfoSimpleOnDeviceBottomLevel(DeviceValidationVersion(), *m_device);
        build_infos_frame_2.emplace_back(std::move(build_info));
        cmd_buffer_frame_2.begin();
        rt::as::BuildAccelerationStructuresKHR(*m_device, cmd_buffer_frame_2.handle(), build_infos_frame_2);

        // Synchronize accesses to scratch buffer memory: next op will be a new acceleration structure build
        auto barrier = vku::InitStruct<VkBufferMemoryBarrier>();
        barrier.buffer = build_infos_frame_2[0].GetScratchBuffer()->handle();
        barrier.size = build_infos_frame_2[0].GetScratchBuffer()->create_info().size;
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        cmd_buffer_frame_2.PipelineBarrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &barrier, 0,
                                           nullptr);
        cmd_buffer_frame_2.end();

        // Submit command buffer
        VkCommandBuffer cmd_buffer_handle = cmd_buffer_frame_2.handle();
        auto submit_info = vku::InitStruct<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_handle;
        vk::QueueSubmit(m_device->GetDefaultQueue()->handle(), 1, &submit_info, fence_frame_2);
    }

    fence_frame_1.wait(kWaitTimeout);
    fence_frame_2.wait(kWaitTimeout);
}
