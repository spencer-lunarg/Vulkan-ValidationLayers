/*
 * Copyright (c) 2020-2026 The Khronos Group Inc.
 * Copyright (c) 2020-2026 Valve Corporation
 * Copyright (c) 2020-2026 LunarG, Inc.
 * Copyright (c) 2020-2022 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "../framework/layer_validation_tests.h"
#include "../framework/descriptor_helper.h"
#include "../framework/ray_tracing_objects.h"
#include "../framework/gpu_av_helper.h"
#include "../utils/math_utils.h"

class NegativeGpuAVRayTracing : public GpuAVRayTracingTest {};

TEST_F(NegativeGpuAVRayTracing, CmdTraceRaysIndirect) {
    TEST_DESCRIPTION("Test debug printf in raygen shader.");
    SetTargetApiVersion(VK_API_VERSION_1_3);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::rayTracingPipelineTraceRaysIndirect);
    RETURN_IF_SKIP(InitGpuAvFramework());
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }

    RETURN_IF_SKIP(InitState());

    vkt::rt::Pipeline pipeline(*this, m_device);

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require
        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
        layout(location = 0) rayPayloadEXT vec3 hit;

        void main() {
          traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen);

    pipeline.AddGlslMissShader(kRayTracingPayloadMinimalGlsl);
    pipeline.AddGlslClosestHitShader(kRayTracingPayloadMinimalGlsl);

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.CreateDescriptorSet();
    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));
    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas.GetDstAS()->handle());
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    pipeline.Build();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_props = vku::InitStructHelper();
    VkPhysicalDeviceProperties2 props2 = vku::InitStructHelper(&rt_pipeline_props);
    vk::GetPhysicalDeviceProperties2(Gpu(), &props2);

    if (rt_pipeline_props.maxRayDispatchInvocationCount == vvl::kU32Max) {
        GTEST_SKIP() << "maxRayDispatchInvocationCount is maxed out, cannot go past it, skipping test";
    }

    // Create and fill buffers storing indirect data (ray query dimensions)
    const VkBufferUsageFlags buffer_usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    vkt::Buffer trace_rays_big_width(*m_device, 4096, buffer_usage, vkt::device_address);

    VkTraceRaysIndirectCommandKHR trace_rays_dim{rt_pipeline_props.maxRayDispatchInvocationCount + 1, 1, 1};

    uint8_t *ray_query_dimensions_buffer_1_ptr = (uint8_t *)trace_rays_big_width.Memory().Map();
    std::memcpy(ray_query_dimensions_buffer_1_ptr, &trace_rays_dim, sizeof(trace_rays_dim));

    trace_rays_dim = {1, rt_pipeline_props.maxRayDispatchInvocationCount + 1, 1};

    vkt::Buffer trace_rays_big_height(*m_device, 4096, buffer_usage, vkt::device_address);

    uint8_t *ray_query_dimensions_buffer_2_ptr = (uint8_t *)trace_rays_big_height.Memory().Map();
    std::memcpy(ray_query_dimensions_buffer_2_ptr, &trace_rays_dim, sizeof(trace_rays_dim));

    trace_rays_dim = {1, 1, rt_pipeline_props.maxRayDispatchInvocationCount + 1};

    vkt::Buffer trace_ray_big_depth(*m_device, 4096, buffer_usage, vkt::device_address);

    uint8_t *ray_query_dimensions_buffer_3_ptr = (uint8_t *)trace_ray_big_depth.Memory().Map();
    std::memcpy(ray_query_dimensions_buffer_3_ptr, &trace_rays_dim, sizeof(trace_rays_dim));

    m_command_buffer.Begin();
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();

    if (uint64_t(PhysicalDeviceProps().limits.maxComputeWorkGroupCount[0]) *
            uint64_t(PhysicalDeviceProps().limits.maxComputeWorkGroupSize[0]) <
        uint64_t(rt_pipeline_props.maxRayDispatchInvocationCount + 1)) {
        m_errorMonitor->SetDesiredError("VUID-VkTraceRaysIndirectCommandKHR-width-03638");
    }
    m_errorMonitor->SetDesiredError("VUID-VkTraceRaysIndirectCommandKHR-width-03641");
    vk::CmdTraceRaysIndirectKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                                &trace_rays_sbt.callable_sbt, trace_rays_big_width.Address());

    if (uint64_t(PhysicalDeviceProps().limits.maxComputeWorkGroupCount[1]) *
            uint64_t(PhysicalDeviceProps().limits.maxComputeWorkGroupSize[1]) <
        uint64_t(rt_pipeline_props.maxRayDispatchInvocationCount + 1)) {
        m_errorMonitor->SetDesiredError("VUID-VkTraceRaysIndirectCommandKHR-height-03639");
    }
    m_errorMonitor->SetDesiredError("VUID-VkTraceRaysIndirectCommandKHR-width-03641");
    vk::CmdTraceRaysIndirectKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                                &trace_rays_sbt.callable_sbt, trace_rays_big_height.Address());

    if (uint64_t(PhysicalDeviceProps().limits.maxComputeWorkGroupCount[2]) *
            uint64_t(PhysicalDeviceProps().limits.maxComputeWorkGroupSize[2]) <
        uint64_t(rt_pipeline_props.maxRayDispatchInvocationCount + 1)) {
        m_errorMonitor->SetDesiredError("VUID-VkTraceRaysIndirectCommandKHR-depth-03640");
    }
    m_errorMonitor->SetDesiredError("VUID-VkTraceRaysIndirectCommandKHR-width-03641");
    vk::CmdTraceRaysIndirectKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                                &trace_rays_sbt.callable_sbt, trace_ray_big_depth.Address());

    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8545
TEST_F(NegativeGpuAVRayTracing, DISABLED_BasicTraceRaysDeferredBuild) {
    TEST_DESCRIPTION(
        "Setup a ray tracing pipeline (ray generation, miss and closest hit shaders, and deferred build) and acceleration "
        "structure, and trace one "
        "ray. Only call traceRay in the ray generation shader");

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    vkt::rt::Pipeline pipeline(*this, m_device);

    // Set shaders

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)
        #extension GL_EXT_buffer_reference : enable

        layout(buffer_reference, std430) readonly buffer RayTracingParams {
            vec4 nothing;
            float Tmin;
            float Tmax;
        };

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
        layout(binding = 1, set = 0) uniform uniform_buffer {
            RayTracingParams rt_params;
        };

        layout(location = 0) rayPayloadEXT vec3 hit;

        void main() {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), rt_params.Tmin, vec3(0,0,1), rt_params.Tmax, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen);

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    // Create TLAS
    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));
    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas.GetDstAS()->handle());

    // Create uniform_buffer
    vkt::Buffer rt_params_buffer(*m_device, 4 * sizeof(float), 0, vkt::device_address);  // missing space for Tmin and Tmax
    vkt::Buffer uniform_buffer(*m_device, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);
    auto data = static_cast<VkDeviceAddress *>(uniform_buffer.Memory().Map());
    data[0] = rt_params_buffer.Address();
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, 16, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    // Add one to use the descriptor slot GPU-AV tried to reserve
    const uint32_t max_bound_desc_sets = m_device->Physical().limits_.maxBoundDescriptorSets + 1;

    // First try to use too many sets in the pipeline layout
    {
        m_errorMonitor->SetDesiredWarning(
            "This Pipeline Layout has too many descriptor sets that will not allow GPU shader instrumentation to be setup for "
            "pipelines created with it");
        std::vector<const vkt::DescriptorSetLayout *> desc_set_layouts(max_bound_desc_sets);
        for (uint32_t i = 0; i < max_bound_desc_sets; i++) {
            desc_set_layouts[i] = &pipeline.GetDescriptorSet().layout_;
        }
        vkt::PipelineLayout bad_pipe_layout(*m_device, desc_set_layouts);
        m_errorMonitor->VerifyFound();
    }

    // Then use the maximum allowed number of sets
    std::vector<const vkt::DescriptorSetLayout *> des_set_layouts(max_bound_desc_sets - 1);
    for (uint32_t i = 0; i < max_bound_desc_sets - 1; i++) {
        des_set_layouts[i] = &pipeline.GetDescriptorSet().layout_;
    }
    VkPipelineLayoutCreateInfo pipe_layout_ci = vku::InitStructHelper();

    pipeline.GetPipelineLayout().Init(*m_device, pipe_layout_ci, des_set_layouts);

    // Deferred pipeline build
    RETURN_IF_SKIP(pipeline.DeferBuild());
    RETURN_IF_SKIP(pipeline.Build());

    // Bind descriptor set, pipeline, and trace rays
    m_command_buffer.Begin();
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();
    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();
}

TEST_F(NegativeGpuAVRayTracing, ArrayOOBBufferRayGenShader) {
    TEST_DESCRIPTION(
        "GPU validation: Verify detection of out-of-bounds descriptor array indexing and use of uninitialized descriptors in a ray "
        "generation shader");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::runtimeDescriptorArray);
    AddRequiredFeature(vkt::Feature::descriptorBindingSampledImageUpdateAfterBind);
    AddRequiredFeature(vkt::Feature::descriptorBindingPartiallyBound);
    AddRequiredFeature(vkt::Feature::descriptorBindingVariableDescriptorCount);
    AddRequiredFeature(vkt::Feature::shaderSampledImageArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::shaderStorageBufferArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::rayTracingPipelineTraceRaysIndirect2);
    AddRequiredFeature(vkt::Feature::accelerationStructureIndirectBuild);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    std::shared_ptr<vkt::as::BuildGeometryInfoKHR> cube_blas;
    vkt::as::BuildGeometryInfoKHR cubes_tlas =
        vkt::as::blueprint::GetCubesTLAS(*m_device, m_command_buffer, *m_default_queue, cube_blas);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        struct UniformBuffer {
            uint ray_payload_i;
        };
        [[vk::binding(1, 0)]] ConstantBuffer<UniformBuffer> uniform_buffer;

        [[vk::binding(2, 0)]] uniform RWStructuredBuffer<uint4> ray_payload_buffer[];

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            RayPayload ray_payload = { ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0) };
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            ray.Origin = float3(0,0,0);
            ray.Direction = float3(0,0,-1);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    // Make a uniform buffer to be passed to the shader that contains the invalid array index.
    vkt::Buffer uniform_buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);

    // Make another buffer to populate the buffer array to be indexed
    vkt::Buffer ray_payload_buffer(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);

    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline.CreateDescriptorIndexingSet();

    pipeline.Build();

    pipeline.GetDescriptorIndexingSet().WriteDescriptorAccelStruct(0, 1, &cubes_tlas.GetDstAS()->handle());
    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    // Intentionally not writing to 6th array element
    for (uint32_t i = 0; i < 5; ++i) {
        pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(2, ray_payload_buffer, 0, VK_WHOLE_SIZE,
                                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
    }

    pipeline.GetDescriptorIndexingSet().UpdateDescriptorSets();

    {
        m_command_buffer.Begin();

        vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                                  &pipeline.GetDescriptorIndexingSet().set_, 0, nullptr);
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

        // Invoke ray gen shader 1
        vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
        vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                            &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

        vkt::Buffer sbt_buffer_ray_gen_1 = pipeline.GetTraceRaysSbtIndirectBuffer(0, 1, 1, 1);
        vk::CmdTraceRaysIndirect2KHR(m_command_buffer, sbt_buffer_ray_gen_1.Address());

        m_command_buffer.End();

        uint32_t *uniform_buffer_ptr = (uint32_t *)uniform_buffer.Memory().Map();

        {
            uniform_buffer_ptr[0] = 25;
            SCOPED_TRACE("Out of Bounds");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-10068", 1);
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysIndirect2KHR-None-10068", 1);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
        {
            uniform_buffer_ptr[0] = 5;
            SCOPED_TRACE("uninitialized");
            m_errorMonitor->SetDesiredError("08114", 3);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
    }
}

TEST_F(NegativeGpuAVRayTracing, ArrayOOBBufferMissShader) {
    TEST_DESCRIPTION(
        "GPU validation: Verify detection of out-of-bounds descriptor array indexing and use of uninitialized descriptors in a "
        "miss shader");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::runtimeDescriptorArray);
    AddRequiredFeature(vkt::Feature::descriptorBindingSampledImageUpdateAfterBind);
    AddRequiredFeature(vkt::Feature::descriptorBindingPartiallyBound);
    AddRequiredFeature(vkt::Feature::descriptorBindingVariableDescriptorCount);
    AddRequiredFeature(vkt::Feature::shaderSampledImageArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::shaderStorageBufferArrayNonUniformIndexing);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    std::shared_ptr<vkt::as::BuildGeometryInfoKHR> cube_blas;
    vkt::as::BuildGeometryInfoKHR cubes_tlas =
        vkt::as::blueprint::GetCubesTLAS(*m_device, m_command_buffer, *m_default_queue, cube_blas);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        struct UniformBuffer {
            uint ray_payload_i;
        };
        [[vk::binding(1, 0)]] ConstantBuffer<UniformBuffer> uniform_buffer;

        [[vk::binding(2, 0)]] uniform RWStructuredBuffer<uint4> ray_payload_buffer[];

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            ray.Origin = float3(0,0,0);
            ray.Direction = float3(0,0,-1);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            payload.payload = ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    // Make a uniform buffer to be passed to the shader that contains the invalid array index.
    vkt::Buffer uniform_buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);

    // Make another buffer to populate the buffer array to be indexed
    vkt::Buffer ray_payload_buffer(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);

    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline.CreateDescriptorIndexingSet();

    pipeline.Build();

    pipeline.GetDescriptorIndexingSet().WriteDescriptorAccelStruct(0, 1, &cubes_tlas.GetDstAS()->handle());
    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    // Intentionally not writing to 6th array element
    for (uint32_t i = 0; i < 5; ++i) {
        pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(2, ray_payload_buffer, 0, VK_WHOLE_SIZE,
                                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
    }

    pipeline.GetDescriptorIndexingSet().UpdateDescriptorSets();

    {
        m_command_buffer.Begin();

        vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                                  &pipeline.GetDescriptorIndexingSet().set_, 0, nullptr);
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

        // Invoke ray gen shader 1
        vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
        vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                            &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

        m_command_buffer.End();

        uint32_t *uniform_buffer_ptr = (uint32_t *)uniform_buffer.Memory().Map();

        {
            uniform_buffer_ptr[0] = 25;
            SCOPED_TRACE("Out of Bounds");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-10068", 1);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
        {
            uniform_buffer_ptr[0] = 5;
            SCOPED_TRACE("uninitialized");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-08114", 2);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
    }
}

TEST_F(NegativeGpuAVRayTracing, ArrayOOBBufferClosetHitShader) {
    TEST_DESCRIPTION(
        "GPU validation: Verify detection of out-of-bounds descriptor array indexing and use of uninitialized descriptors in a "
        "closest hit shader");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::runtimeDescriptorArray);
    AddRequiredFeature(vkt::Feature::descriptorBindingSampledImageUpdateAfterBind);
    AddRequiredFeature(vkt::Feature::descriptorBindingPartiallyBound);
    AddRequiredFeature(vkt::Feature::descriptorBindingVariableDescriptorCount);
    AddRequiredFeature(vkt::Feature::shaderSampledImageArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::shaderStorageBufferArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::accelerationStructureIndirectBuild);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    std::shared_ptr<vkt::as::BuildGeometryInfoKHR> cube_blas;
    vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::GetCubesTLAS(*m_device, m_command_buffer, *m_default_queue, cube_blas);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        struct UniformBuffer {
            uint ray_payload_i;
        };
        [[vk::binding(1, 0)]] ConstantBuffer<UniformBuffer> uniform_buffer;

        [[vk::binding(2, 0)]] uniform RWStructuredBuffer<uint4> ray_payload_buffer[];

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
            payload.payload = ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0);
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    // Make a uniform buffer to be passed to the shader that contains the invalid array index.
    vkt::Buffer uniform_buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);

    // Make another buffer to populate the buffer array to be indexed
    vkt::Buffer ray_payload_buffer(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);

    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline.CreateDescriptorIndexingSet();

    pipeline.Build();

    pipeline.GetDescriptorIndexingSet().WriteDescriptorAccelStruct(0, 1, &tlas.GetDstAS()->handle());
    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    // Intentionally not writing to 6th array element
    for (uint32_t i = 0; i < 5; ++i) {
        pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(2, ray_payload_buffer, 0, VK_WHOLE_SIZE,
                                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
    }

    pipeline.GetDescriptorIndexingSet().UpdateDescriptorSets();

    {
        m_command_buffer.Begin();

        vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                                  &pipeline.GetDescriptorIndexingSet().set_, 0, nullptr);
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

        // Invoke ray gen shader 1
        vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
        vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                            &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

        m_command_buffer.End();

        uint32_t *uniform_buffer_ptr = (uint32_t *)uniform_buffer.Memory().Map();

        {
            uniform_buffer_ptr[0] = 25;
            SCOPED_TRACE("Out of Bounds");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-10068", 1);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
        {
            uniform_buffer_ptr[0] = 5;
            SCOPED_TRACE("uninitialized");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-08114", 2);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
    }
}

TEST_F(NegativeGpuAVRayTracing, ArrayOOBBufferTwoClosetHitShader) {
    TEST_DESCRIPTION(
        "GPU validation: Verify detection of out-of-bounds descriptor array indexing and use of uninitialized descriptors in two "
        "different closest hit shaders");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::runtimeDescriptorArray);
    AddRequiredFeature(vkt::Feature::descriptorBindingSampledImageUpdateAfterBind);
    AddRequiredFeature(vkt::Feature::descriptorBindingPartiallyBound);
    AddRequiredFeature(vkt::Feature::descriptorBindingVariableDescriptorCount);
    AddRequiredFeature(vkt::Feature::shaderSampledImageArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::shaderStorageBufferArrayNonUniformIndexing);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    std::shared_ptr<vkt::as::BuildGeometryInfoKHR> cube_blas;
    vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::GetCubesTLAS(*m_device, m_command_buffer, *m_default_queue, cube_blas);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        struct UniformBuffer {
            uint ray_payload_i;
        };
        [[vk::binding(1, 0)]] ConstantBuffer<UniformBuffer> uniform_buffer;

        [[vk::binding(2, 0)]] uniform RWStructuredBuffer<uint4> ray_payload_buffer[];

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
            payload.payload += ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0);

            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;
            const uint32_t miss_shader_i = 1;

            // Supposed to hit cube 2, and invoke closestHitShader2
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(0,0,1);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, miss_shader_i, ray, payload);
        }

        [shader("closesthit")]
        void closestHitShader2(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = 999 * barycentric_coords;
            payload.payload += ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0);
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader2");

    // Make a uniform buffer to be passed to the shader that contains the invalid array index.
    vkt::Buffer uniform_buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);

    // Make another buffer to populate the buffer array to be indexed
    vkt::Buffer ray_payload_buffer(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);

    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline.CreateDescriptorIndexingSet();

    pipeline.Build();

    pipeline.GetDescriptorIndexingSet().WriteDescriptorAccelStruct(0, 1, &tlas.GetDstAS()->handle());
    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    // Intentionally not writing to 6th array element
    for (uint32_t i = 0; i < 5; ++i) {
        pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(2, ray_payload_buffer, 0, VK_WHOLE_SIZE,
                                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
    }

    pipeline.GetDescriptorIndexingSet().UpdateDescriptorSets();

    {
        m_command_buffer.Begin();

        vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                                  &pipeline.GetDescriptorIndexingSet().set_, 0, nullptr);
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

        // Invoke ray gen shader 1
        vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
        vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                            &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

        m_command_buffer.End();

        uint32_t *uniform_buffer_ptr = (uint32_t *)uniform_buffer.Memory().Map();

        {
            uniform_buffer_ptr[0] = 25;
            SCOPED_TRACE("Out of Bounds");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-10068", 2);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
        {
            uniform_buffer_ptr[0] = 5;
            SCOPED_TRACE("uninitialized");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-08114", 3);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
    }
}

TEST_F(NegativeGpuAVRayTracing, ArrayOOBBufferRayGenShaderGPL) {
    TEST_DESCRIPTION(
        "GPU validation: Verify detection of out-of-bounds descriptor array indexing and use of uninitialized descriptors in a ray "
        "generation shader coming from a graphics pipeline library");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::rayTracingPipelineTraceRaysIndirect);
    AddRequiredFeature(vkt::Feature::rayTracingPipelineTraceRaysIndirect2);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::runtimeDescriptorArray);
    AddRequiredFeature(vkt::Feature::descriptorBindingSampledImageUpdateAfterBind);
    AddRequiredFeature(vkt::Feature::descriptorBindingPartiallyBound);
    AddRequiredFeature(vkt::Feature::descriptorBindingVariableDescriptorCount);
    AddRequiredFeature(vkt::Feature::shaderSampledImageArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::shaderStorageBufferArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::graphicsPipelineLibrary);
    AddRequiredFeature(vkt::Feature::pipelineLibraryGroupHandles);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    std::shared_ptr<vkt::as::BuildGeometryInfoKHR> cube_blas;
    vkt::as::BuildGeometryInfoKHR cubes_tlas =
        vkt::as::blueprint::GetCubesTLAS(*m_device, m_command_buffer, *m_default_queue, cube_blas);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        struct UniformBuffer {
            uint ray_payload_i;
        };
        [[vk::binding(1, 0)]] ConstantBuffer<UniformBuffer> uniform_buffer;

        [[vk::binding(2, 0)]] uniform RWStructuredBuffer<uint4> ray_payload_buffer[];

        [[vk::binding(10, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            ray_payload.payload = ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0);
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            ray.Origin = float3(0,0,0);
            ray.Direction = float3(0,0,-1);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload ray_payload)
        {
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload ray_payload, in BuiltInTriangleIntersectionAttributes attr)
        {
        }
    )slang";

    vkt::rt::Pipeline pipeline_lib(*this, m_device);
    pipeline_lib.InitLibraryInfo(16 * sizeof(float), false);
    pipeline_lib.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10);
    pipeline_lib.CreateDescriptorIndexingSet();

    pipeline_lib.BuildPipeline();

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.InitLibraryInfo(16 * sizeof(float), true);
    pipeline.AddLibrary(pipeline_lib);
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    // Make a uniform buffer to be passed to the shader that contains the invalid array index.
    vkt::Buffer uniform_buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);

    // Make another buffer to populate the buffer array to be indexed
    vkt::Buffer ray_payload_buffer(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);

    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10);
    pipeline.CreateDescriptorIndexingSet();

    pipeline.Build();

    pipeline.GetDescriptorIndexingSet().WriteDescriptorAccelStruct(0, 1, &cubes_tlas.GetDstAS()->handle());
    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    // Intentionally not writing to 6th array element
    for (uint32_t i = 0; i < 5; ++i) {
        pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(2, ray_payload_buffer, 0, VK_WHOLE_SIZE,
                                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
    }
    vkt::Buffer debug_buffer(*m_device, 16 * sizeof(uint32_t),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostVisibleMemProps);
    uint32_t *debug_buffer_ptr = (uint32_t *)debug_buffer.Memory().Map();
    memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(10, debug_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    pipeline.GetDescriptorIndexingSet().UpdateDescriptorSets();

    {
        m_command_buffer.Begin();

        vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                                  &pipeline.GetDescriptorIndexingSet().set_, 0, nullptr);
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

        // Invoke ray gen shader 1
        vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
        vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                            &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

        vkt::Buffer sbt_buffer_ray_gen_1 = pipeline.GetTraceRaysSbtIndirectBuffer(0, 1, 1, 1);
        vk::CmdTraceRaysIndirect2KHR(m_command_buffer, sbt_buffer_ray_gen_1.Address());

        m_command_buffer.End();

        uint32_t *uniform_buffer_ptr = (uint32_t *)uniform_buffer.Memory().Map();
        {
            uniform_buffer_ptr[0] = 42;
            SCOPED_TRACE("Out of Bounds");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-10068", 1);
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysIndirect2KHR-None-10068", 1);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
        {
            uniform_buffer_ptr[0] = 5;
            SCOPED_TRACE("uninitialized");
            m_errorMonitor->SetDesiredError("08114", 3);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
    }
    ASSERT_EQ(debug_buffer_ptr[0], 4);
}

TEST_F(NegativeGpuAVRayTracing, ArrayOOBBufferMissShaderGPL) {
    TEST_DESCRIPTION(
        "GPU validation: Verify detection of out-of-bounds descriptor array indexing and use of uninitialized descriptors in a "
        "miss shader coming from a graphics pipeline library");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::rayTracingPipelineTraceRaysIndirect);
    AddRequiredFeature(vkt::Feature::rayTracingPipelineTraceRaysIndirect2);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);
    AddRequiredFeature(vkt::Feature::runtimeDescriptorArray);
    AddRequiredFeature(vkt::Feature::descriptorBindingSampledImageUpdateAfterBind);
    AddRequiredFeature(vkt::Feature::descriptorBindingPartiallyBound);
    AddRequiredFeature(vkt::Feature::descriptorBindingVariableDescriptorCount);
    AddRequiredFeature(vkt::Feature::shaderSampledImageArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::shaderStorageBufferArrayNonUniformIndexing);
    AddRequiredFeature(vkt::Feature::graphicsPipelineLibrary);
    AddRequiredFeature(vkt::Feature::pipelineLibraryGroupHandles);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    std::shared_ptr<vkt::as::BuildGeometryInfoKHR> cube_blas;
    vkt::as::BuildGeometryInfoKHR cubes_tlas =
        vkt::as::blueprint::GetCubesTLAS(*m_device, m_command_buffer, *m_default_queue, cube_blas);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        struct UniformBuffer {
            uint ray_payload_i;
        };
        [[vk::binding(1, 0)]] ConstantBuffer<UniformBuffer> uniform_buffer;

        [[vk::binding(2, 0)]] uniform RWStructuredBuffer<uint4> ray_payload_buffer[];

        [[vk::binding(10, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;

        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            ray.Origin = float3(0,0,0);
            ray.Direction = float3(0,0,-1);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload ray_payload)
        {
            InterlockedAdd(debug_buffer[0], 1);
            ray_payload.payload = ray_payload_buffer[uniform_buffer.ray_payload_i].Load(0);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload ray_payload, in BuiltInTriangleIntersectionAttributes attr)
        {
        }
    )slang";

    vkt::rt::Pipeline pipeline_lib(*this, m_device);
    pipeline_lib.InitLibraryInfo(16 * sizeof(float), false);
    pipeline_lib.AddSlangMissShader(slang_shader, "missShader");
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline_lib.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10);
    pipeline_lib.CreateDescriptorIndexingSet();

    pipeline_lib.BuildPipeline();

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.InitLibraryInfo(16 * sizeof(float), true);
    pipeline.AddLibrary(pipeline_lib);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    // Make a uniform buffer to be passed to the shader that contains the invalid array index.
    vkt::Buffer uniform_buffer(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostVisibleMemProps);

    // Make another buffer to populate the buffer array to be indexed
    vkt::Buffer ray_payload_buffer(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);

    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 6);
    pipeline.AddDescriptorIndexingBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10);
    pipeline.CreateDescriptorIndexingSet();

    pipeline.Build();

    pipeline.GetDescriptorIndexingSet().WriteDescriptorAccelStruct(0, 1, &cubes_tlas.GetDstAS()->handle());
    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(1, uniform_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    // Intentionally not writing to 6th array element
    for (uint32_t i = 0; i < 5; ++i) {
        pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(2, ray_payload_buffer, 0, VK_WHOLE_SIZE,
                                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
    }
    vkt::Buffer debug_buffer(*m_device, 16 * sizeof(uint32_t),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostVisibleMemProps);
    uint32_t *debug_buffer_ptr = (uint32_t *)debug_buffer.Memory().Map();
    memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    pipeline.GetDescriptorIndexingSet().WriteDescriptorBufferInfo(10, debug_buffer, 0, VK_WHOLE_SIZE,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    pipeline.GetDescriptorIndexingSet().UpdateDescriptorSets();

    {
        m_command_buffer.Begin();

        vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                                  &pipeline.GetDescriptorIndexingSet().set_, 0, nullptr);
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

        // Invoke ray gen shader 1
        vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
        vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                            &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

        vkt::Buffer sbt_buffer_ray_gen_1 = pipeline.GetTraceRaysSbtIndirectBuffer(0, 1, 1, 1);
        vk::CmdTraceRaysIndirect2KHR(m_command_buffer, sbt_buffer_ray_gen_1.Address());

        m_command_buffer.End();

        uint32_t *uniform_buffer_ptr = (uint32_t *)uniform_buffer.Memory().Map();
        {
            uniform_buffer_ptr[0] = 42;
            SCOPED_TRACE("Out of Bounds");
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-10068", 1);
            m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysIndirect2KHR-None-10068", 1);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
        {
            uniform_buffer_ptr[0] = 5;
            SCOPED_TRACE("uninitialized");
            m_errorMonitor->SetDesiredError("08114", 3);
            m_default_queue->SubmitAndWait(m_command_buffer);
            m_errorMonitor->VerifyFound();
        }
    }
    ASSERT_EQ(debug_buffer_ptr[0], 4);
}

TEST_F(NegativeGpuAVRayTracing, InvalidBlasReference1) {
    TEST_DESCRIPTION(
        "Validate an invalid BLAS reference in a TLAS build - first element BLAS ref invalid, subsequent ones valid."
        "Trace a ray into the built TLAS to confirm it was built correctly without the invalid ref but with the valid ones.");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    vkt::as::GeometryKHR cube(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
    vkt::as::BuildGeometryInfoKHR cube_blas = vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));

    m_command_buffer.Begin();
    cube_blas.BuildCmdBuffer(m_command_buffer);
    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    std::vector<vkt::as::GeometryKHR> cube_instances(1);
    cube_instances[0].SetType(vkt::as::GeometryKHR::Type::Instance);

    VkAccelerationStructureInstanceKHR cube_instance_1{};
    cube_instance_1.transform.matrix[0][0] = 1.0f;
    cube_instance_1.transform.matrix[1][1] = 1.0f;
    cube_instance_1.transform.matrix[2][2] = 1.0f;
    cube_instance_1.transform.matrix[0][3] = 50.0f;
    cube_instance_1.transform.matrix[1][3] = 0.0f;
    cube_instance_1.transform.matrix[2][3] = 0.0f;
    cube_instance_1.mask = 0xff;
    cube_instance_1.instanceCustomIndex = 0;
    // Cube instance 1 will be associated to closest hit shader 1
    cube_instance_1.instanceShaderBindingTableRecordOffset = 0;
    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);
    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    VkAccelerationStructureInstanceKHR cube_instance_2{};
    cube_instance_2.transform.matrix[0][0] = 1.0f;
    cube_instance_2.transform.matrix[1][1] = 1.0f;
    cube_instance_2.transform.matrix[2][2] = 1.0f;
    cube_instance_2.transform.matrix[0][3] = 0.0f;
    cube_instance_2.transform.matrix[1][3] = 0.0f;
    cube_instance_2.transform.matrix[2][3] = 50.0f;
    cube_instance_2.mask = 0xff;
    cube_instance_2.instanceCustomIndex = 0;
    // Cube instance 2 will be associated to closest hit shader 1
    cube_instance_2.instanceShaderBindingTableRecordOffset = 0;

    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_2);

    cube_instances[0].UpdateAccelerationStructureInstance(0, [](VkAccelerationStructureInstanceKHR &instance) {
        instance.accelerationStructureReference = static_cast<uint64_t>(0xbaadbeef);
    });

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_build_info;
    {
        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(cube_instances));
        tlas_build_info.emplace_back(std::move(tlas));
        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_build_info);
        m_command_buffer.End();

        m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12281",
                                             "Infos\\[0\\].pGeometries\\[0\\].geometry.instances<VkAccelerationStructureInstance>"
                                             "\\[0\\].accelerationStructureReference \\(0xbaadbeef\\)");
        m_default_queue->Submit(m_command_buffer);
        m_device->Wait();
        m_errorMonitor->VerifyFound();
    }
    // Buffer used to count invocations for the 3 shaders
    vkt::Buffer debug_buffer(*m_device, 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             kHostVisibleMemProps);
    auto debug_buffer_ptr = static_cast<uint32_t *>(debug_buffer.Memory().Map());
    std::memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        [[vk::binding(1, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            InterlockedAdd(debug_buffer[1], 1);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            InterlockedAdd(debug_buffer[2], 1);
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    pipeline.Build();

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_build_info[0].GetDstAS()->handle());
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, debug_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    m_command_buffer.Begin();

    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
    vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                        &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly built
    ASSERT_EQ(debug_buffer_ptr[0], 1);
    ASSERT_EQ(debug_buffer_ptr[1], 0);
    ASSERT_EQ(debug_buffer_ptr[2], 1);
}

TEST_F(NegativeGpuAVRayTracing, InvalidBlasReference2) {
    TEST_DESCRIPTION(
        "Validate an invalid BLAS reference in a TLAS build - first element BLAS ref invalid, subsequent ones valid."
        "Trace a ray into the built TLAS to confirm it was built correctly without the invalid ref but with the valid ones."
        "Add other BLAS builds to the build command, and assert they were also done.");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    vkt::as::GeometryKHR cube(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
    vkt::as::BuildGeometryInfoKHR cube_blas = vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));

    m_command_buffer.Begin();
    cube_blas.BuildCmdBuffer(m_command_buffer);
    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    std::vector<vkt::as::GeometryKHR> tlas_1(1);
    tlas_1[0].SetType(vkt::as::GeometryKHR::Type::Instance);

    VkAccelerationStructureInstanceKHR cube_instance_1{};
    cube_instance_1.transform.matrix[0][0] = 1.0f;
    cube_instance_1.transform.matrix[1][1] = 1.0f;
    cube_instance_1.transform.matrix[2][2] = 1.0f;
    cube_instance_1.transform.matrix[0][3] = 50.0f;
    cube_instance_1.transform.matrix[1][3] = 0.0f;
    cube_instance_1.transform.matrix[2][3] = 0.0f;
    cube_instance_1.mask = 0xff;
    cube_instance_1.instanceCustomIndex = 0;
    // Cube instance 1 will be associated to closest hit shader 1
    cube_instance_1.instanceShaderBindingTableRecordOffset = 0;
    tlas_1[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);
    tlas_1[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    VkAccelerationStructureInstanceKHR cube_instance_2{};
    cube_instance_2.transform.matrix[0][0] = 1.0f;
    cube_instance_2.transform.matrix[1][1] = 1.0f;
    cube_instance_2.transform.matrix[2][2] = 1.0f;
    cube_instance_2.transform.matrix[0][3] = 0.0f;
    cube_instance_2.transform.matrix[1][3] = 0.0f;
    cube_instance_2.transform.matrix[2][3] = 50.0f;
    cube_instance_2.mask = 0xff;
    cube_instance_2.instanceCustomIndex = 0;
    // Cube instance 2 will be associated to closest hit shader 1
    cube_instance_2.instanceShaderBindingTableRecordOffset = 0;

    tlas_1[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_2);

    tlas_1[0].UpdateAccelerationStructureInstance(0, [](VkAccelerationStructureInstanceKHR &instance) {
        instance.accelerationStructureReference = static_cast<uint64_t>(0xbaadbeef);
    });

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_and_blass_build_info_1;
    {
        vkt::as::GeometryKHR cube_1(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
        vkt::as::BuildGeometryInfoKHR cube_blas_1 =
            vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));
        tlas_and_blass_build_info_1.emplace_back(std::move(cube_blas_1));

        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(tlas_1));
        tlas_and_blass_build_info_1.emplace_back(std::move(tlas));

        vkt::as::GeometryKHR cube_2(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
        vkt::as::BuildGeometryInfoKHR cube_blas_2 =
            vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));
        tlas_and_blass_build_info_1.emplace_back(std::move(cube_blas_2));

        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_and_blass_build_info_1);
        m_command_buffer.End();

        m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12281",
                                             "Infos\\[1\\].pGeometries\\[0\\].geometry.instances<VkAccelerationStructureInstance>"
                                             "\\[0\\].accelerationStructureReference \\(0xbaadbeef\\)");
        m_default_queue->Submit(m_command_buffer);
        m_device->Wait();
        m_errorMonitor->VerifyFound();
    }

    // Build a 2nd TLAS referencing the 2 BLAS built along with the first TLAS.
    // Rays will be traced into this TLAS, to make sure those BLAS were correctly built.
    std::vector<vkt::as::GeometryKHR> tlas_2(1);
    tlas_2[0].SetType(vkt::as::GeometryKHR::Type::Instance);
    tlas_2[0].AddInstanceDeviceAccelStructRef(*m_device, tlas_and_blass_build_info_1[0].GetDstAS()->handle(), cube_instance_1);
    tlas_2[0].AddInstanceDeviceAccelStructRef(*m_device, tlas_and_blass_build_info_1[2].GetDstAS()->handle(), cube_instance_2);

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_build_info_2;
    {
        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(tlas_2));
        tlas_build_info_2.emplace_back(std::move(tlas));

        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_build_info_2);
        m_command_buffer.End();

        m_default_queue->Submit(m_command_buffer);
        m_device->Wait();
    }

    // Buffer used to count invocations for the 3 shaders
    vkt::Buffer debug_buffer(*m_device, 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             kHostVisibleMemProps);
    auto debug_buffer_ptr = static_cast<uint32_t *>(debug_buffer.Memory().Map());
    std::memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        [[vk::binding(1, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);

            // Will hit cube 2
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(0,0,1);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);

        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            InterlockedAdd(debug_buffer[1], 1);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            InterlockedAdd(debug_buffer[2], 1);
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    pipeline.Build();

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_and_blass_build_info_1[1].GetDstAS()->handle());
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, debug_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    m_command_buffer.Begin();

    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
    vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                        &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly built
    ASSERT_EQ(debug_buffer_ptr[0], 1);
    ASSERT_EQ(debug_buffer_ptr[1], 0);
    ASSERT_EQ(debug_buffer_ptr[2], 2);

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_build_info_2[0].GetDstAS()->handle());

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly built
    ASSERT_EQ(debug_buffer_ptr[0], 2);
    ASSERT_EQ(debug_buffer_ptr[1], 0);
    ASSERT_EQ(debug_buffer_ptr[2], 4);
}

TEST_F(NegativeGpuAVRayTracing, InvalidBlasReference3) {
    TEST_DESCRIPTION(
        "Validate an invalid BLAS reference in a TLAS build - first element BLAS ref invalid because its underlying buffer has "
        "been destroyed, subsequent ones valid."
        "Trace a ray into the built TLAS to confirm it was built correctly without the invalid ref but with the valid ones.");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    vkt::as::GeometryKHR cube(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
    vkt::as::BuildGeometryInfoKHR cube_blas = vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));

    m_command_buffer.Begin();
    cube_blas.BuildCmdBuffer(m_command_buffer);
    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    std::vector<vkt::as::GeometryKHR> cube_instances(1);
    cube_instances[0].SetType(vkt::as::GeometryKHR::Type::Instance);

    VkAccelerationStructureInstanceKHR cube_instance_1{};
    cube_instance_1.transform.matrix[0][0] = 1.0f;
    cube_instance_1.transform.matrix[1][1] = 1.0f;
    cube_instance_1.transform.matrix[2][2] = 1.0f;
    cube_instance_1.transform.matrix[0][3] = 50.0f;
    cube_instance_1.transform.matrix[1][3] = 0.0f;
    cube_instance_1.transform.matrix[2][3] = 0.0f;
    cube_instance_1.mask = 0xff;
    cube_instance_1.instanceCustomIndex = 0;
    // Cube instance 1 will be associated to closest hit shader 1
    cube_instance_1.instanceShaderBindingTableRecordOffset = 0;
    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    VkAccelerationStructureInstanceKHR cube_instance_2{};
    cube_instance_2.transform.matrix[0][0] = 1.0f;
    cube_instance_2.transform.matrix[1][1] = 1.0f;
    cube_instance_2.transform.matrix[2][2] = 1.0f;
    cube_instance_2.transform.matrix[0][3] = 0.0f;
    cube_instance_2.transform.matrix[1][3] = 0.0f;
    cube_instance_2.transform.matrix[2][3] = 50.0f;
    cube_instance_2.mask = 0xff;
    cube_instance_2.instanceCustomIndex = 0;
    // Cube instance 2 will be associated to closest hit shader 1
    cube_instance_2.instanceShaderBindingTableRecordOffset = 0;

    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_2);

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_build_info;
    {
        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(cube_instances));
        tlas_build_info.emplace_back(std::move(tlas));
        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_build_info);
        m_command_buffer.End();

        const VkDeviceAddress cube_blas_addr = cube_blas.GetDstAS()->GetAccelerationStructureDeviceAddress();
        // Destroy buffer, but BLAS will be referenced in a TLAS build command
        cube_blas.GetDstAS()->GetBuffer().Destroy();

        std::ostringstream expected_error_1;
        expected_error_1 << "Infos\\[0\\].pGeometries\\[0\\].geometry.instances<VkAccelerationStructureInstance>\\[0\\]."
                            "accelerationStructureReference \\(0x"
                         << std::hex << cube_blas_addr << "\\).*underlying buffer.*VkAccelerationStructureKHR.*"
                         << CastFromHandle<uint64_t>(cube_blas.GetDstAS()->handle());
        std::ostringstream expected_error_2;
        expected_error_2 << "Infos\\[0\\].pGeometries\\[0\\].geometry.instances<VkAccelerationStructureInstance>\\[1\\]."
                            "accelerationStructureReference \\(0x"
                         << std::hex << cube_blas_addr << "\\).*underlying buffer.*VkAccelerationStructureKHR.*"
                         << CastFromHandle<uint64_t>(cube_blas.GetDstAS()->handle());
        m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12281", expected_error_1.str());
        m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12281", expected_error_2.str());
        m_default_queue->Submit(m_command_buffer);
        m_device->Wait();
        m_errorMonitor->VerifyFound();
    }
    // Buffer used to count invocations for the 3 shaders
    vkt::Buffer debug_buffer(*m_device, 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             kHostVisibleMemProps);
    auto debug_buffer_ptr = static_cast<uint32_t *>(debug_buffer.Memory().Map());
    std::memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        [[vk::binding(1, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            InterlockedAdd(debug_buffer[1], 1);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            InterlockedAdd(debug_buffer[2], 1);
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    pipeline.Build();

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_build_info[0].GetDstAS()->handle());
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, debug_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    m_command_buffer.Begin();

    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
    vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                        &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly *not* built
    ASSERT_EQ(debug_buffer_ptr[0], 1);
    ASSERT_EQ(debug_buffer_ptr[1], 1);
    ASSERT_EQ(debug_buffer_ptr[2], 0);
}

TEST_F(NegativeGpuAVRayTracing, AccelerationStructureBufferUsage) {
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    RETURN_IF_SKIP(InitGpuAvFramework());
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(heap_props);

    const VkDeviceSize descriptor_size =
        vk::GetPhysicalDeviceDescriptorSizeEXT(gpu_, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    const uint32_t push_data_offset = 72u;
    const uint32_t address_data_offset = 384u;

    vkt::rt::Pipeline pipeline(*this, m_device);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mapping.sourceData.indirectAddress.pushOffset = push_data_offset;
    mapping.sourceData.indirectAddress.addressOffset = address_data_offset;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

        layout(location = 0) rayPayloadEXT vec3 hit;

        void main() {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen, nullptr, &mapping_info);

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddCreateInfoFlags2(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT);
    pipeline.Build();

    vkt::Buffer uniform_buffer(*m_device, address_data_offset + sizeof(VkDeviceAddress), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR,
                               vkt::device_address);
    VkDeviceAddress device_address = uniform_buffer.Address();

    VkDeviceAddressRangeEXT device_address_range;
    device_address_range.address = device_address;
    device_address_range.size = 0;

    VkResourceDescriptorInfoEXT resource_info = vku::InitStructHelper();
    resource_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    resource_info.data.pAddressRange = &device_address_range;

    vkt::Buffer address_buffer(*m_device, address_data_offset + sizeof(VkDeviceAddress), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR,
                               vkt::device_address);
    uint8_t *address_data = static_cast<uint8_t *>(address_buffer.Memory().Map());

    VkHostAddressRangeEXT descriptor;
    descriptor.address = address_data + address_data_offset;
    descriptor.size = descriptor_size;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &resource_info, &descriptor);

    VkDeviceAddress address = address_buffer.Address();

    VkPushDataInfoEXT push_data_info = vku::InitStructHelper();
    push_data_info.offset = push_data_offset;
    push_data_info.data.address = &address;
    push_data_info.data.size = sizeof(VkDeviceAddress);

    m_command_buffer.Begin();
    vk::CmdPushDataEXT(m_command_buffer, &push_data_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-11440");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVRayTracing, AccelerationStructureBufferAlignment) {
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    RETURN_IF_SKIP(InitGpuAvFramework());
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(heap_props);

    const VkDeviceSize descriptor_size =
        vk::GetPhysicalDeviceDescriptorSizeEXT(gpu_, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    const uint32_t push_data_offset = 72u;
    const uint32_t address_data_offset = 384u;

    vkt::rt::Pipeline pipeline(*this, m_device);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mapping.sourceData.indirectAddress.pushOffset = push_data_offset;
    mapping.sourceData.indirectAddress.addressOffset = address_data_offset;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

        layout(location = 0) rayPayloadEXT vec3 hit;

        void main() {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen, nullptr, &mapping_info);

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddCreateInfoFlags2(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT);
    pipeline.Build();

    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));

    VkDeviceAddress tlas_device_address = tlas.GetDstAS()->GetBufferDeviceAddress() + 128;

    VkDeviceAddressRangeEXT device_address_range;
    device_address_range.address = tlas_device_address;
    device_address_range.size = 0;

    VkResourceDescriptorInfoEXT resource_info = vku::InitStructHelper();
    resource_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    resource_info.data.pAddressRange = &device_address_range;

    vkt::Buffer address_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, vkt::device_address);
    uint8_t *address_data = static_cast<uint8_t *>(address_buffer.Memory().Map());

    VkHostAddressRangeEXT descriptor;
    descriptor.address = address_data + address_data_offset;
    descriptor.size = descriptor_size;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &resource_info, &descriptor);

    VkDeviceAddress address = address_buffer.Address();

    VkPushDataInfoEXT push_data_info = vku::InitStructHelper();
    push_data_info.offset = push_data_offset;
    push_data_info.data.address = &address;
    push_data_info.data.size = sizeof(VkDeviceAddress);

    m_command_buffer.Begin();
    vk::CmdPushDataEXT(m_command_buffer, &push_data_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-11443");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// Todo
TEST_F(NegativeGpuAVRayTracing, DISABLED_AccelerationStructureHeapAlignment) {
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAvFramework());
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(heap_props);

    const VkDeviceSize descriptor_size =
        vk::GetPhysicalDeviceDescriptorSizeEXT(gpu_, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);

    vkt::rt::Pipeline pipeline(*this, m_device);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mapping.sourceData.constantOffset = {};

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)
        #extension GL_EXT_descriptor_heap : require

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas[];

        layout(location = 0) rayPayloadEXT vec3 hit;

        void main() {
            traceRayEXT(tlas[0], gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen, nullptr, &mapping_info);

    /*const std::string ray_gen = R"(
               OpCapability UntypedPointersKHR
               OpCapability RayTracingKHR
               OpCapability DescriptorHeapEXT
               OpExtension "SPV_EXT_descriptor_heap"
               OpExtension "SPV_KHR_ray_tracing"
               OpExtension "SPV_KHR_untyped_pointers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint RayGenerationKHR %4 "main" %7 %28
               OpSource GLSL 460
               OpSourceExtension "GL_EXT_descriptor_heap"
               OpSourceExtension "GL_EXT_ray_tracing"
               OpName %4 "main"
               OpName %7 "resource_heap"
               OpName %28 "hit"
               OpDecorate %7 BuiltIn ResourceHeapEXT
               OpDecorateId %14 ArrayStrideIdEXT %13
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeUntypedPointerKHR UniformConstant
          %7 = OpUntypedVariableKHR %6 UniformConstant
          %8 = OpTypeInt 32 1
          %9 = OpConstant %8 0
         %10 = OpTypeAccelerationStructureKHR
         %11 = OpTypeUntypedPointerKHR Uniform
         %13 = OpConstantSizeOfEXT %8 %10
         %14 = OpTypeRuntimeArray %10
         %16 = OpTypeInt 32 0
         %17 = OpConstant %16 1
         %18 = OpConstant %16 255
         %19 = OpConstant %16 0
         %20 = OpTypeFloat 32
         %21 = OpTypeVector %20 3
         %22 = OpConstant %20 0
         %23 = OpConstant %20 1
         %24 = OpConstantComposite %21 %22 %22 %23
         %25 = OpConstant %20 0.100000001
         %26 = OpConstant %20 1000
         %27 = OpTypePointer RayPayloadKHR %21
         %28 = OpVariable %27 RayPayloadKHR
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpUntypedAccessChainKHR %11 %14 %7 %9
         %15 = OpLoad %10 %12
               OpTraceRayKHR %15 %17 %18 %19 %19 %19 %24 %25 %24 %26 %28
               OpReturn
               OpFunctionEnd
        )";
    pipeline.AddSpirvRayGenShader(ray_gen.c_str(), "main");*/

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddCreateInfoFlags2(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT);
    pipeline.Build();

    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));

    VkDeviceSize heap_size = heap_props.bufferDescriptorAlignment + heap_props.minResourceHeapReservedRange;
    heap_size = Align(heap_size, heap_props.bufferDescriptorAlignment);
    heap_size = Align(heap_size, heap_props.imageDescriptorAlignment);

    VkBufferUsageFlags2CreateInfo buffer_usage = vku::InitStructHelper();
    buffer_usage.usage = VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateFlagsInfo allocate_flag_info = vku::InitStructHelper();
    allocate_flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    vkt::Buffer resource_heap(*m_device, vkt::Buffer::CreateInfo(heap_size, 0, {}, &buffer_usage),
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &allocate_flag_info);
    uint8_t *heap_data = static_cast<uint8_t *>(resource_heap.Memory().Map());

    VkDeviceAddress tlas_device_address = tlas.GetDstAS()->GetBufferDeviceAddress();

    VkDeviceAddressRangeEXT device_address_range;
    device_address_range.address = tlas_device_address;
    device_address_range.size = descriptor_size;

    VkResourceDescriptorInfoEXT resource_info = vku::InitStructHelper();
    resource_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    resource_info.data.pAddressRange = &device_address_range;

    VkHostAddressRangeEXT descriptor;
    descriptor.address = heap_data;
    descriptor.size = descriptor_size;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &resource_info, &descriptor);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = resource_heap.Address();
    bind_resource_info.heapRange.size = resource_heap.CreateInfo().size;
    bind_resource_info.reservedRangeOffset = resource_heap.CreateInfo().size - heap_props.minResourceHeapReservedRange;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;

    m_command_buffer.Begin();
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11350");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVRayTracing, BLASBuiltAndUsedInTLAS) {
    TEST_DESCRIPTION("In the same vkCmdBuildAccelerationStructures, build a BLAS and also use it in a TLAS build");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    vkt::as::GeometryKHR cube(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
    vkt::as::BuildGeometryInfoKHR cube_blas = vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));

    m_command_buffer.Begin();
    cube_blas.BuildCmdBuffer(m_command_buffer);
    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    std::vector<vkt::as::GeometryKHR> cube_instances(1);
    cube_instances[0].SetType(vkt::as::GeometryKHR::Type::Instance);

    VkAccelerationStructureInstanceKHR cube_instance_1{};
    cube_instance_1.transform.matrix[0][0] = 1.0f;
    cube_instance_1.transform.matrix[1][1] = 1.0f;
    cube_instance_1.transform.matrix[2][2] = 1.0f;
    cube_instance_1.transform.matrix[0][3] = 50.0f;
    cube_instance_1.transform.matrix[1][3] = 0.0f;
    cube_instance_1.transform.matrix[2][3] = 0.0f;
    cube_instance_1.mask = 0xff;
    cube_instance_1.instanceCustomIndex = 0;
    // Cube instance 1 will be associated to closest hit shader 1
    cube_instance_1.instanceShaderBindingTableRecordOffset = 0;
    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    VkAccelerationStructureInstanceKHR cube_instance_2{};
    cube_instance_2.transform.matrix[0][0] = 1.0f;
    cube_instance_2.transform.matrix[1][1] = 1.0f;
    cube_instance_2.transform.matrix[2][2] = 1.0f;
    cube_instance_2.transform.matrix[0][3] = 0.0f;
    cube_instance_2.transform.matrix[1][3] = 0.0f;
    cube_instance_2.transform.matrix[2][3] = 50.0f;
    cube_instance_2.mask = 0xff;
    cube_instance_2.instanceCustomIndex = 0;
    // Cube instance 2 will be associated to closest hit shader 1
    cube_instance_2.instanceShaderBindingTableRecordOffset = 0;

    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_2);

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_build_info;
    {
        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(cube_instances));
        tlas_build_info.emplace_back(std::move(tlas));

        // Create a blas, backed by a buffer also used to back a blas referenced in the previous TLAS.
        // => Their memory overlaps, building them in the same command is invalid.
        vkt::as::GeometryKHR cube_1(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
        vkt::as::BuildGeometryInfoKHR cube_blas_1 =
            vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));
        const VkAccelerationStructureBuildSizesInfoKHR cube_blas_size_info = cube_blas_1.GetSizeInfo();

        VkBufferCreateInfo cube_blas_1_buffer_ci = vku::InitStructHelper();
        cube_blas_1_buffer_ci.size = cube_blas_size_info.accelerationStructureSize;
        cube_blas_1_buffer_ci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        vkt::Buffer cube_blas_1_buffer;
        cube_blas_1_buffer.InitNoMemory(*m_device, cube_blas_1_buffer_ci);
        cube_blas_1_buffer.BindMemory(cube_blas.GetDstAS()->GetBuffer().Memory(), 0);

        cube_blas_1.GetDstAS()->SetDeviceBuffer(std::move(cube_blas_1_buffer));
        tlas_build_info.emplace_back(std::move(cube_blas_1));

        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_build_info);
        m_command_buffer.End();

        m_errorMonitor->SetDesiredError("VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03706", 2);
        m_default_queue->Submit(m_command_buffer);
        m_device->Wait();
        m_errorMonitor->VerifyFound();
    }
    // Buffer used to count invocations for the 3 shaders
    vkt::Buffer debug_buffer(*m_device, 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             kHostVisibleMemProps);
    auto debug_buffer_ptr = static_cast<uint32_t *>(debug_buffer.Memory().Map());
    std::memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        [[vk::binding(1, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            InterlockedAdd(debug_buffer[1], 1);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            InterlockedAdd(debug_buffer[2], 1);
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    pipeline.Build();

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_build_info[0].GetDstAS()->handle());
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, debug_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    m_command_buffer.Begin();

    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
    vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                        &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly built, but is empty
    ASSERT_EQ(debug_buffer_ptr[0], 1);  // Ray gen shader invocations count
    ASSERT_EQ(debug_buffer_ptr[1], 1);  // Miss shader invocations count
    ASSERT_EQ(debug_buffer_ptr[2], 0);  // Closest hit shader invocations count
}

TEST_F(NegativeGpuAVRayTracing, BLASUpdatedAndUsedInTLAS) {
    TEST_DESCRIPTION("In the same vkCmdBuildAccelerationStructures, update a BLAS and also use it in a TLAS build");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    vkt::as::GeometryKHR cube(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
    vkt::as::BuildGeometryInfoKHR cube_blas = vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));
    cube_blas.AddFlags(VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

    m_command_buffer.Begin();
    cube_blas.BuildCmdBuffer(m_command_buffer);
    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    std::vector<vkt::as::GeometryKHR> cube_instances(1);
    cube_instances[0].SetType(vkt::as::GeometryKHR::Type::Instance);

    VkAccelerationStructureInstanceKHR cube_instance_1{};
    cube_instance_1.transform.matrix[0][0] = 1.0f;
    cube_instance_1.transform.matrix[1][1] = 1.0f;
    cube_instance_1.transform.matrix[2][2] = 1.0f;
    cube_instance_1.transform.matrix[0][3] = 50.0f;
    cube_instance_1.transform.matrix[1][3] = 0.0f;
    cube_instance_1.transform.matrix[2][3] = 0.0f;
    cube_instance_1.mask = 0xff;
    cube_instance_1.instanceCustomIndex = 0;
    // Cube instance 1 will be associated to closest hit shader 1
    cube_instance_1.instanceShaderBindingTableRecordOffset = 0;
    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    VkAccelerationStructureInstanceKHR cube_instance_2{};
    cube_instance_2.transform.matrix[0][0] = 1.0f;
    cube_instance_2.transform.matrix[1][1] = 1.0f;
    cube_instance_2.transform.matrix[2][2] = 1.0f;
    cube_instance_2.transform.matrix[0][3] = 0.0f;
    cube_instance_2.transform.matrix[1][3] = 0.0f;
    cube_instance_2.transform.matrix[2][3] = 50.0f;
    cube_instance_2.mask = 0xff;
    cube_instance_2.instanceCustomIndex = 0;
    // Cube instance 2 will be associated to closest hit shader 1
    cube_instance_2.instanceShaderBindingTableRecordOffset = 0;

    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_2);

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_build_info;
    {
        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(cube_instances));
        tlas_build_info.emplace_back(std::move(tlas));

        // Update cube_blas while it is also referenced in TLAS build
        cube_blas.SetMode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR);
        cube_blas.SetSrcAS(cube_blas.GetDstAS());
        tlas_build_info.emplace_back(std::move(cube_blas));

        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_build_info);
        m_command_buffer.End();

        m_errorMonitor->SetDesiredError("VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03706", 2);
        m_default_queue->Submit(m_command_buffer);
        m_device->Wait();
        m_errorMonitor->VerifyFound();
    }
    // Buffer used to count invocations for the 3 shaders
    vkt::Buffer debug_buffer(*m_device, 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             kHostVisibleMemProps);
    auto debug_buffer_ptr = static_cast<uint32_t *>(debug_buffer.Memory().Map());
    std::memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        [[vk::binding(1, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            InterlockedAdd(debug_buffer[1], 1);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            InterlockedAdd(debug_buffer[2], 1);
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    pipeline.Build();

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_build_info[0].GetDstAS()->handle());
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, debug_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    m_command_buffer.Begin();

    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
    vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                        &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly built, but is empty
    ASSERT_EQ(debug_buffer_ptr[0], 1);  // Ray gen shader invocations count
    ASSERT_EQ(debug_buffer_ptr[1], 1);  // Miss shader invocations count
    ASSERT_EQ(debug_buffer_ptr[2], 0);  // Closest hit shader invocations count
}

// This gets a device lost, because the VUID is only caught in post processing
TEST_F(NegativeGpuAVRayTracing, DISABLED_ShaderRecordAddress) {
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(heap_props);

    const uint32_t shader_record_address_offset_1 = 0u;
    const uint32_t shader_record_address_offset_2 = 64u;

    const VkDeviceSize descriptor_size =
        vk::GetPhysicalDeviceDescriptorSizeEXT(gpu_, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    const VkDeviceSize reserved_offset = Align(descriptor_size, heap_props.resourceHeapAlignment);

    const VkDeviceSize heap_size = reserved_offset + heap_props.minResourceHeapReservedRange;
    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    vkt::rt::Pipeline pipeline(*this, m_device);

    VkDescriptorSetAndBindingMappingEXT mappings[3];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[0].sourceData.constantOffset = {};
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT;
    mappings[1].sourceData.shaderRecordAddressOffset = shader_record_address_offset_1;
    mappings[2] = vku::InitStructHelper();
    mappings[2].descriptorSet = 2u;
    mappings[2].firstBinding = 0u;
    mappings[2].bindingCount = 1u;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT;
    mappings[2].sourceData.shaderRecordAddressOffset = shader_record_address_offset_2;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 3u;
    mapping_info.pMappings = mappings;

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

        layout(location = 0) rayPayloadEXT vec3 hit;

        layout(set = 1, binding = 0) buffer A { uint a; };
        layout(set = 2, binding = 0) buffer B { uint b; };

        void main() {
            b = a;
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen, nullptr, &mapping_info);

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddCreateInfoFlags2(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT);
    pipeline.SetShaderRecordDataSize(256u);
    pipeline.Build();

    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));

    VkDeviceAddress tlas_device_address = tlas.GetDstAS()->GetBufferDeviceAddress();

    VkDeviceAddressRangeEXT device_address_range;
    device_address_range.address = tlas_device_address;
    device_address_range.size = 0;

    VkResourceDescriptorInfoEXT resource_info = vku::InitStructHelper();
    resource_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    resource_info.data.pAddressRange = &device_address_range;

    VkHostAddressRangeEXT descriptor;
    descriptor.address = heap_data;
    descriptor.size = descriptor_size;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &resource_info, &descriptor);

    VkBindHeapInfoEXT bind_heap_info = vku::InitStructHelper();
    bind_heap_info.heapRange.address = heap.Address();
    bind_heap_info.heapRange.size = heap_size;
    bind_heap_info.reservedRangeOffset = reserved_offset;
    bind_heap_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;

    vkt::Buffer buffer1(*m_device, sizeof(uint32_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *data1 = static_cast<uint32_t *>(buffer1.Memory().Map());
    *data1 = 48u;

    vkt::Buffer buffer2(*m_device, sizeof(uint32_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *data2 = static_cast<uint32_t *>(buffer2.Memory().Map());

    VkDeviceAddress address1 = buffer1.Address();
    //VkDeviceAddress address2 = buffer2.Address();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(rt_pipeline_props);

    uint8_t *shader_record_data =
        const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(pipeline.GetTraceRaysSbtBuffer().Memory().Map()));
    uint32_t shader_record_offset = rt_pipeline_props.shaderGroupHandleSize;
    uint32_t offset1 = shader_record_offset + shader_record_address_offset_1;
    //uint32_t offset2 = shader_record_offset + shader_record_address_offset_2;
    memcpy(shader_record_data + offset1, &address1, sizeof(VkDeviceAddress));
    //memcpy(shader_record_data + offset2, &address2, sizeof(VkDeviceAddress));

    m_command_buffer.Begin();
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_heap_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-11319");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();

    ASSERT_EQ(*data1, *data2);
}

TEST_F(NegativeGpuAVRayTracing, ShaderRecordAddressOOB) {
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(heap_props);

    const uint32_t shader_record_address_offset_1 = 32u;
    const uint32_t shader_record_address_offset_2 = 72u;

    const VkDeviceSize descriptor_size =
        vk::GetPhysicalDeviceDescriptorSizeEXT(gpu_, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    const VkDeviceSize reserved_offset = Align(descriptor_size, heap_props.resourceHeapAlignment);

    const VkDeviceSize heap_size = reserved_offset + heap_props.minResourceHeapReservedRange;
    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    vkt::rt::Pipeline pipeline(*this, m_device);

    VkDescriptorSetAndBindingMappingEXT mappings[3];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[0].sourceData.constantOffset = {};
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT;
    mappings[1].sourceData.shaderRecordAddressOffset = shader_record_address_offset_1;
    mappings[2] = vku::InitStructHelper();
    mappings[2].descriptorSet = 2u;
    mappings[2].firstBinding = 0u;
    mappings[2].bindingCount = 1u;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT;
    mappings[2].sourceData.shaderRecordAddressOffset = shader_record_address_offset_2;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 3u;
    mapping_info.pMappings = mappings;

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

        layout(location = 0) rayPayloadEXT vec3 hit;

        layout(set = 1, binding = 0) buffer A { uvec2 a; };
        layout(set = 2, binding = 0) buffer B { uvec2 b; };

        void main() {
            b.y = a.x;
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen, nullptr, &mapping_info);

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddCreateInfoFlags2(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT);
    pipeline.SetShaderRecordDataSize(256u);
    pipeline.Build();

    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));

    VkDeviceAddress tlas_device_address = tlas.GetDstAS()->GetBufferDeviceAddress();

    VkDeviceAddressRangeEXT device_address_range;
    device_address_range.address = tlas_device_address;
    device_address_range.size = 0;

    VkResourceDescriptorInfoEXT resource_info = vku::InitStructHelper();
    resource_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    resource_info.data.pAddressRange = &device_address_range;

    VkHostAddressRangeEXT descriptor;
    descriptor.address = heap_data;
    descriptor.size = descriptor_size;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &resource_info, &descriptor);

    VkBindHeapInfoEXT bind_heap_info = vku::InitStructHelper();
    bind_heap_info.heapRange.address = heap.Address();
    bind_heap_info.heapRange.size = heap_size;
    bind_heap_info.reservedRangeOffset = reserved_offset;
    bind_heap_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;

    vkt::Buffer buffer1(*m_device, sizeof(uint32_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Buffer buffer2(*m_device, sizeof(uint32_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    VkDeviceAddress address1 = buffer1.Address();
    VkDeviceAddress address2 = buffer2.Address();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(rt_pipeline_props);

    uint8_t *shader_record_data =
        const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(pipeline.GetTraceRaysSbtBuffer().Memory().Map()));
    uint32_t shader_record_offset = rt_pipeline_props.shaderGroupHandleSize;
    uint32_t offset1 = shader_record_offset + shader_record_address_offset_1;
    uint32_t offset2 = shader_record_offset + shader_record_address_offset_2;
    memcpy(shader_record_data + offset1, &address1, sizeof(VkDeviceAddress));
    memcpy(shader_record_data + offset2, &address2, sizeof(VkDeviceAddress));

    m_command_buffer.Begin();
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_heap_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-11320");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// Not yet implemented
TEST_F(NegativeGpuAVRayTracing, DISABLED_ShaderRecordData) {
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());

    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(heap_props);

    const uint32_t shader_record_address_offset = 0u;
    const uint32_t shader_record_data_offset = 64u;

    const VkDeviceSize descriptor_size =
        vk::GetPhysicalDeviceDescriptorSizeEXT(gpu_, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    const VkDeviceSize reserved_offset = Align(descriptor_size, heap_props.resourceHeapAlignment);

    const VkDeviceSize heap_size = reserved_offset + heap_props.minResourceHeapReservedRange;
    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    vkt::rt::Pipeline pipeline(*this, m_device);

    VkDescriptorSetAndBindingMappingEXT mappings[3];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[0].sourceData.constantOffset = {};
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT;
    mappings[1].sourceData.shaderRecordAddressOffset = shader_record_address_offset;
    mappings[2] = vku::InitStructHelper();
    mappings[2].descriptorSet = 2u;
    mappings[2].firstBinding = 0u;
    mappings[2].bindingCount = 1u;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_DATA_EXT;
    mappings[2].sourceData.shaderRecordAddressOffset = shader_record_data_offset;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 3u;
    mapping_info.pMappings = mappings;

    const char *ray_gen = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require // Requires SPIR-V 1.5 (Vulkan 1.2)

        layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

        layout(location = 0) rayPayloadEXT vec3 hit;

        layout(set = 1, binding = 0) buffer A { uint a; };
        layout(set = 2, binding = 0) uniform B { uint b; };

        void main() {
            a = b;
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0,0,1), 0.1, vec3(0,0,1), 1000.0, 0);
        }
    )glsl";
    pipeline.SetGlslRayGenShader(ray_gen, nullptr, &mapping_info);

    const char *miss = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;

        void main() {
            hit = vec3(0.1, 0.2, 0.3);
        }
    )glsl";
    pipeline.AddGlslMissShader(miss);

    const char *closest_hit = R"glsl(
        #version 460
        #extension GL_EXT_ray_tracing : require

        layout(location = 0) rayPayloadInEXT vec3 hit;
        hitAttributeEXT vec2 baryCoord;

        void main() {
            const vec3 barycentricCoords = vec3(1.0f - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
            hit = barycentricCoords;
        }
    )glsl";
    pipeline.AddGlslClosestHitShader(closest_hit);

    pipeline.AddCreateInfoFlags2(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT);
    pipeline.SetShaderRecordDataSize(shader_record_data_offset);
    pipeline.Build();

    vkt::as::BuildGeometryInfoKHR tlas(vkt::as::blueprint::BuildOnDeviceTopLevel(*m_device, *m_default_queue, m_command_buffer));

    VkDeviceAddress tlas_device_address = tlas.GetDstAS()->GetBufferDeviceAddress();

    VkDeviceAddressRangeEXT device_address_range;
    device_address_range.address = tlas_device_address;
    device_address_range.size = 0;

    VkResourceDescriptorInfoEXT resource_info = vku::InitStructHelper();
    resource_info.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    resource_info.data.pAddressRange = &device_address_range;

    VkHostAddressRangeEXT descriptor;
    descriptor.address = heap_data;
    descriptor.size = descriptor_size;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &resource_info, &descriptor);

    VkBindHeapInfoEXT bind_heap_info = vku::InitStructHelper();
    bind_heap_info.heapRange.address = heap.Address();
    bind_heap_info.heapRange.size = heap_size;
    bind_heap_info.reservedRangeOffset = reserved_offset;
    bind_heap_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;

    vkt::Buffer buffer(*m_device, sizeof(uint32_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *data = static_cast<uint32_t *>(buffer.Memory().Map());
    *data = 48u;

    VkDeviceAddress address = buffer.Address();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_props = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(rt_pipeline_props);

    uint8_t *shader_record_data =
        const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(pipeline.GetTraceRaysSbtBuffer().Memory().Map()));
    uint32_t shader_record_offset = rt_pipeline_props.shaderGroupHandleSize;
    uint32_t offset1 = shader_record_offset + shader_record_address_offset;
    uint32_t offset2 = shader_record_offset + shader_record_data_offset;
    memcpy(shader_record_data + offset1, &address, sizeof(VkDeviceAddress));
    shader_record_data[offset2] = 123;

    m_command_buffer.Begin();
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_heap_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkt::rt::TraceRaysSbt trace_rays_sbt = pipeline.GetTraceRaysSbt();
    vk::CmdTraceRaysKHR(m_command_buffer, &trace_rays_sbt.ray_gen_sbt, &trace_rays_sbt.miss_sbt, &trace_rays_sbt.hit_sbt,
                        &trace_rays_sbt.callable_sbt, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdTraceRaysKHR-None-11398");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVRayTracing, TLASinBLASlist) {
    TEST_DESCRIPTION("Setup a RT pipeline, create a TLAS that contains a reference to another TLAS");

    RETURN_IF_SKIP(CheckSlangSupport());

    SetTargetApiVersion(VK_API_VERSION_1_2);

    AddRequiredExtensions(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::rayTracingPipeline);
    AddRequiredFeature(vkt::Feature::accelerationStructure);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::maintenance4);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    VkValidationFeaturesEXT validation_features = GetGpuAvValidationFeatures();
    RETURN_IF_SKIP(InitFrameworkForRayTracingTest(&validation_features));
    if (!CanEnableGpuAV(*this)) {
        GTEST_SKIP() << "Requirements for GPU-AV are not met";
    }
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    vkt::as::GeometryKHR cube(vkt::as::blueprint::GeometryCubeOnDeviceInfo(*m_device));
    vkt::as::BuildGeometryInfoKHR cube_blas = vkt::as::blueprint::BuildGeometryInfoOnDeviceBottomLevel(*m_device, std::move(cube));

    m_command_buffer.Begin();
    cube_blas.BuildCmdBuffer(m_command_buffer);
    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    std::vector<vkt::as::GeometryKHR> cube_instances(1);
    cube_instances[0].SetType(vkt::as::GeometryKHR::Type::Instance);

    VkAccelerationStructureInstanceKHR cube_instance_1{};
    cube_instance_1.transform.matrix[0][0] = 1.0f;
    cube_instance_1.transform.matrix[1][1] = 1.0f;
    cube_instance_1.transform.matrix[2][2] = 1.0f;
    cube_instance_1.transform.matrix[0][3] = 50.0f;
    cube_instance_1.transform.matrix[1][3] = 0.0f;
    cube_instance_1.transform.matrix[2][3] = 0.0f;
    cube_instance_1.mask = 0xff;
    cube_instance_1.instanceCustomIndex = 0;
    // Cube instance 1 will be associated to closest hit shader 1
    cube_instance_1.instanceShaderBindingTableRecordOffset = 0;
    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    VkAccelerationStructureInstanceKHR cube_instance_2{};
    cube_instance_2.transform.matrix[0][0] = 1.0f;
    cube_instance_2.transform.matrix[1][1] = 1.0f;
    cube_instance_2.transform.matrix[2][2] = 1.0f;
    cube_instance_2.transform.matrix[0][3] = 0.0f;
    cube_instance_2.transform.matrix[1][3] = 0.0f;
    cube_instance_2.transform.matrix[2][3] = 50.0f;
    cube_instance_2.mask = 0xff;
    cube_instance_2.instanceCustomIndex = 0;
    // Cube instance 2 will be associated to closest hit shader 1
    cube_instance_2.instanceShaderBindingTableRecordOffset = 0;

    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_2);

    // Create a TLAS, and add a reference to it in the cubes_instances BLAS list
    // This list being used to create a TLAS, this is invalid
    std::vector<vkt::as::GeometryKHR> cube_instances_2(1);
    cube_instances_2[0].SetType(vkt::as::GeometryKHR::Type::Instance);
    cube_instances_2[0].AddInstanceDeviceAccelStructRef(*m_device, cube_blas.GetDstAS()->handle(), cube_instance_1);

    vkt::as::BuildGeometryInfoKHR unwanted_tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(cube_instances_2));
    m_command_buffer.Begin();
    unwanted_tlas.BuildCmdBuffer(m_command_buffer);

    m_command_buffer.End();

    m_default_queue->Submit(m_command_buffer);
    m_device->Wait();

    cube_instances[0].AddInstanceDeviceAccelStructRef(*m_device, unwanted_tlas.GetDstAS()->handle(), cube_instance_1);

    std::vector<vkt::as::BuildGeometryInfoKHR> tlas_build_info;
    {
        vkt::as::BuildGeometryInfoKHR tlas = vkt::as::blueprint::CreateTLAS(*m_device, std::move(cube_instances));
        tlas_build_info.emplace_back(std::move(tlas));
        m_command_buffer.Begin();
        vkt::as::BuildAccelerationStructuresKHR(m_command_buffer, tlas_build_info);
        m_command_buffer.End();

        m_default_queue->Submit(m_command_buffer);
        m_errorMonitor->SetDesiredError("UID-vkCmdBuildAccelerationStructuresKHR-pInfos-12281");
        m_device->Wait();
        m_errorMonitor->VerifyFound();
    }

    // Past this point, goal is to make sure using `tlas` is safe, and the TLAS reference found in its BLAS reference list
    // has just been removed and is not causing a crash.

    // Buffer used to count invocations for the 3 shaders
    vkt::Buffer debug_buffer(*m_device, 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             kHostVisibleMemProps);
    auto debug_buffer_ptr = static_cast<uint32_t *>(debug_buffer.Memory().Map());
    std::memset(debug_buffer_ptr, 0, (size_t)debug_buffer.CreateInfo().size);

    const char *slang_shader = R"slang(
        [[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
        [[vk::binding(1, 0)]] RWStructuredBuffer<uint32_t> debug_buffer;

        struct RayPayload {
            uint4 payload;
            float3 hit;
        };

        [shader("raygeneration")]
        void rayGenShader()
        {
            InterlockedAdd(debug_buffer[0], 1);
            RayPayload ray_payload = {};
            RayDesc ray;
            ray.TMin = 0.01;
            ray.TMax = 1000.0;

            // Will hit cube 1
            ray.Origin = float3(0,0,0);
            ray.Direction = float3(1,0,0);
            TraceRay(tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, ray_payload);
        }

        [shader("miss")]
        void missShader(inout RayPayload payload)
        {
            InterlockedAdd(debug_buffer[1], 1);
            payload.hit = float3(0.1, 0.2, 0.3);
        }

        [shader("closesthit")]
        void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
        {
            InterlockedAdd(debug_buffer[2], 1);
            const float3 barycentric_coords = float3(1.0f - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x,
                attr.barycentrics.y);
            payload.hit = barycentric_coords;
        }
    )slang";

    vkt::rt::Pipeline pipeline(*this, m_device);
    pipeline.AddSlangRayGenShader(slang_shader, "rayGenShader");
    pipeline.AddSlangMissShader(slang_shader, "missShader");
    pipeline.AddSlangClosestHitShader(slang_shader, "closestHitShader");

    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0);
    pipeline.AddBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    pipeline.CreateDescriptorSet();

    pipeline.Build();

    pipeline.GetDescriptorSet().WriteDescriptorAccelStruct(0, 1, &tlas_build_info[0].GetDstAS()->handle());
    pipeline.GetDescriptorSet().WriteDescriptorBufferInfo(1, debug_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline.GetDescriptorSet().UpdateDescriptorSets();

    m_command_buffer.Begin();

    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.GetPipelineLayout(), 0, 1,
                              &pipeline.GetDescriptorSet().set_, 0, nullptr);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    vkt::rt::TraceRaysSbt sbt_ray_gen_1 = pipeline.GetTraceRaysSbt(0);
    vk::CmdTraceRaysKHR(m_command_buffer, &sbt_ray_gen_1.ray_gen_sbt, &sbt_ray_gen_1.miss_sbt, &sbt_ray_gen_1.hit_sbt,
                        &sbt_ray_gen_1.callable_sbt, 1, 1, 1);

    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);

    // Make sure expected ray tracing setup worked, indicating the TLAS was correctly built
    ASSERT_EQ(debug_buffer_ptr[0], 1);
    ASSERT_EQ(debug_buffer_ptr[1], 0);
    ASSERT_EQ(debug_buffer_ptr[2], 1);
}
