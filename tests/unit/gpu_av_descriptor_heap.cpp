/*
 * Copyright (c) 2025 The Khronos Group Inc.
 * Copyright (c) 2025 Valve Corporation
 * Copyright (c) 2025 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <vulkan/vulkan_core.h>
#include "../framework/layer_validation_tests.h"
#include "../framework/pipeline_helper.h"
#include "../framework/buffer_helper.h"
#include "../utils/math_utils.h"

void GpuAVDescriptorHeap::InitGpuAVDescriptorHeap(bool safe_mode) {
    SetTargetApiVersion(VK_API_VERSION_1_3);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    RETURN_IF_SKIP(InitGpuAvFramework({}, safe_mode));
    RETURN_IF_SKIP(InitState());
    GetPhysicalDeviceProperties2(heap_props);
}

void GpuAVDescriptorHeap::CreateResourceHeap(VkDeviceSize app_size) {
    const uint32_t heap_size = AlignResource(app_size) + heap_props.minResourceHeapReservedRange;
    VkBufferUsageFlags2CreateInfo buffer_usage = vku::InitStructHelper();
    buffer_usage.usage = VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateFlagsInfo allocate_flag_info = vku::InitStructHelper();
    allocate_flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    resource_heap_.Init(*m_device, vkt::Buffer::CreateInfo(heap_size, 0, {}, &buffer_usage),
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &allocate_flag_info);
    resource_heap_data_ = static_cast<uint8_t *>(resource_heap_.Memory().Map());
}

void GpuAVDescriptorHeap::CreateSamplerHeap(VkDeviceSize app_size, bool use_embedded_samplers) {
    embedded_samplers = use_embedded_samplers;
    const VkDeviceSize reserved_range =
        (embedded_samplers ? heap_props.minSamplerHeapReservedRangeWithEmbedded : heap_props.minSamplerHeapReservedRange);
    const uint32_t heap_size = AlignSampler(app_size + reserved_range);

    VkBufferUsageFlags2CreateInfo buffer_usage = vku::InitStructHelper();
    buffer_usage.usage = VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateFlagsInfo allocate_flag_info = vku::InitStructHelper();
    allocate_flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    sampler_heap_.Init(*m_device, vkt::Buffer::CreateInfo(heap_size, 0, {}, &buffer_usage),
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &allocate_flag_info);
    sampler_heap_data_ = static_cast<uint8_t *>(sampler_heap_.Memory().Map());
}

void GpuAVDescriptorHeap::BindResourceHeap() {
    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = resource_heap_.Address();
    bind_resource_info.heapRange.size = resource_heap_.CreateInfo().size;
    bind_resource_info.reservedRangeOffset = resource_heap_.CreateInfo().size - heap_props.minResourceHeapReservedRange;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);
}

void GpuAVDescriptorHeap::BindSamplerHeap() {
    const VkDeviceSize min_reserved_range =
        embedded_samplers ? heap_props.minSamplerHeapReservedRangeWithEmbedded : heap_props.minSamplerHeapReservedRange;
    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = sampler_heap_.Address();
    bind_resource_info.heapRange.size = sampler_heap_.CreateInfo().size;
    bind_resource_info.reservedRangeOffset = sampler_heap_.CreateInfo().size - min_reserved_range;
    bind_resource_info.reservedRangeSize = min_reserved_range;
    vk::CmdBindSamplerHeapEXT(m_command_buffer, &bind_resource_info);
}

size_t GpuAVDescriptorHeap::AlignResource(size_t offset) {
    return Align(Align(offset, heap_props.bufferDescriptorAlignment), heap_props.imageDescriptorAlignment);
}

size_t GpuAVDescriptorHeap::AlignSampler(size_t offset) { return Align(offset, heap_props.samplerDescriptorAlignment); }

class NegativeGpuAVDescriptorHeap : public GpuAVDescriptorHeap {};

TEST_F(NegativeGpuAVDescriptorHeap, IndexBufferOOB) {
    TEST_DESCRIPTION("Validate overruning the index buffer");
    AddRequiredExtensions(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.gp_ci_.layout = VK_NULL_HANDLE;
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 1;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {1, 2, 3});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    m_errorMonitor->SetDesiredErrorRegex("VUID-VkDrawIndexedIndirectCommand-robustBufferAccess2-08798",
                                         "Index 4 is not within the bound index buffer.");
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, NoMappings) {
    TEST_DESCRIPTION("Validate illegal index buffer values with no VkShaderDescriptorSetAndBindingMappingInfoEXT provided");
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, NoHeapBound) {
    TEST_DESCRIPTION("Validate illegal index buffer values when no descriptor heap is bound");
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.shader_stages_[0].pNext = &mapping_info;
    pipe.shader_stages_[1].pNext = &mapping_info;
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, HeapBoundBeforePipeline) {
    TEST_DESCRIPTION("Validate illegal index buffer values when descriptor heap is bound before the pipeline");
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize descriptor_size = AlignResource(heap_props.bufferDescriptorAlignment * 4u);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.shader_stages_[0].pNext = &mapping_info;
    pipe.shader_stages_[1].pNext = &mapping_info;
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, HeapBoundAfterPipeline) {
    TEST_DESCRIPTION("Validate illegal index buffer values when descriptor heap is bound after pipeline");
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize descriptor_size = AlignResource(heap_props.bufferDescriptorAlignment * 4u);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.shader_stages_[0].pNext = &mapping_info;
    pipe.shader_stages_[1].pNext = &mapping_info;
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);

    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, SamplerHeapBound) {
    TEST_DESCRIPTION("Validate illegal index buffer values when sampler heap is bound");
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize descriptor_size = AlignSampler(heap_props.samplerDescriptorAlignment * 4u);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minSamplerHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.shader_stages_[0].pNext = &mapping_info;
    pipe.shader_stages_[1].pNext = &mapping_info;
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);

    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minSamplerHeapReservedRange;
    vk::CmdBindSamplerHeapEXT(m_command_buffer, &bind_resource_info);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, MappingsUsed) {
    TEST_DESCRIPTION("Validate illegal index buffer values when custom mappings are used");
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize descriptor_size = AlignResource(heap_props.bufferDescriptorAlignment * 4u);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    vkt::Buffer buffer1(*m_device, 256u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);
    vkt::Buffer buffer2(*m_device, 256u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);
    vkt::Buffer buffer3(*m_device, 256u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

    VkDeviceAddressRangeEXT address_ranges[3];
    address_ranges[0].address = buffer1.Address();
    address_ranges[0].size = 256u;
    address_ranges[1].address = buffer2.Address();
    address_ranges[1].size = 256u;
    address_ranges[2].address = buffer3.Address();
    address_ranges[2].size = 256u;

    VkResourceDescriptorInfoEXT resource_infos[3];
    resource_infos[0] = vku::InitStructHelper();
    resource_infos[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resource_infos[0].data.pAddressRange = &address_ranges[0];
    resource_infos[1] = vku::InitStructHelper();
    resource_infos[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resource_infos[1].data.pAddressRange = &address_ranges[1];
    resource_infos[2] = vku::InitStructHelper();
    resource_infos[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resource_infos[2].data.pAddressRange = &address_ranges[2];

    VkHostAddressRangeEXT host_addresses[3];
    host_addresses[0].address = heap_data;
    host_addresses[0].size = heap_props.bufferDescriptorSize;
    host_addresses[1].address = heap_data + heap_props.bufferDescriptorAlignment;
    host_addresses[1].size = heap_props.bufferDescriptorSize;
    host_addresses[2].address = heap_data + heap_props.bufferDescriptorAlignment * 2u;
    host_addresses[2].size = heap_props.bufferDescriptorSize;
    vk::WriteResourceDescriptorsEXT(*m_device, 3u, resource_infos, host_addresses);

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        layout(set=0, binding=0) buffer Buf1 { uint data1[]; } buf1;
        layout(set=0, binding=1) readonly buffer Buf2 { uint data2[]; } buf2;
        layout(set=1, binding=2) buffer Buf3 { uint data3[]; } buf3;
        void main() {
            buf3.data3[gl_VertexIndex] = buf1.data1[gl_VertexIndex] + buf2.data2[gl_VertexIndex];
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkDescriptorSetAndBindingMappingEXT mappings[3];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[0].sourceData.constantOffset.heapOffset = 0;
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 0u;
    mappings[1].firstBinding = 1u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mappings[2] = vku::InitStructHelper();
    mappings[2].descriptorSet = 1u;
    mappings[2].firstBinding = 2u;
    mappings[2].bindingCount = 1u;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[2].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment * 2u);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 3u;
    mapping_info.pMappings = mappings;

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.shader_stages_[0].pNext = &mapping_info;
    pipe.shader_stages_[1].pNext = &mapping_info;
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, DispatchWorkgroupSize) {
    TEST_DESCRIPTION("GPU validation: Validate VkDispatchIndirectCommand with descriptor heap");
    SetTargetApiVersion(VK_API_VERSION_1_3);
    AddRequiredExtensions(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::descriptorHeap);
    RETURN_IF_SKIP(InitGpuAvFramework());

    PFN_vkSetPhysicalDeviceLimitsEXT fpvkSetPhysicalDeviceLimitsEXT = nullptr;
    PFN_vkGetOriginalPhysicalDeviceLimitsEXT fpvkGetOriginalPhysicalDeviceLimitsEXT = nullptr;
    if (!LoadDeviceProfileLayer(fpvkSetPhysicalDeviceLimitsEXT, fpvkGetOriginalPhysicalDeviceLimitsEXT)) {
        GTEST_SKIP() << "Failed to load device profile layer.";
    }

    VkPhysicalDeviceProperties props;
    fpvkGetOriginalPhysicalDeviceLimitsEXT(Gpu(), &props.limits);
    props.limits.maxComputeWorkGroupCount[0] = 2;
    props.limits.maxComputeWorkGroupCount[1] = 2;
    props.limits.maxComputeWorkGroupCount[2] = 2;
    fpvkSetPhysicalDeviceLimitsEXT(Gpu(), &props.limits);

    RETURN_IF_SKIP(InitState());
    GetPhysicalDeviceProperties2(heap_props);

    vkt::Buffer indirect_buffer(*m_device, 5 * sizeof(VkDispatchIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                kHostVisibleMemProps);
    VkDispatchIndirectCommand *ptr = static_cast<VkDispatchIndirectCommand *>(indirect_buffer.Memory().Map());
    // VkDispatchIndirectCommand[0]
    ptr->x = 4;  // over
    ptr->y = 2;
    ptr->z = 1;
    // VkDispatchIndirectCommand[1]
    ptr++;
    ptr->x = 2;
    ptr->y = 3;  // over
    ptr->z = 1;
    // VkDispatchIndirectCommand[2] - valid in between
    ptr++;
    ptr->x = 1;
    ptr->y = 1;
    ptr->z = 1;
    // VkDispatchIndirectCommand[3]
    ptr++;
    ptr->x = 0;  // allowed
    ptr->y = 2;
    ptr->z = 3;  // over
    // VkDispatchIndirectCommand[4]
    ptr++;
    ptr->x = 3;  // over
    ptr->y = 2;
    ptr->z = 3;  // over

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = pipe.cs_.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);

    m_errorMonitor->SetDesiredError("VUID-VkDispatchIndirectCommand-x-00417");
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 0);

    m_errorMonitor->SetDesiredError("VUID-VkDispatchIndirectCommand-y-00418");
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, sizeof(VkDispatchIndirectCommand));

    // valid
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 2 * sizeof(VkDispatchIndirectCommand));

    m_errorMonitor->SetDesiredError("VUID-VkDispatchIndirectCommand-z-00419");
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 3 * sizeof(VkDispatchIndirectCommand));

    // Only expect to have the first error return
    m_errorMonitor->SetDesiredError("VUID-VkDispatchIndirectCommand-x-00417");
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 4 * sizeof(VkDispatchIndirectCommand));

    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    // Check again in a 2nd submitted command buffer
    m_command_buffer.Reset();
    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);

    m_errorMonitor->SetDesiredError("VUID-VkDispatchIndirectCommand-x-00417");
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 0);

    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 2 * sizeof(VkDispatchIndirectCommand));

    m_errorMonitor->SetDesiredError("VUID-VkDispatchIndirectCommand-x-00417");
    vk::CmdDispatchIndirect(m_command_buffer, indirect_buffer, 0);

    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, HeapRebound) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize descriptor_size = AlignResource(heap_props.bufferDescriptorAlignment * 4u);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap1(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    vkt::Buffer heap2(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";
    VkShaderObj vs(*m_device, vsSource, VK_SHADER_STAGE_VERTEX_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.gp_ci_.layout = VK_NULL_HANDLE;

    VkVertexInputBindingDescription input_binding = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription input_attrib = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    pipe.vi_ci_.pVertexBindingDescriptions = &input_binding;
    pipe.vi_ci_.vertexBindingDescriptionCount = 1;
    pipe.vi_ci_.pVertexAttributeDescriptions = &input_attrib;
    pipe.vi_ci_.vertexAttributeDescriptionCount = 1;
    pipe.shader_stages_ = {vs.GetStageCreateInfo(), pipe.fs_->GetStageCreateInfo()};
    pipe.gp_ci_.stageCount = pipe.shader_stages_.size();
    pipe.gp_ci_.pStages = pipe.shader_stages_.data();
    pipe.CreateGraphicsPipeline(false);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap1.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    bind_resource_info.heapRange.address = heap2.Address();
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ShaderObjects) {
    TEST_DESCRIPTION("Validate illegal index buffer values with shader objects");
    AddRequiredExtensions(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderObject);
    AddRequiredFeature(vkt::Feature::dynamicRendering);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitDynamicRenderTarget();

    const VkDeviceSize descriptor_size = AlignResource(heap_props.bufferDescriptorAlignment * 4u);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);

    const char *vsSource = R"glsl(
        #version 450
        layout(location=0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    )glsl";

    const auto vspv = GLSLToSPV(VK_SHADER_STAGE_VERTEX_BIT, vsSource);
    VkShaderCreateInfoEXT vert_ci = vku::InitStructHelper();
    vert_ci.flags = VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;
    vert_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_ci.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vert_ci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    vert_ci.codeSize = vspv.size() * sizeof(vspv[0]);
    vert_ci.pCode = vspv.data();
    vert_ci.pName = "main";

    const auto fspv = GLSLToSPV(VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentMinimalGlsl);
    VkShaderCreateInfoEXT frag_ci = vku::InitStructHelper();
    frag_ci.flags = VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;
    frag_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_ci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    frag_ci.codeSize = fspv.size() * sizeof(fspv[0]);
    frag_ci.pCode = fspv.data();
    frag_ci.pName = "main";

    const vkt::Shader vert_shader(*m_device, vert_ci);
    const vkt::Shader frag_shader(*m_device, frag_ci);

    VkDrawIndexedIndirectCommand draw_params{};
    draw_params.indexCount = 3;
    draw_params.instanceCount = 1;
    draw_params.firstIndex = 0;
    draw_params.vertexOffset = 0;
    draw_params.firstInstance = 0;
    vkt::Buffer draw_params_buffer = vkt::IndirectBuffer<VkDrawIndexedIndirectCommand>(*m_device, {draw_params});

    VkCommandBufferBeginInfo begin_info = vku::InitStructHelper();
    m_command_buffer.Begin(&begin_info);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_command_buffer.BeginRenderingColor(GetDynamicRenderTarget(), GetRenderTargetArea());
    VkShaderStageFlagBits stages[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkShaderEXT shaders[2] = {vert_shader.handle(), frag_shader.handle()};
    vk::CmdBindShadersEXT(m_command_buffer, 2, stages, shaders);

    vk::CmdSetRasterizerDiscardEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetCullModeEXT(m_command_buffer, VK_CULL_MODE_NONE);
    vk::CmdSetDepthTestEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetStencilTestEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetPolygonModeEXT(m_command_buffer, VK_POLYGON_MODE_FILL);
    vk::CmdSetPrimitiveTopology(m_command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vk::CmdSetPrimitiveRestartEnableEXT(m_command_buffer, VK_FALSE);
    VkColorBlendEquationEXT colorBlendEquation = {
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD,
    };
    vk::CmdSetColorBlendEquationEXT(m_command_buffer, 0, 1u, &colorBlendEquation);
    VkViewport viewport = {0.0f, 0.0f, 32.0f, 32.0f, 0.0f, 1.0f};
    vk::CmdSetViewportWithCountEXT(m_command_buffer, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {32, 32}};
    vk::CmdSetScissorWithCountEXT(m_command_buffer, 1, &scissor);
    vk::CmdSetDepthBiasEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetRasterizationSamplesEXT(m_command_buffer, VK_SAMPLE_COUNT_1_BIT);
    VkSampleMask sampleMask = 0xFFFFFFFF;
    vk::CmdSetSampleMaskEXT(m_command_buffer, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    vk::CmdSetAlphaToCoverageEnableEXT(m_command_buffer, VK_FALSE);
    VkBool32 colorBlendEnable = VK_FALSE;
    vk::CmdSetColorBlendEnableEXT(m_command_buffer, 0u, 1u, &colorBlendEnable);
    VkColorComponentFlags colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    vk::CmdSetColorWriteMaskEXT(m_command_buffer, 0u, 1u, &colorWriteMask);

    VkVertexInputBindingDescription2EXT binding = vku::InitStructHelper();
    binding.binding = 0u;
    binding.stride = 3 * sizeof(float);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    binding.divisor = 1u;

    VkVertexInputAttributeDescription2EXT attribute = vku::InitStructHelper();
    attribute.location = 0u;
    attribute.binding = 0u;
    attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute.offset = 0u;
    vk::CmdSetVertexInputEXT(m_command_buffer, 1u, &binding, 1u, &attribute);

    vkt::Buffer index_buffer = vkt::IndexBuffer<uint32_t>(*m_device, {0, 666, 42});
    vkt::Buffer vertex_buffer = vkt::VertexBuffer<float>(*m_device, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    VkDeviceSize vertex_buffer_offset = 0;
    vk::CmdBindIndexBuffer(m_command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vk::CmdBindVertexBuffers(m_command_buffer, 0, 1, &vertex_buffer.handle(), &vertex_buffer_offset);

    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 666");
    m_errorMonitor->SetDesiredErrorRegex("VUID-vkCmdDrawIndexedIndirect-None-02721", "Vertex index 42");
    vk::CmdDrawIndexedIndirect(m_command_buffer, draw_params_buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    m_command_buffer.EndRendering();
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressUniformBufferUsage) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize descriptor_size = AlignResource(write_offset + resource_stride);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = heap_data + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 2u;
    mappings[0].firstBinding = 3u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 2, binding = 3) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[3];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11438");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressStorageBufferUsage) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize descriptor_size = AlignResource(write_offset + resource_stride);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = heap_data + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 2u;
    mappings[0].firstBinding = 3u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 2, binding = 3) buffer ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[3];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11439");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressUnalignedUniformBuffer) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize minUniformBufferOffsetAlignment = m_device->Physical().limits_.minUniformBufferOffsetAlignment;
    if (minUniformBufferOffsetAlignment < 2) {
        GTEST_SKIP() << "minUniformBufferOffsetAlignment is less than 2.";
    }

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 64, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize descriptor_size = AlignResource(write_offset + resource_stride);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = heap_data + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 2u;
    mappings[0].firstBinding = 3u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 2, binding = 3) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[3];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address() + minUniformBufferOffsetAlignment / 2;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11441");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressUnalignedStorageBuffer) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize minStorageBufferOffsetAlignment = m_device->Physical().limits_.minStorageBufferOffsetAlignment;
    if (minStorageBufferOffsetAlignment < 2) {
        GTEST_SKIP() << "minStorageBufferOffsetAlignment is less than 2.";
    }

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 8, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize descriptor_size = AlignResource(write_offset + resource_stride);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = heap_data + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 2u;
    mappings[0].firstBinding = 3u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 2, binding = 3) buffer ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[3];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address() + minStorageBufferOffsetAlignment / 2;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11442");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UnalignedBufferDescriptor) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t push_data_offset = 8u;
    const uint32_t push_offset = 4u;
    const uint32_t heap_offset = 3u;
    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * (push_offset + heap_offset);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mapping.sourceData.pushIndex = {};
    mapping.sourceData.pushIndex.heapOffset = heap_offset * static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.pushIndex.pushOffset = push_data_offset;
    mapping.sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment) + 1;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_data_offset;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11297");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UnalignedImageDescriptor) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t push_offset = 1u;
    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = AlignResource(image_offset + image_size);
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_size = AlignResource(buffer_offset + buffer_size);

    CreateResourceHeap(resource_heap_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = Align(heap_props.minSamplerHeapReservedRange, heap_props.samplerDescriptorAlignment);
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;
    const VkDeviceSize sampler_heap_size = AlignResource(sampler_offset + sampler_size);

    CreateSamplerHeap(sampler_heap_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    char const *cs_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform texture2D tex;
        layout(set = 0, binding = 1) uniform sampler sampl;
        layout(set = 1, binding = 0) buffer ssbo {
            vec4 data;
        };
        void main() {
            data = texture(sampler2D(tex, sampl), vec2(0.5f));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkDescriptorSetAndBindingMappingEXT mappings[3];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mappings[0].sourceData.pushIndex = {};
    mappings[0].sourceData.pushIndex.heapOffset = static_cast<uint32_t>(image_offset);
    mappings[0].sourceData.pushIndex.pushOffset = static_cast<uint32_t>(image_offset);
    mappings[0].sourceData.pushIndex.heapIndexStride =
        static_cast<uint32_t>(heap_props.imageDescriptorAlignment + heap_props.imageDescriptorAlignment / 2u);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 0u;
    mappings[1].firstBinding = 1u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset = {};
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(sampler_offset);
    mappings[2] = vku::InitStructHelper();
    mappings[2].descriptorSet = 1u;
    mappings[2].firstBinding = 0u;
    mappings[2].bindingCount = 1u;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[2].sourceData.constantOffset = {};
    mappings[2].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(buffer_offset);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 3u;
    mapping_info.pMappings = mappings;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.cp_ci_.stage.pNext = &mapping_info;
    pipe.CreateComputePipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = 0u;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();

    vk::CmdPushDataEXT(m_command_buffer, &push_data);

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDispatch-None-11298");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UnalignedSamplerDescriptor) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t push_offset = 1u;
    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = AlignResource(image_offset + image_size);
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_size = AlignResource(buffer_offset + buffer_size);

    CreateResourceHeap(resource_heap_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = Align(heap_props.minSamplerHeapReservedRange, heap_props.samplerDescriptorAlignment);
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;
    const VkDeviceSize sampler_heap_size = AlignResource(sampler_offset + sampler_size);

    CreateSamplerHeap(sampler_heap_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    char const *cs_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform texture2D tex;
        layout(set = 0, binding = 1) uniform sampler sampl;
        layout(set = 1, binding = 0) buffer ssbo {
            vec4 data;
        };
        void main() {
            data = texture(sampler2D(tex, sampl), vec2(0.5f));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkDescriptorSetAndBindingMappingEXT mappings[3];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[0].sourceData.constantOffset = {};
    mappings[0].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(image_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 0u;
    mappings[1].firstBinding = 1u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mappings[1].sourceData.pushIndex = {};
    mappings[1].sourceData.pushIndex.heapOffset = static_cast<uint32_t>(sampler_offset);
    mappings[1].sourceData.pushIndex.pushOffset = 0u;
    mappings[1].sourceData.pushIndex.heapIndexStride =
        static_cast<uint32_t>(heap_props.samplerDescriptorAlignment + heap_props.samplerDescriptorAlignment / 2u);
    mappings[2] = vku::InitStructHelper();
    mappings[2].descriptorSet = 1u;
    mappings[2].firstBinding = 0u;
    mappings[2].bindingCount = 1u;
    mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[2].sourceData.constantOffset = {};
    mappings[2].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(buffer_offset);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 3u;
    mapping_info.pMappings = mappings;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.cp_ci_.stage.pNext = &mapping_info;
    pipe.CreateComputePipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = 0u;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();

    vk::CmdPushDataEXT(m_command_buffer, &push_data);

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDispatch-None-11299");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UnalignedSamplerDescriptorVert) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    AddRequiredFeature(vkt::Feature::fragmentStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t push_offset = 1u;
    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = AlignResource(image_offset + image_size);
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_size = AlignResource(buffer_offset + buffer_size);

    CreateResourceHeap(resource_heap_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = Align(heap_props.minSamplerHeapReservedRange, heap_props.samplerDescriptorAlignment);
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;
    const VkDeviceSize sampler_heap_size = AlignResource(sampler_offset + sampler_size);

    CreateSamplerHeap(sampler_heap_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    char const *vs_source = R"glsl(
        #version 460
        layout(set = 0, binding = 0) uniform texture2D tex;
        layout(set = 0, binding = 1) uniform sampler sampl;
        layout(set = 1, binding = 0) buffer ssbo {
            vec4 data;
        };
        void main() {
           data.xy = texture(sampler2D(tex, sampl), vec2(0.5f)).xy;
           gl_Position = vec4(1);
        }
    )glsl";
    char const *fs_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform texture2D tex;
        layout(set = 0, binding = 1) uniform sampler sampl;
        layout(set = 1, binding = 0) buffer ssbo {
            vec4 data;
        };
        void main() {
            if (gl_FragCoord.x == 0 && gl_FragCoord.y == 0) {
                data.zw = texture(sampler2D(tex, sampl), vec2(0.5f)).zw;
            }
        }
    )glsl";
    VkShaderObj vs_module = VkShaderObj(*m_device, vs_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj fs_module = VkShaderObj(*m_device, fs_source, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkDescriptorSetAndBindingMappingEXT vert_mappings[3];
    vert_mappings[0] = vku::InitStructHelper();
    vert_mappings[0].descriptorSet = 0u;
    vert_mappings[0].firstBinding = 0u;
    vert_mappings[0].bindingCount = 1u;
    vert_mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    vert_mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    vert_mappings[0].sourceData.constantOffset = {};
    vert_mappings[0].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(image_offset);
    vert_mappings[1] = vku::InitStructHelper();
    vert_mappings[1].descriptorSet = 0u;
    vert_mappings[1].firstBinding = 1u;
    vert_mappings[1].bindingCount = 1u;
    vert_mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    vert_mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    vert_mappings[1].sourceData.pushIndex = {};
    vert_mappings[1].sourceData.pushIndex.heapOffset = static_cast<uint32_t>(sampler_offset);
    vert_mappings[1].sourceData.pushIndex.pushOffset = 0u;
    vert_mappings[1].sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(heap_props.samplerDescriptorAlignment);
    vert_mappings[2] = vku::InitStructHelper();
    vert_mappings[2].descriptorSet = 1u;
    vert_mappings[2].firstBinding = 0u;
    vert_mappings[2].bindingCount = 1u;
    vert_mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    vert_mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    vert_mappings[2].sourceData.constantOffset = {};
    vert_mappings[2].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(buffer_offset);

    VkDescriptorSetAndBindingMappingEXT frag_mappings[3];
    for (uint32_t i = 0; i < 3; ++i) {
        frag_mappings[i] = vert_mappings[i];
    }
    vert_mappings[1].sourceData.pushIndex.heapIndexStride =
        static_cast<uint32_t>(heap_props.samplerDescriptorAlignment + heap_props.samplerDescriptorAlignment / 2u);

    VkShaderDescriptorSetAndBindingMappingInfoEXT vert_mapping_info = vku::InitStructHelper();
    vert_mapping_info.mappingCount = 3u;
    vert_mapping_info.pMappings = vert_mappings;

    VkShaderDescriptorSetAndBindingMappingInfoEXT frag_mapping_info = vku::InitStructHelper();
    frag_mapping_info.mappingCount = 3u;
    frag_mapping_info.pMappings = frag_mappings;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vs_module.GetStageCreateInfo();
    stages[0].pNext = &vert_mapping_info;
    stages[1] = fs_module.GetStageCreateInfo();
    stages[1].pNext = &frag_mapping_info;

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.gp_ci_.layout = VK_NULL_HANDLE;
    pipe.gp_ci_.stageCount = 2u;
    pipe.gp_ci_.pStages = stages;
    pipe.CreateGraphicsPipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = 0u;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();

    vk::CmdPushDataEXT(m_command_buffer, &push_data);

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);
    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11299");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UnalignedSamplerDescriptorFrag) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    AddRequiredFeature(vkt::Feature::fragmentStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t push_offset = 1u;
    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = AlignResource(image_offset + image_size);
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_size = AlignResource(buffer_offset + buffer_size);

    CreateResourceHeap(resource_heap_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = Align(heap_props.minSamplerHeapReservedRange, heap_props.samplerDescriptorAlignment);
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;
    const VkDeviceSize sampler_heap_size = AlignResource(sampler_offset + sampler_size);

    CreateSamplerHeap(sampler_heap_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    char const *vs_source = R"glsl(
        #version 460
        layout(set = 0, binding = 0) uniform texture2D tex;
        layout(set = 0, binding = 1) uniform sampler sampl;
        layout(set = 1, binding = 0) buffer ssbo {
            vec4 data;
        };
        void main() {
           data.xy = texture(sampler2D(tex, sampl), vec2(0.5f)).xy;
           gl_Position = vec4(1);
        }
    )glsl";
    char const *fs_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform texture2D tex;
        layout(set = 0, binding = 1) uniform sampler sampl;
        layout(set = 1, binding = 0) buffer ssbo {
            vec4 data;
        };
        void main() {
            if (gl_FragCoord.x == 0 && gl_FragCoord.y == 0) {
                data.zw = texture(sampler2D(tex, sampl), vec2(0.5f)).zw;
            }
        }
    )glsl";
    VkShaderObj vs_module = VkShaderObj(*m_device, vs_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj fs_module = VkShaderObj(*m_device, fs_source, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkDescriptorSetAndBindingMappingEXT vert_mappings[3];
    vert_mappings[0] = vku::InitStructHelper();
    vert_mappings[0].descriptorSet = 0u;
    vert_mappings[0].firstBinding = 0u;
    vert_mappings[0].bindingCount = 1u;
    vert_mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
    vert_mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    vert_mappings[0].sourceData.constantOffset = {};
    vert_mappings[0].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(image_offset);
    vert_mappings[1] = vku::InitStructHelper();
    vert_mappings[1].descriptorSet = 0u;
    vert_mappings[1].firstBinding = 1u;
    vert_mappings[1].bindingCount = 1u;
    vert_mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    vert_mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    vert_mappings[1].sourceData.pushIndex = {};
    vert_mappings[1].sourceData.pushIndex.heapOffset = static_cast<uint32_t>(sampler_offset);
    vert_mappings[1].sourceData.pushIndex.pushOffset = 0u;
    vert_mappings[1].sourceData.pushIndex.heapIndexStride =
        static_cast<uint32_t>(heap_props.samplerDescriptorAlignment + heap_props.samplerDescriptorAlignment / 2);
    vert_mappings[2] = vku::InitStructHelper();
    vert_mappings[2].descriptorSet = 1u;
    vert_mappings[2].firstBinding = 0u;
    vert_mappings[2].bindingCount = 1u;
    vert_mappings[2].resourceMask = VK_SPIRV_RESOURCE_TYPE_ALL_EXT;
    vert_mappings[2].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    vert_mappings[2].sourceData.constantOffset = {};
    vert_mappings[2].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(buffer_offset);

    VkDescriptorSetAndBindingMappingEXT frag_mappings[3];
    for (uint32_t i = 0; i < 3; ++i) {
        frag_mappings[i] = vert_mappings[i];
    }
    frag_mappings[1].sourceData.pushIndex.heapIndexStride =
        static_cast<uint32_t>(heap_props.samplerDescriptorAlignment + heap_props.samplerDescriptorAlignment / 2u);

    VkShaderDescriptorSetAndBindingMappingInfoEXT vert_mapping_info = vku::InitStructHelper();
    vert_mapping_info.mappingCount = 3u;
    vert_mapping_info.pMappings = vert_mappings;

    VkShaderDescriptorSetAndBindingMappingInfoEXT frag_mapping_info = vku::InitStructHelper();
    frag_mapping_info.mappingCount = 3u;
    frag_mapping_info.pMappings = frag_mappings;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vs_module.GetStageCreateInfo();
    stages[0].pNext = &vert_mapping_info;
    stages[1] = fs_module.GetStageCreateInfo();
    stages[1].pNext = &frag_mapping_info;

    CreatePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.gp_ci_.layout = VK_NULL_HANDLE;
    pipe.gp_ci_.stageCount = 2u;
    pipe.gp_ci_.pStages = stages;
    pipe.CreateGraphicsPipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = 0u;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();

    vk::CmdPushDataEXT(m_command_buffer, &push_data);

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);
    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11299");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, MappingSourceHeapWithIndirectIndex) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * 7u;
    const uint32_t push_offset = 8u;
    const uint32_t address_offset = 16u;
    vkt::Buffer heap_index(*m_device, sizeof(uint32_t) + address_offset, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, vkt::device_address);
    uint32_t *heap_index_data = static_cast<uint32_t *>(heap_index.Memory().Map());
    heap_index_data[address_offset / sizeof(uint32_t)] = static_cast<uint32_t>(offset / heap_props.bufferDescriptorAlignment);

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT;
    mapping.sourceData.indirectIndex = {};
    mapping.sourceData.indirectIndex.heapOffset = 0u;
    mapping.sourceData.indirectIndex.pushOffset = push_offset;
    mapping.sourceData.indirectIndex.addressOffset = address_offset;
    mapping.sourceData.indirectIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment) + 1u;
    mapping.sourceData.indirectIndex.heapArrayStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress heap_index_address = heap_index.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_offset;
    push_data.data.address = &heap_index_address;
    push_data.data.size = sizeof(heap_index_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11297");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UnalignedIndirectBuffer) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * 7u;
    const uint32_t push_offset = 8u;
    const uint32_t address_offset = 16u;
    vkt::Buffer heap_index(*m_device, sizeof(uint32_t) + address_offset, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, vkt::device_address);
    uint32_t *heap_index_data = static_cast<uint32_t *>(heap_index.Memory().Map());
    heap_index_data[address_offset / sizeof(uint32_t)] = static_cast<uint32_t>(offset / heap_props.bufferDescriptorAlignment);

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT;
    mapping.sourceData.indirectIndex = {};
    mapping.sourceData.indirectIndex.heapOffset = 0u;
    mapping.sourceData.indirectIndex.pushOffset = push_offset;
    mapping.sourceData.indirectIndex.addressOffset = address_offset;
    mapping.sourceData.indirectIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.indirectIndex.heapArrayStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress heap_index_address = heap_index.Address() + 2;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_offset;
    push_data.data.address = &heap_index_address;
    push_data.data.size = sizeof(heap_index_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11300");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, MappingSourceIndirectAddress) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t read_offset = 48u;
    const uint32_t indirect_offset = 288u;

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Buffer indirect_buffer(*m_device, sizeof(uint32_t) * 4 + indirect_offset, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR,
                                vkt::device_address);
    uint8_t *indirect_data = static_cast<uint8_t *>(indirect_buffer.Memory().Map());
    VkDeviceAddress *indirect_data_address = reinterpret_cast<VkDeviceAddress *>(indirect_data + indirect_offset);
    *indirect_data_address = read_buffer.Address();

    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mappings[0].sourceData.indirectAddress.pushOffset = static_cast<uint32_t>(read_offset);
    mappings[0].sourceData.indirectAddress.addressOffset = static_cast<uint32_t>(indirect_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress indirect_address = indirect_buffer.Address() + 4;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &indirect_address;
    push_data.data.size = sizeof(indirect_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11304");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, IndirectAddressBufferUsage) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t read_offset = 48u;
    const uint32_t indirect_offset = 288u;

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             vkt::device_address);
    vkt::Buffer indirect_buffer(*m_device, sizeof(uint32_t) * 4 + indirect_offset, VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT,
                                vkt::device_address);
    uint8_t *indirect_data = static_cast<uint8_t *>(indirect_buffer.Memory().Map());
    VkDeviceAddress *indirect_data_address = reinterpret_cast<VkDeviceAddress *>(indirect_data + indirect_offset);
    *indirect_data_address = read_buffer.Address();

    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mappings[0].sourceData.indirectAddress.pushOffset = static_cast<uint32_t>(read_offset);
    mappings[0].sourceData.indirectAddress.addressOffset = static_cast<uint32_t>(indirect_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress indirect_address = indirect_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &indirect_address;
    push_data.data.size = sizeof(indirect_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11437");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, IndirectIndexInvalidAddress) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = 0;
    const uint32_t push_offset = 0u;
    const uint32_t address_offset = 0u;
    vkt::Buffer heap_index(*m_device, sizeof(uint32_t) + address_offset, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, vkt::device_address);
    uint32_t *heap_index_data = static_cast<uint32_t *>(heap_index.Memory().Map());
    heap_index_data[address_offset / sizeof(uint32_t)] = static_cast<uint32_t>(offset / heap_props.bufferDescriptorAlignment);

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT;
    mapping.sourceData.indirectIndex = {};
    mapping.sourceData.indirectIndex.heapOffset = 0u;
    mapping.sourceData.indirectIndex.pushOffset = push_offset;
    mapping.sourceData.indirectIndex.addressOffset = address_offset;
    mapping.sourceData.indirectIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.indirectIndex.heapArrayStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress heap_index_address = heap_index.Address() - 8;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_offset;
    push_data.data.address = &heap_index_address;
    push_data.data.size = sizeof(heap_index_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11301");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressInvalidAddress) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
                write_data[1] = read_data[1];
                write_data[2] = read_data[2];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address() - 64;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11302");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// Test results in a VK_ERROR_DEVICE_LOST and VUID is only reported in post processing
TEST_F(NegativeGpuAVDescriptorHeap, DISABLED_IndirectAddressInvalidPushAddress) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t read_offset = 96u;
    const uint32_t indirect_offset = 88u;

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Buffer indirect_buffer(*m_device, sizeof(uint32_t) * 64, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint8_t *indirect_data = static_cast<uint8_t *>(indirect_buffer.Memory().Map());
    VkDeviceAddress *indirect_data_address = reinterpret_cast<VkDeviceAddress *>(indirect_data + indirect_offset);
    *indirect_data_address = read_buffer.Address();

    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mappings[0].sourceData.indirectAddress.pushOffset = static_cast<uint32_t>(read_offset);
    mappings[0].sourceData.indirectAddress.addressOffset = static_cast<uint32_t>(indirect_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
                write_data[1] = read_data[1];
                write_data[2] = read_data[2];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress indirect_address = indirect_buffer.Address() - 8;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &indirect_address;
    push_data.data.size = sizeof(indirect_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11305");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, IndirectAddressInvalidAddress) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t read_offset = 96u;
    const uint32_t indirect_offset = 88u;

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Buffer indirect_buffer(*m_device, sizeof(uint32_t) * 64, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint8_t *indirect_data = static_cast<uint8_t *>(indirect_buffer.Memory().Map());
    VkDeviceAddress *indirect_data_address = reinterpret_cast<VkDeviceAddress *>(indirect_data + indirect_offset);
    *indirect_data_address = read_buffer.Address() - 64;

    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mappings[0].sourceData.indirectAddress.pushOffset = static_cast<uint32_t>(read_offset);
    mappings[0].sourceData.indirectAddress.addressOffset = static_cast<uint32_t>(indirect_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
                write_data[1] = read_data[1];
                write_data[2] = read_data[2];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress indirect_address = indirect_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &indirect_address;
    push_data.data.size = sizeof(indirect_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11306");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// Currently results in a VK_ERROR_DEVICE_LOST
TEST_F(NegativeGpuAVDescriptorHeap, DISABLED_BufferPointerOffset) {
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(resource_stride);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    const std::string spirv = R"(
               OpCapability Shader
               OpCapability UntypedPointersKHR
               OpCapability DescriptorHeapEXT
               OpExtension "SPV_EXT_descriptor_heap"
               OpExtension "SPV_KHR_untyped_pointers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %7
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_descriptor_heap"
               OpName %4 "main"
               OpName %7 "resource_heap"
               OpName %12 "ssbo"
               OpMemberName %12 0 "data"
               OpDecorate %7 BuiltIn ResourceHeapEXT
               OpDecorate %12 Block
               OpMemberDecorate %12 0 BuiltIn ResourceHeapEXT
               OpMemberDecorate %12 0 Offset 0
               OpDecorateId %20 ArrayStrideIdEXT %19
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeUntypedPointerKHR UniformConstant
          %7 = OpUntypedVariableKHR %6 UniformConstant
          %8 = OpTypeInt 32 1
          %9 = OpConstant %8 1
         %10 = OpTypeFloat 32
         %11 = OpTypeVector %10 4
         %12 = OpTypeStruct %11
         %13 = OpConstant %8 0
         %14 = OpConstant %10 1
         %15 = OpConstantComposite %11 %14 %14 %14 %14
         %16 = OpTypeUntypedPointerKHR StorageBuffer
         %18 = OpTypeBufferEXT StorageBuffer
         %19 = OpConstant %8 )" +
                              std::to_string(heap_props.bufferDescriptorAlignment / 2) + R"(
         %20 = OpTypeRuntimeArray %18
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %17 = OpUntypedAccessChainKHR %16 %20 %7 %9
         %21 = OpBufferPointerEXT %16 %17
         %22 = OpUntypedAccessChainKHR %16 %12 %21 %13
               OpStore %22 %15
               OpReturn
               OpFunctionEnd
    )";

    VkShaderObj cs_module = VkShaderObj::CreateFromASM(this, spirv.c_str(), VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-bufferDescriptorAlignment-11384");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, SamplerPointerOffset) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t buffer_index = 16u;

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = heap_props.bufferDescriptorAlignment * buffer_index;
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_app_size = buffer_offset + buffer_size;

    CreateResourceHeap(resource_heap_app_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = 0u;
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorAlignment;

    CreateSamplerHeap(sampler_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    const std::string spirv = R"(
               OpCapability Shader
               OpCapability UntypedPointersKHR
               OpCapability DescriptorHeapEXT
               OpExtension "SPV_EXT_descriptor_heap"
               OpExtension "SPV_KHR_untyped_pointers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %7 %20
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_descriptor_heap"
               OpName %4 "main"
               OpName %7 "resource_heap"
               OpName %12 "ssbo"
               OpMemberName %12 0 "data"
               OpName %20 "sampler_heap"
               OpDecorate %7 BuiltIn ResourceHeapEXT
               OpDecorate %12 Block
               OpMemberDecorate %12 0 BuiltIn ResourceHeapEXT
               OpMemberDecorate %12 0 Offset 0
               OpDecorateId %18 ArrayStrideIdEXT %17
               OpDecorate %20 BuiltIn SamplerHeapEXT
               OpDecorateId %25 ArrayStrideIdEXT %24
               OpDecorateId %38 ArrayStrideIdEXT %37
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeUntypedPointerKHR UniformConstant
          %7 = OpUntypedVariableKHR %6 UniformConstant
          %8 = OpTypeInt 32 1
          %9 = OpConstant %8 16
         %10 = OpTypeFloat 32
         %11 = OpTypeVector %10 4
         %12 = OpTypeStruct %11
         %13 = OpConstant %8 0
         %14 = OpTypeImage %10 2D 0 0 0 1 Unknown
         %15 = OpTypeUntypedPointerKHR Uniform
         %17 = OpConstantSizeOfEXT %8 %14
         %18 = OpTypeRuntimeArray %14
         %20 = OpUntypedVariableKHR %6 UniformConstant
         %21 = OpConstant %8 1
         %22 = OpTypeSampler
         %24 = OpConstant %8 )" +
                              std::to_string(heap_props.samplerDescriptorAlignment / 2) + R"(
         %25 = OpTypeRuntimeArray %22
         %27 = OpTypeSampledImage %14
         %29 = OpTypeVector %10 2
         %30 = OpConstant %10 0.5
         %31 = OpConstantComposite %29 %30 %30
         %32 = OpConstant %10 0
         %34 = OpTypeUntypedPointerKHR StorageBuffer
         %36 = OpTypeBufferEXT StorageBuffer
         %37 = OpConstantSizeOfEXT %8 %36
         %38 = OpTypeRuntimeArray %36
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %16 = OpUntypedAccessChainKHR %15 %18 %7 %13
         %19 = OpLoad %14 %16
         %23 = OpUntypedAccessChainKHR %15 %25 %20 %21
         %26 = OpLoad %22 %23
         %28 = OpSampledImage %27 %19 %26
         %33 = OpImageSampleExplicitLod %11 %28 %31 Lod %32
         %35 = OpUntypedAccessChainKHR %34 %38 %7 %9
         %39 = OpBufferPointerEXT %34 %35
         %40 = OpUntypedAccessChainKHR %34 %12 %39 %13
               OpStore %40 %33
               OpReturn
               OpFunctionEnd
    )";

    VkShaderObj cs_module = VkShaderObj::CreateFromASM(this, spirv.c_str(), VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-samplerDescriptorAlignment-11348");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ImagePointerOffset) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t buffer_index = 16u;

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = heap_props.bufferDescriptorAlignment * buffer_index;
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_app_size = buffer_offset + buffer_size;

    CreateResourceHeap(resource_heap_app_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = 0u;
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorAlignment;

    CreateSamplerHeap(sampler_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    const std::string spirv = R"(
               OpCapability Shader
               OpCapability UntypedPointersKHR
               OpCapability DescriptorHeapEXT
               OpExtension "SPV_EXT_descriptor_heap"
               OpExtension "SPV_KHR_untyped_pointers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %7 %21
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_descriptor_heap"
               OpName %4 "main"
               OpName %7 "resource_heap"
               OpName %12 "ssbo"
               OpMemberName %12 0 "data"
               OpName %21 "sampler_heap"
               OpDecorate %7 BuiltIn ResourceHeapEXT
               OpDecorate %12 Block
               OpMemberDecorate %12 0 BuiltIn ResourceHeapEXT
               OpMemberDecorate %12 0 Offset 0
               OpDecorateId %19 ArrayStrideIdEXT %18
               OpDecorate %21 BuiltIn SamplerHeapEXT
               OpDecorateId %25 ArrayStrideIdEXT %24
               OpDecorateId %38 ArrayStrideIdEXT %37
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeUntypedPointerKHR UniformConstant
          %7 = OpUntypedVariableKHR %6 UniformConstant
          %8 = OpTypeInt 32 1
          %9 = OpConstant %8 16
         %10 = OpTypeFloat 32
         %11 = OpTypeVector %10 4
         %12 = OpTypeStruct %11
         %13 = OpConstant %8 0
         %14 = OpConstant %8 1
         %15 = OpTypeImage %10 2D 0 0 0 1 Unknown
         %16 = OpTypeUntypedPointerKHR Uniform
         %18 = OpConstant %8 )" +
                              std::to_string(heap_props.imageDescriptorAlignment / 2) + R"(
         %19 = OpTypeRuntimeArray %15
         %21 = OpUntypedVariableKHR %6 UniformConstant
         %22 = OpTypeSampler
         %24 = OpConstantSizeOfEXT %8 %22
         %25 = OpTypeRuntimeArray %22
         %27 = OpTypeSampledImage %15
         %29 = OpTypeVector %10 2
         %30 = OpConstant %10 0.5
         %31 = OpConstantComposite %29 %30 %30
         %32 = OpConstant %10 0
         %34 = OpTypeUntypedPointerKHR StorageBuffer
         %36 = OpTypeBufferEXT StorageBuffer
         %37 = OpConstantSizeOfEXT %8 %36
         %38 = OpTypeRuntimeArray %36
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %17 = OpUntypedAccessChainKHR %16 %19 %7 %14
         %20 = OpLoad %15 %17
         %23 = OpUntypedAccessChainKHR %16 %25 %21 %13
         %26 = OpLoad %22 %23
         %28 = OpSampledImage %27 %20 %26
         %33 = OpImageSampleExplicitLod %11 %28 %31 Lod %32
         %35 = OpUntypedAccessChainKHR %34 %38 %7 %9
         %39 = OpBufferPointerEXT %34 %35
         %40 = OpUntypedAccessChainKHR %34 %12 %39 %13
               OpStore %40 %33
               OpReturn
               OpFunctionEnd
    )";

    VkShaderObj cs_module = VkShaderObj::CreateFromASM(this, spirv.c_str(), VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-imageDescriptorAlignment-11349");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ImageTexelPointerOffset) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = 1u;
    const VkDeviceSize resource_size = heap_props.imageDescriptorSize;

    CreateResourceHeap(resource_size);

    VkDeviceSize buffer_size = sizeof(uint32_t) * 4;
    vkt::Buffer texel_buffer(*m_device, buffer_size, VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, vkt::device_address);

    VkHostAddressRangeEXT resource_host;
    resource_host.address = resource_heap_data_ + offset * heap_props.imageDescriptorAlignment;
    resource_host.size = resource_size;

    VkTexelBufferDescriptorInfoEXT texel_buffer_info = vku::InitStructHelper();
    texel_buffer_info.format = VK_FORMAT_R32_UINT;
    texel_buffer_info.addressRange.address = texel_buffer.Address();
    texel_buffer_info.addressRange.size = buffer_size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    descriptor_info.data.pTexelBuffer = &texel_buffer_info;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &resource_host);

    const std::string spirv = R"(
               OpCapability Shader
               OpCapability UntypedPointersKHR
               OpCapability DescriptorHeapEXT
               OpExtension "SPV_EXT_descriptor_heap"
               OpExtension "SPV_KHR_untyped_pointers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %19 %13
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_descriptor_heap"
               OpName %4 "main"
               OpName %9 "coord"
               OpName %13 "gl_GlobalInvocationID"
               OpName %19 "resource_heap"
               OpDecorate %13 BuiltIn GlobalInvocationId
               OpDecorate %19 BuiltIn ResourceHeapEXT
               OpDecorateId %25 ArrayStrideIdEXT %24
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeVector %6 2
          %8 = OpTypePointer Function %7
         %10 = OpTypeInt 32 0
         %11 = OpTypeVector %10 3
         %12 = OpTypePointer Input %11
         %13 = OpVariable %12 Input
         %14 = OpTypeVector %10 2
         %18 = OpTypeUntypedPointerKHR UniformConstant
         %19 = OpUntypedVariableKHR %18 UniformConstant
         %20 = OpConstant %6 1
         %21 = OpTypeImage %10 2D 0 0 0 2 R32ui
         %22 = OpTypeUntypedPointerKHR Uniform
         %24 = OpConstant %6 )" +
                              std::to_string(heap_props.imageDescriptorAlignment / 2) + R"(
         %25 = OpTypeRuntimeArray %21
         %27 = OpConstant %10 1
         %28 = OpConstant %10 0
         %29 = OpTypePointer Image %10
         %30 = OpTypeUntypedPointerKHR Image
         %33 = OpConstantComposite %11 %27 %27 %27
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %15 = OpLoad %11 %13
         %16 = OpVectorShuffle %14 %15 %15 0 1
         %17 = OpBitcast %7 %16
               OpStore %9 %17
         %23 = OpUntypedAccessChainKHR %22 %25 %19 %20
         %26 = OpLoad %7 %9
         %31 = OpUntypedImageTexelPointerEXT %30 %21 %23 %26 %28
         %32 = OpAtomicIAdd %10 %31 %27 %28 %27
               OpReturn
               OpFunctionEnd
    )";

    VkShaderObj cs_module = VkShaderObj::CreateFromASM(this, spirv.c_str(), VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-imageDescriptorAlignment-11383");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressProtectedBuffer) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    AddRequiredFeature(vkt::Feature::protectedMemory);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    VkPhysicalDeviceVulkan11Properties props11 = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(props11);
    if (props11.protectedNoFault) {
        GTEST_SKIP() << "Test requires protectedNoFault to not be supported.";
    }

    VkBufferCreateInfo buffer_ci = vku::InitStructHelper();
    buffer_ci.flags = VK_BUFFER_CREATE_PROTECTED_BIT;
    buffer_ci.size = sizeof(uint32_t) * 4;
    buffer_ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateFlagsInfo allocate_flag_info = vku::InitStructHelper();
    allocate_flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    vkt::Buffer read_buffer(*m_device, buffer_ci, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            &allocate_flag_info);
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
                write_data[1] = read_data[1];
                write_data[2] = read_data[2];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-protectedNoFault-11455");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, IndirectIndexProtectedBuffer) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    AddRequiredFeature(vkt::Feature::protectedMemory);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    VkPhysicalDeviceVulkan11Properties props11 = vku::InitStructHelper();
    GetPhysicalDeviceProperties2(props11);
    if (props11.protectedNoFault) {
        GTEST_SKIP() << "Test requires protectedNoFault to not be supported.";
    }

    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * 7u;
    const uint32_t push_offset = 8u;
    const uint32_t address_offset = 16u;
    vkt::Buffer heap_index(*m_device, sizeof(uint32_t) + address_offset, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, vkt::device_address);
    uint32_t *heap_index_data = static_cast<uint32_t *>(heap_index.Memory().Map());
    heap_index_data[address_offset / sizeof(uint32_t)] = static_cast<uint32_t>(offset / heap_props.bufferDescriptorAlignment);

    VkBufferCreateInfo buffer_ci = vku::InitStructHelper();
    buffer_ci.flags = VK_BUFFER_CREATE_PROTECTED_BIT;
    buffer_ci.size = sizeof(uint32_t) * 4;
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateFlagsInfo allocate_flag_info = vku::InitStructHelper();
    allocate_flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    vkt::Buffer buffer(*m_device, buffer_ci, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &allocate_flag_info);

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT;
    mapping.sourceData.indirectIndex = {};
    mapping.sourceData.indirectIndex.heapOffset = 0u;
    mapping.sourceData.indirectIndex.pushOffset = push_offset;
    mapping.sourceData.indirectIndex.addressOffset = address_offset;
    mapping.sourceData.indirectIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.indirectIndex.heapArrayStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress heap_index_address = heap_index.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_offset;
    push_data.data.address = &heap_index_address;
    push_data.data.size = sizeof(heap_index_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-protectedNoFault-11456");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// Currently gpuav does s vkCmdCopyBuffer inside a render pass, which is not legal
TEST_F(NegativeGpuAVDescriptorHeap, DISABLED_DeviceLocalHeap) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    AddRequiredFeature(vkt::Feature::synchronization2);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    uint32_t *read_data = static_cast<uint32_t *>(read_buffer.Memory().Map());
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize descriptor_size = AlignResource(write_offset + resource_stride);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT;
    VkMemoryAllocateFlagsInfo allocate_flag_info = vku::InitStructHelper();
    allocate_flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    vkt::Buffer heap(*m_device, vkt::Buffer::CreateInfo(heap_size, usage), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     &allocate_flag_info);

    vkt::Buffer transfer_buffer(*m_device, descriptor_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(transfer_buffer.Memory().Map());

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = heap_data + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 2u;
    mappings[0].firstBinding = 3u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 2, binding = 3) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[3];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    VkBufferCopy copy;
    copy.srcOffset = 0u;
    copy.dstOffset = 0u;
    copy.size = descriptor_size;
    vk::CmdCopyBuffer(m_command_buffer, transfer_buffer, heap, 1u, &copy);
    VkMemoryBarrier2 memory_barrier = vku::InitStructHelper();
    memory_barrier.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    memory_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory_barrier.dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT;
    VkDependencyInfo dependency_info = vku::InitStructHelper();
    dependency_info.memoryBarrierCount = 1u;
    dependency_info.pMemoryBarriers = &memory_barrier;
    vk::CmdPipelineBarrier2(m_command_buffer, &dependency_info);
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11438");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, DISABLED_ResourceHeapInvalidBuffer) {
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const std::vector<uint32_t> offsets = {0u, 77u, 133u};

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize heap_size = resource_stride * (offsets.back() + 1);
    CreateResourceHeap(heap_size);
    for (const auto offset : offsets) {
        vkt::Buffer uniform_buffer(*m_device, sizeof(float) * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, vkt::device_address);

        VkHostAddressRangeEXT descriptor_host;
        descriptor_host.address = resource_heap_data_ + resource_stride * offset;
        descriptor_host.size = resource_stride;

        VkDeviceAddressRangeEXT device_range;
        device_range.address = uniform_buffer.Address();
        device_range.size = uniform_buffer.CreateInfo().size;

        VkResourceDescriptorInfoEXT descriptor_info;
        descriptor_info = vku::InitStructHelper();
        descriptor_info.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_info.data.pAddressRange = &device_range;
        vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

        std::string cs_source = R"glsl(
            #version 450
            #extension GL_EXT_descriptor_heap : require
            layout(local_size_x = 1) in;
            layout(descriptor_heap) buffer A { vec4 data; } heap[];
            void main() {
                heap[)glsl" + std::to_string(offset) +
                                R"glsl(].data = vec4(1.0f, 2.0f, 3.0f, 4.0f);
            }
        )glsl";
        VkShaderObj cs_module = VkShaderObj(*m_device, cs_source.c_str(), VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

        VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
        pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

        CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
        pipe.cp_ci_.layout = VK_NULL_HANDLE;
        pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
        pipe.CreateComputePipeline(false);

        m_command_buffer.Begin();
        vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        BindResourceHeap();
        vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
        m_command_buffer.End();

        m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11342");
        m_default_queue->SubmitAndWait(m_command_buffer);
        m_errorMonitor->VerifyFound();

        vkt::Buffer storage_buffer(*m_device, sizeof(float) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

        device_range.address = storage_buffer.Address();
        device_range.size = storage_buffer.CreateInfo().size;

        descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_info.data.pAddressRange = &device_range;
        vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

        m_default_queue->SubmitAndWait(m_command_buffer);

        storage_buffer.Destroy();

        m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11342");
        m_default_queue->SubmitAndWait(m_command_buffer);
        m_errorMonitor->VerifyFound();
    }
}

TEST_F(NegativeGpuAVDescriptorHeap, SamplerHeapInvalidDescriptor) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t buffer_index = 16u;

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = heap_props.bufferDescriptorAlignment * buffer_index;
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_app_size = buffer_offset + buffer_size;

    CreateResourceHeap(resource_heap_app_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = 0u;
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;

    CreateSamplerHeap(sampler_size);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(descriptor_heap) uniform texture2D heapTextures[];
        layout(descriptor_heap) uniform sampler heapSamplers[];
        layout(descriptor_heap) buffer ssbo {
            vec4 data;
        } heapBuffer[];
        void main() {
            heapBuffer[16].data = texture(sampler2D(heapTextures[0], heapSamplers[0]), vec2(0.5f));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11340");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);
    m_default_queue->SubmitAndWait(m_command_buffer);
}

TEST_F(NegativeGpuAVDescriptorHeap, ResourceHeapInvalidImage) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t buffer_index = 16u;

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = heap_props.bufferDescriptorAlignment * buffer_index;
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_app_size = buffer_offset + buffer_size;

    CreateResourceHeap(resource_heap_app_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = 0u;
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;

    CreateSamplerHeap(sampler_size);

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo();

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(descriptor_heap) uniform texture2D heapTextures[];
        layout(descriptor_heap) uniform sampler heapSamplers[];
        layout(descriptor_heap) buffer ssbo {
            vec4 data;
        } heapBuffer[];
        void main() {
            heapBuffer[16].data = texture(sampler2D(heapTextures[0], heapSamplers[0]), vec2(0.5f));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    image.Destroy();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11341");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ResourceHeapIncompatibleImageDimensions) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t buffer_index = 16u;

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = heap_props.bufferDescriptorAlignment * buffer_index;
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_app_size = buffer_offset + buffer_size;

    CreateResourceHeap(resource_heap_app_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_SINT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    VkImageCreateInfo image_ci = image.CreateInfo();
    image_ci.imageType = VK_IMAGE_TYPE_3D;
    image_ci.format = VK_FORMAT_R8G8B8A8_SINT;
    image_ci.extent = {32u, 32u, 1u};
    image_ci.mipLevels = 1u;
    image_ci.arrayLayers = 1u;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    vkt::Image image_3d(*m_device, image_ci);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_GENERAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(descriptor_heap, rgba8i) uniform iimage2D heapImages[];
        layout(descriptor_heap) buffer ssbo {
	        ivec4 data;
        } heapBuffer[];
        void main() {
	        heapBuffer[16].data = imageLoad(heapImages[0], ivec2(0));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    image_barrier.image = image_3d;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    VkImageViewCreateInfo view_info_3d = image_3d.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);
    view_info_3d.viewType = VK_IMAGE_VIEW_TYPE_3D;
    image_info.pView = &view_info_3d;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, descriptor_info, resource_host);

    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11345");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ResourceHeapIncompatibleImageFormat) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t buffer_index = 16u;

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = heap_props.bufferDescriptorAlignment * buffer_index;
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;
    const VkDeviceSize resource_heap_app_size = buffer_offset + buffer_size;

    CreateResourceHeap(resource_heap_app_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image iimage(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_SINT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    vkt::Image uimage(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = iimage.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_GENERAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(descriptor_heap, rgba8i) uniform iimage2D heapImages[];
        layout(descriptor_heap) buffer ssbo {
	        ivec4 data;
        } heapBuffer[];
        void main() {
	        heapBuffer[16].data = imageLoad(heapImages[0], ivec2(0));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = iimage;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    image_barrier.image = uimage;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    VkImageViewCreateInfo uview_info = uimage.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);
    image_info.pView = &uview_info;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, descriptor_info, resource_host);

    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Result-11343");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ImageTexelPointerInvalidDescriptor) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = 1u;
    const VkDeviceSize resource_size = heap_props.imageDescriptorSize;
    const VkDeviceSize heap_size = resource_size * (offset + 1);
    CreateResourceHeap(heap_size);

    VkDeviceSize buffer_size = sizeof(uint32_t) * 4;
    vkt::Buffer texel_buffer(*m_device, buffer_size, VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, vkt::device_address);

    VkHostAddressRangeEXT resource_host;
    resource_host.address = resource_heap_data_ + offset * heap_props.imageDescriptorAlignment;
    resource_host.size = resource_size;

    VkTexelBufferDescriptorInfoEXT texel_buffer_info = vku::InitStructHelper();
    texel_buffer_info.format = VK_FORMAT_R32_UINT;
    texel_buffer_info.addressRange.address = texel_buffer.Address();
    texel_buffer_info.addressRange.size = buffer_size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    descriptor_info.data.pTexelBuffer = &texel_buffer_info;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &resource_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(descriptor_heap, r32ui) uniform uimage2D images[];
        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            imageAtomicAdd(images[1], coord, 1u);
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    texel_buffer.Destroy();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Image-11379");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ImageTexelPointerInvalidFormat) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = 1u;
    const VkDeviceSize resource_size = heap_props.imageDescriptorSize;
    const VkDeviceSize heap_size = resource_size * (offset + 1);
    CreateResourceHeap(heap_size);

    VkDeviceSize buffer_size = sizeof(uint32_t) * 4;
    vkt::Buffer texel_buffer(*m_device, buffer_size, VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, vkt::device_address);

    VkHostAddressRangeEXT resource_host;
    resource_host.address = resource_heap_data_ + offset * heap_props.imageDescriptorAlignment;
    resource_host.size = resource_size;

    VkTexelBufferDescriptorInfoEXT texel_buffer_info = vku::InitStructHelper();
    texel_buffer_info.format = VK_FORMAT_R32_SINT;
    texel_buffer_info.addressRange.address = texel_buffer.Address();
    texel_buffer_info.addressRange.size = buffer_size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    descriptor_info.data.pTexelBuffer = &texel_buffer_info;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &resource_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(descriptor_heap, r32ui) uniform uimageBuffer texelBuffer[];
        void main() {
            imageAtomicAdd(texelBuffer[1], int(gl_GlobalInvocationID.x), 1u);
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Image-11380");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, SamplerCustomBorderColor) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::customBorderColors);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize image_offset = 0u;
    const VkDeviceSize image_size = heap_props.imageDescriptorSize;
    const VkDeviceSize buffer_offset = Align(image_size, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize buffer_size = heap_props.bufferDescriptorSize;

    CreateResourceHeap(buffer_offset + buffer_size);

    vkt::Buffer buffer(*m_device, sizeof(float) * 4u, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Image image(*m_device, 32u, 32u, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkHostAddressRangeEXT resource_host[2];
    resource_host[0].address = resource_heap_data_ + image_offset;
    resource_host[0].size = image_size;
    resource_host[1].address = resource_heap_data_ + buffer_offset;
    resource_host[1].size = buffer_size;

    VkImageViewCreateInfo view_info = image.BasicViewCreatInfo(VK_IMAGE_ASPECT_COLOR_BIT);

    VkImageDescriptorInfoEXT image_info = vku::InitStructHelper();
    image_info.pView = &view_info;
    image_info.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDeviceAddressRangeEXT buffer_address_range;
    buffer_address_range.address = buffer.Address();
    buffer_address_range.size = sizeof(float) * 4u;

    VkResourceDescriptorInfoEXT descriptor_info[2];
    descriptor_info[0] = vku::InitStructHelper();
    descriptor_info[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_info[0].data.pImage = &image_info;
    descriptor_info[1] = vku::InitStructHelper();
    descriptor_info[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info[1].data.pAddressRange = &buffer_address_range;
    vk::WriteResourceDescriptorsEXT(*m_device, 2u, descriptor_info, resource_host);

    const VkDeviceSize sampler_offset = 0u;
    const VkDeviceSize sampler_size = heap_props.samplerDescriptorSize;
    const VkDeviceSize sampler_heap_size = heap_props.samplerDescriptorAlignment * 4;

    CreateSamplerHeap(sampler_heap_size);

    VkSamplerCustomBorderColorCreateInfoEXT border_color = vku::InitStructHelper();
    border_color.customBorderColor.float32[0] = 0.2f;
    border_color.customBorderColor.float32[1] = 0.2f;
    border_color.customBorderColor.float32[2] = 0.2f;
    border_color.customBorderColor.float32[3] = 0.2f;
    border_color.format = VK_FORMAT_R8G8B8A8_UNORM;

    uint32_t custom_border_color_index = 0;
    VkResult res = vk::RegisterCustomBorderColorEXT(*m_device, &border_color, VK_TRUE, &custom_border_color_index);
    ASSERT_EQ(res, VK_SUCCESS);

    VkSamplerCustomBorderColorIndexCreateInfoEXT customBorderColorIndexCreateInfo = vku::InitStructHelper();
    customBorderColorIndexCreateInfo.pNext = &border_color;
    customBorderColorIndexCreateInfo.index = custom_border_color_index;

    VkSamplerCreateInfo sampler_info = SafeSaneSamplerCreateInfo(&customBorderColorIndexCreateInfo);

    VkHostAddressRangeEXT sampler_host;
    sampler_host.address = sampler_heap_data_ + sampler_offset;
    sampler_host.size = sampler_size;
    vk::WriteSamplerDescriptorsEXT(*m_device, 1u, &sampler_info, &sampler_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(descriptor_heap) uniform texture2D textures[];
        layout(descriptor_heap) uniform sampler samplers[];
        layout(descriptor_heap) buffer ssbo {
            vec4 data;
        } buffers[];
        void main() {
            buffers[2].data = texture(sampler2D(textures[0], samplers[0]), vec2(0.5f));
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();

    VkImageMemoryBarrier image_barrier = vku::InitStructHelper();
    image_barrier.srcAccessMask = VK_ACCESS_NONE;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 0u,
                           nullptr, 1u, &image_barrier);

    VkClearColorValue color = {};
    color.float32[0] = 0.2f;
    color.float32[1] = 0.4f;
    color.float32[2] = 0.6f;
    color.float32[3] = 0.8f;
    vk::CmdClearColorImage(m_command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1u,
                           &image_barrier.subresourceRange);

    image_barrier.srcAccessMask = image_barrier.dstAccessMask;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = image_barrier.newLayout;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk::CmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                           0u, nullptr, 1u, &image_barrier);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    BindSamplerHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_default_queue->SubmitAndWait(m_command_buffer);

    vk::UnregisterCustomBorderColorEXT(*m_device, custom_border_color_index);

    m_errorMonitor->SetDesiredError("VUID-vkCmdDispatch-index-11450");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushAddressOOB) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

    const uint32_t read_offset = 48u;
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    const VkDeviceSize descriptor_size = AlignResource(write_offset + resource_stride);
    const VkDeviceSize heap_size = descriptor_size + heap_props.minResourceHeapReservedRange;

    vkt::Buffer heap(*m_device, heap_size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, vkt::device_address);
    uint8_t *heap_data = static_cast<uint8_t *>(heap.Memory().Map());

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = heap_data + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 2u;
    mappings[0].firstBinding = 3u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
    mappings[0].sourceData.pushAddressOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 2, binding = 3) buffer ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[3];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress read_address = read_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_address;
    push_data.data.size = sizeof(read_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);

    VkBindHeapInfoEXT bind_resource_info = vku::InitStructHelper();
    bind_resource_info.heapRange.address = heap.Address();
    bind_resource_info.heapRange.size = heap_size;
    bind_resource_info.reservedRangeOffset = descriptor_size;
    bind_resource_info.reservedRangeSize = heap_props.minResourceHeapReservedRange;
    vk::CmdBindResourceHeapEXT(m_command_buffer, &bind_resource_info);

    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11398");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, IndirectAddressOOB) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const uint32_t read_offset = 48u;
    const uint32_t indirect_offset = 288u;

    vkt::Buffer read_buffer(*m_device, sizeof(uint32_t) * 4 - 1, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);
    vkt::Buffer indirect_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, vkt::device_address);
    uint8_t *indirect_data = static_cast<uint8_t *>(indirect_buffer.Memory().Map());
    VkDeviceAddress *indirect_data_address = reinterpret_cast<VkDeviceAddress *>(indirect_data + indirect_offset);
    *indirect_data_address = read_buffer.Address();

    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT;
    mappings[0].sourceData.indirectAddress.pushOffset = static_cast<uint32_t>(read_offset);
    mappings[0].sourceData.indirectAddress.addressOffset = static_cast<uint32_t>(indirect_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data = read_data;
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkDeviceAddress indirect_address = indirect_buffer.Address();

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &indirect_address;
    push_data.data.size = sizeof(indirect_address);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11398");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, UniformBufferOOB) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, vkt::device_address);

    const uint32_t push_data_offset = 8u;
    const uint32_t push_offset = 4u;
    const uint32_t heap_offset = 3u;
    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * (push_offset + heap_offset);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mapping.sourceData.pushIndex = {};
    mapping.sourceData.pushIndex.heapOffset = heap_offset * static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.pushIndex.pushOffset = push_data_offset;
    mapping.sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform Data {
            vec4 data[32];
        };
        void main() {
            gl_Position = data[gl_VertexIndex];
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_data_offset;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 16u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11372");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, StorageBufferOOB) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

    const uint32_t push_data_offset = 8u;
    const uint32_t push_offset = 4u;
    const uint32_t heap_offset = 3u;
    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * (push_offset + heap_offset);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mapping.sourceData.pushIndex = {};
    mapping.sourceData.pushIndex.heapOffset = heap_offset * static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.pushIndex.pushOffset = push_data_offset;
    mapping.sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex + 4] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_data_offset;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11373");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ShaderObjectStorageBufferOOB) {
    AddRequiredExtensions(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    AddRequiredFeature(vkt::Feature::shaderObject);
    AddRequiredFeature(vkt::Feature::dynamicRendering);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitDynamicRenderTarget();

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

    const uint32_t push_data_offset = 8u;
    const uint32_t push_offset = 4u;
    const uint32_t heap_offset = 3u;
    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * (push_offset + heap_offset);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mapping.sourceData.pushIndex = {};
    mapping.sourceData.pushIndex.heapOffset = heap_offset * static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.pushIndex.pushOffset = push_data_offset;
    mapping.sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex + 4] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    const auto vspv = GLSLToSPV(VK_SHADER_STAGE_VERTEX_BIT, vert_source);
    VkShaderCreateInfoEXT vert_ci = vku::InitStructHelper(&mapping_info);
    vert_ci.flags = VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;
    vert_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_ci.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vert_ci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    vert_ci.codeSize = vspv.size() * sizeof(vspv[0]);
    vert_ci.pCode = vspv.data();
    vert_ci.pName = "main";

    const auto fspv = GLSLToSPV(VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentMinimalGlsl);
    VkShaderCreateInfoEXT frag_ci = vku::InitStructHelper(&mapping_info);
    frag_ci.flags = VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;
    frag_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_ci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    frag_ci.codeSize = fspv.size() * sizeof(fspv[0]);
    frag_ci.pCode = fspv.data();
    frag_ci.pName = "main";

    const vkt::Shader vert_shader(*m_device, vert_ci);
    const vkt::Shader frag_shader(*m_device, frag_ci);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_data_offset;
    push_data.data.address = &push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();

    m_command_buffer.BeginRenderingColor(GetDynamicRenderTarget(), GetRenderTargetArea());
    VkShaderStageFlagBits stages[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkShaderEXT shaders[2] = {vert_shader.handle(), frag_shader.handle()};
    vk::CmdBindShadersEXT(m_command_buffer, 2, stages, shaders);

    vk::CmdSetRasterizerDiscardEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetCullModeEXT(m_command_buffer, VK_CULL_MODE_NONE);
    vk::CmdSetDepthTestEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetStencilTestEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetPolygonModeEXT(m_command_buffer, VK_POLYGON_MODE_FILL);
    vk::CmdSetPrimitiveTopology(m_command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vk::CmdSetPrimitiveRestartEnableEXT(m_command_buffer, VK_FALSE);
    VkColorBlendEquationEXT colorBlendEquation = {
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD,
    };
    vk::CmdSetColorBlendEquationEXT(m_command_buffer, 0, 1u, &colorBlendEquation);
    VkViewport viewport = {0.0f, 0.0f, 32.0f, 32.0f, 0.0f, 1.0f};
    vk::CmdSetViewportWithCountEXT(m_command_buffer, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {32, 32}};
    vk::CmdSetScissorWithCountEXT(m_command_buffer, 1, &scissor);
    vk::CmdSetDepthBiasEnableEXT(m_command_buffer, VK_FALSE);
    vk::CmdSetRasterizationSamplesEXT(m_command_buffer, VK_SAMPLE_COUNT_1_BIT);
    VkSampleMask sampleMask = 0xFFFFFFFF;
    vk::CmdSetSampleMaskEXT(m_command_buffer, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    vk::CmdSetAlphaToCoverageEnableEXT(m_command_buffer, VK_FALSE);
    VkBool32 colorBlendEnable = VK_FALSE;
    vk::CmdSetColorBlendEnableEXT(m_command_buffer, 0u, 1u, &colorBlendEnable);
    VkColorComponentFlags colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    vk::CmdSetColorWriteMaskEXT(m_command_buffer, 0u, 1u, &colorWriteMask);

    VkVertexInputBindingDescription2EXT binding = vku::InitStructHelper();
    binding.binding = 0u;
    binding.stride = 3 * sizeof(float);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    binding.divisor = 1u;

    VkVertexInputAttributeDescription2EXT attribute = vku::InitStructHelper();
    attribute.location = 0u;
    attribute.binding = 0u;
    attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute.offset = 0u;
    vk::CmdSetVertexInputEXT(m_command_buffer, 1u, &binding, 1u, &attribute);

    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRendering();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11374");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

// Test results in device lost
TEST_F(NegativeGpuAVDescriptorHeap, DISABLED_DescriptorHeapOOB) {
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());

    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(resource_stride);

    vkt::Buffer buffer(*m_device, 32, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_;
    descriptor_host.size = resource_stride;
    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = 16;
    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(local_size_x = 1) in;
        layout(descriptor_heap) buffer A { uint a; } heap[];
        layout(push_constant) uniform PushConstant {
            uint b;
        };
        void main() {
            heap[1].a = b;
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    uint32_t src_data = 4321u;

    VkPushDataInfoEXT push_data_info = vku::InitStructHelper();
    push_data_info.offset = 0u;
    push_data_info.data.size = sizeof(uint32_t);
    push_data_info.data.address = &src_data;

    VkPushConstantRange push_const_range;
    push_const_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_const_range.offset = 0u;
    push_const_range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeline_layout_info = vku::InitStructHelper();
    pipeline_layout_info.pushConstantRangeCount = 1u;
    pipeline_layout_info.pPushConstantRanges = &push_const_range;
    vkt::PipelineLayout pipeline_layout(*m_device, pipeline_layout_info);

    m_command_buffer.Begin();
    vk::CmdPushConstants(m_command_buffer, pipeline_layout.handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &src_data);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data_info);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11309");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, DescriptorHeapOOBMapping) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vkt::device_address);

    const uint32_t push_data_offset = 0u;
    const uint32_t push_offset = 0u;
    const uint32_t heap_offset = 0u;
    const VkDeviceSize offset = heap_props.bufferDescriptorAlignment * (push_offset + heap_offset);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;
    CreateResourceHeap(offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = buffer.Address();
    device_range.size = buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mapping = vku::InitStructHelper();
    mapping.descriptorSet = 0u;
    mapping.firstBinding = 0u;
    mapping.bindingCount = 1u;
    mapping.resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
    mapping.sourceData.pushIndex = {};
    mapping.sourceData.pushIndex.heapOffset = heap_offset * static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);
    mapping.sourceData.pushIndex.pushOffset = push_data_offset;
    mapping.sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(heap_props.bufferDescriptorAlignment);

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 1u;
    mapping_info.pMappings = &mapping;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) buffer Data {
            uvec4 data;
        };
        void main() {
            data[gl_VertexIndex] = 1u;
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    uint32_t invalid_push_offset = 2u;

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = push_data_offset;
    push_data.data.address = &invalid_push_offset;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11309");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, HeapDataOOB) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_heap_offset = 0;
    const uint32_t read_push_data_offset = 0u;
    const VkDeviceSize write_offset = Align(heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;

    const VkDeviceSize heap_size = AlignResource(write_offset + resource_stride);
    CreateResourceHeap(heap_size);

    const uint32_t read_push_offset = static_cast<uint32_t>(heap_size);
    const uint32_t read_offset = read_heap_offset + read_push_offset;

    uint32_t *read_data = reinterpret_cast<uint32_t *>(resource_heap_data_ + read_offset);
    for (uint32_t i = 0; i < 4; ++i) {
        read_data[i] = i + 1;
    }

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT;
    mappings[0].sourceData.heapData.heapOffset = static_cast<uint32_t>(read_heap_offset);
    mappings[0].sourceData.heapData.pushOffset = static_cast<uint32_t>(read_push_data_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[0] = read_data[0];
                write_data[1] = read_data[1];
                write_data[2] = read_data[2];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_push_data_offset;
    push_data.data.address = &read_push_offset;
    push_data.data.size = sizeof(read_push_offset);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11398");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, PushDataOOB) {
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::vertexPipelineStoresAndAtomics);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    vkt::Buffer write_buffer(*m_device, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR, vkt::device_address);

    const uint32_t read_offset = static_cast<uint32_t>(heap_props.maxPushDataSize) - sizeof(4);
    const VkDeviceSize write_offset =
        Align(read_offset + heap_props.bufferDescriptorAlignment * 7u, heap_props.bufferDescriptorAlignment);
    const VkDeviceSize resource_stride = heap_props.bufferDescriptorSize;

    CreateResourceHeap(write_offset + resource_stride);

    VkHostAddressRangeEXT descriptor_host;
    descriptor_host.address = resource_heap_data_ + write_offset;
    descriptor_host.size = resource_stride;

    VkDeviceAddressRangeEXT device_range;
    device_range.address = write_buffer.Address();
    device_range.size = write_buffer.CreateInfo().size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pAddressRange = &device_range;

    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &descriptor_host);

    VkDescriptorSetAndBindingMappingEXT mappings[2];
    mappings[0] = vku::InitStructHelper();
    mappings[0].descriptorSet = 0u;
    mappings[0].firstBinding = 0u;
    mappings[0].bindingCount = 1u;
    mappings[0].resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    mappings[0].source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT;
    mappings[0].sourceData.pushDataOffset = static_cast<uint32_t>(read_offset);
    mappings[1] = vku::InitStructHelper();
    mappings[1].descriptorSet = 1u;
    mappings[1].firstBinding = 0u;
    mappings[1].bindingCount = 1u;
    mappings[1].resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
    mappings[1].source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mappings[1].sourceData.constantOffset.heapOffset = static_cast<uint32_t>(write_offset);
    mappings[1].sourceData.constantOffset.heapArrayStride = 0u;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info = vku::InitStructHelper();
    mapping_info.mappingCount = 2u;
    mapping_info.pMappings = mappings;

    char const *vert_source = R"glsl(
        #version 450
        layout(set = 0, binding = 0) uniform ReadData {
            uvec4 read_data;
        };
        layout(set = 1, binding = 0) buffer WriteData {
            uvec4 write_data;
        };
        void main() {
            if (gl_VertexIndex == 0) {
                write_data[1] = read_data[1];
            }
            gl_Position = vec4(1.0f);
            gl_PointSize = 1.0f;
        }
    )glsl";

    VkShaderObj vert_module = VkShaderObj(*m_device, vert_source, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj frag_module = VkShaderObj(*m_device, kFragmentMinimalGlsl, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkPipelineShaderStageCreateInfo stages[2];
    stages[0] = vert_module.GetStageCreateInfo();
    stages[0].pNext = &mapping_info;
    stages[1] = frag_module.GetStageCreateInfo();

    CreatePipelineHelper descriptor_heap_pipe(*this, &pipeline_create_flags_2_create_info);
    descriptor_heap_pipe.gp_ci_.layout = VK_NULL_HANDLE;
    descriptor_heap_pipe.gp_ci_.stageCount = 2u;
    descriptor_heap_pipe.gp_ci_.pStages = stages;
    descriptor_heap_pipe.CreateGraphicsPipeline(false);

    uint32_t read_data = 0;
    VkPushDataInfoEXT push_data = vku::InitStructHelper();
    push_data.offset = read_offset;
    push_data.data.address = &read_data;
    push_data.data.size = sizeof(uint32_t);

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);

    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, descriptor_heap_pipe);
    BindResourceHeap();
    vk::CmdPushDataEXT(m_command_buffer, &push_data);
    vk::CmdDraw(m_command_buffer, 3u, 1u, 0u, 0u);

    m_command_buffer.EndRenderPass();
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-vkCmdDraw-None-11398");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVDescriptorHeap, ImageTexelPointerInvalidDim) {
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredExtensions(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderUntypedPointers);
    RETURN_IF_SKIP(InitGpuAVDescriptorHeap());
    InitRenderTarget();

    const VkDeviceSize offset = 0u;
    const VkDeviceSize resource_size = heap_props.imageDescriptorSize;
    const VkDeviceSize heap_size = resource_size * (offset + 1);
    CreateResourceHeap(heap_size);

    VkDeviceSize buffer_size = sizeof(uint32_t) * 4;
    vkt::Buffer texel_buffer(*m_device, buffer_size, VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, vkt::device_address);

    VkHostAddressRangeEXT resource_host;
    resource_host.address = resource_heap_data_ + offset * heap_props.imageDescriptorAlignment;
    resource_host.size = resource_size;

    VkTexelBufferDescriptorInfoEXT texel_buffer_info = vku::InitStructHelper();
    texel_buffer_info.format = VK_FORMAT_R32_SINT;
    texel_buffer_info.addressRange.address = texel_buffer.Address();
    texel_buffer_info.addressRange.size = buffer_size;

    VkResourceDescriptorInfoEXT descriptor_info;
    descriptor_info = vku::InitStructHelper();
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    descriptor_info.data.pTexelBuffer = &texel_buffer_info;
    vk::WriteResourceDescriptorsEXT(*m_device, 1u, &descriptor_info, &resource_host);

    char const *cs_source = R"glsl(
        #version 450
        #extension GL_EXT_descriptor_heap : require
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(descriptor_heap, r32ui) uniform uimage2D images[];
        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            imageAtomicAdd(images[0], coord, 1u);
        }
    )glsl";
    VkShaderObj cs_module = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2);

    VkPipelineCreateFlags2CreateInfoKHR pipeline_create_flags_2_create_info = vku::InitStructHelper();
    pipeline_create_flags_2_create_info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    CreateComputePipelineHelper pipe(*this, &pipeline_create_flags_2_create_info);
    pipe.cp_ci_.layout = VK_NULL_HANDLE;
    pipe.cp_ci_.stage = cs_module.GetStageCreateInfo();
    pipe.CreateComputePipeline(false);

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    BindResourceHeap();
    vk::CmdDispatch(m_command_buffer, 1u, 1u, 1u);
    m_command_buffer.End();
    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-Image-11382");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}
