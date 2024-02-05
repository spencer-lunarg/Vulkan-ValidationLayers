/* Copyright (c) 2024 The Khronos Group Inc.
 * Copyright (c) 2024 Valve Corporation
 * Copyright (c) 2024 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../framework/layer_validation_tests.h"
#include "../framework/pipeline_helper.h"
#include "../framework/descriptor_helper.h"
#include "../framework/gpu_av_helper.h"

TEST_F(PositiveGpuAVSpirv, LoopPhi) {
    TEST_DESCRIPTION("Loop that has the Phi parent pointed to itself");
    RETURN_IF_SKIP(InitGpuAvFramework());
    RETURN_IF_SKIP(InitState());
    InitRenderTarget();

    VkMemoryPropertyFlags mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vkt::Buffer buffer_uniform(*m_device, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, mem_props);
    vkt::Buffer buffer_storage(*m_device, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem_props);

    uint32_t *data = (uint32_t *)buffer_uniform.memory().map();
    data[0] = 4;  // Scene.lightCount
    buffer_uniform.memory().unmap();

    OneOffDescriptorSet descriptor_set(m_device, {
                                                     {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
                                                     {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
                                                 });
    const vkt::PipelineLayout pipeline_layout(*m_device, {&descriptor_set.layout_});

    descriptor_set.WriteDescriptorBufferInfo(0, buffer_uniform.handle(), 0, VK_WHOLE_SIZE);
    descriptor_set.WriteDescriptorBufferInfo(1, buffer_storage.handle(), 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    descriptor_set.UpdateDescriptorSets();

    // compiled with
    //  dxc -spirv -T ps_6_0 -E psmain -fspv-target-env=vulkan1.1 a.hlsl -fo a.spv
    //
    //  struct SceneData {
    //      uint lightCount;
    //  };
    //  struct Light {
    //      float3 position;
    //  };
    //  ConstantBuffer<SceneData> Scene  : register(b0, space0);
    //  StructuredBuffer<Light>   Lights : register(t1, space0);
    //
    //  float4 main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD) : SV_TARGET {
    //      float3 color = (float3)0;
    //      for (uint i = 0; i < Scene.lightCount; ++i) {
    //          color += normalize(Lights[i].position);
    //      }
    //      return float4(color, 1.0);
    //  }
    //
    // Produces this nasty loop pattern where it seems at first glance the Phi 2nd Parent is actually after it
    //
    // %1 = OpLabel
    // OpBranch %2
    //
    // %2 = OpLabel
    // %phi = OpPhi %int %A %1 %B %3
    // OpLoopMerge %4 %3 None
    // OpBranchConditional %bool %3 %4
    //
    // %3 = OpLabel
    // %B = OpIAdd %int %_ %_
    // OpBranch %2
    //
    // %4 = OpLabel
    const char *fs_source = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %psmain "main" %out_var_SV_TARGET
               OpExecutionMode %psmain OriginUpperLeft
               OpDecorate %out_var_SV_TARGET Location 0
               OpDecorate %Scene DescriptorSet 0
               OpDecorate %Scene Binding 0
               OpDecorate %Lights DescriptorSet 0
               OpDecorate %Lights Binding 1
               OpMemberDecorate %type_ConstantBuffer_SceneData 0 Offset 0
               OpDecorate %type_ConstantBuffer_SceneData Block
               OpMemberDecorate %Light 0 Offset 0
               OpDecorate %_runtimearr_Light ArrayStride 16
               OpMemberDecorate %type_StructuredBuffer_Light 0 Offset 0
               OpMemberDecorate %type_StructuredBuffer_Light 0 NonWritable
               OpDecorate %type_StructuredBuffer_Light BufferBlock
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %v3float = OpTypeVector %float 3
         %13 = OpConstantComposite %v3float %float_0 %float_0 %float_0
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
     %uint_1 = OpConstant %uint 1
    %float_1 = OpConstant %float 1
%type_ConstantBuffer_SceneData = OpTypeStruct %uint
%_ptr_Uniform_type_ConstantBuffer_SceneData = OpTypePointer Uniform %type_ConstantBuffer_SceneData
      %Light = OpTypeStruct %v3float
%_runtimearr_Light = OpTypeRuntimeArray %Light
%type_StructuredBuffer_Light = OpTypeStruct %_runtimearr_Light
%_ptr_Uniform_type_StructuredBuffer_Light = OpTypePointer Uniform %type_StructuredBuffer_Light
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %25 = OpTypeFunction %void
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
       %bool = OpTypeBool
%_ptr_Uniform_v3float = OpTypePointer Uniform %v3float
      %Scene = OpVariable %_ptr_Uniform_type_ConstantBuffer_SceneData Uniform
     %Lights = OpVariable %_ptr_Uniform_type_StructuredBuffer_Light Uniform
%out_var_SV_TARGET = OpVariable %_ptr_Output_v4float Output
     %psmain = OpFunction %void None %25
         %29 = OpLabel
               OpBranch %30
         %30 = OpLabel
         %31 = OpPhi %v3float %13 %29 %32 %33
         %34 = OpPhi %uint %uint_0 %29 %35 %33
         %36 = OpAccessChain %_ptr_Uniform_uint %Scene %int_0
         %37 = OpLoad %uint %36
         %38 = OpULessThan %bool %34 %37
               OpLoopMerge %39 %33 None
               OpBranchConditional %38 %33 %39
         %33 = OpLabel
         %40 = OpAccessChain %_ptr_Uniform_v3float %Lights %int_0 %34 %int_0
         %41 = OpLoad %v3float %40
         %42 = OpExtInst %v3float %1 Normalize %41
         %32 = OpFAdd %v3float %31 %42
         %35 = OpIAdd %uint %34 %uint_1
               OpBranch %30
         %39 = OpLabel
         %43 = OpCompositeExtract %float %31 0
         %44 = OpCompositeExtract %float %31 1
         %45 = OpCompositeExtract %float %31 2
         %46 = OpCompositeConstruct %v4float %43 %44 %45 %float_1
               OpStore %out_var_SV_TARGET %46
               OpReturn
               OpFunctionEnd
        )";
    VkShaderObj fs(this, fs_source, VK_SHADER_STAGE_FRAGMENT_BIT, SPV_ENV_VULKAN_1_0, SPV_SOURCE_ASM);

    CreatePipelineHelper pipe(*this);
    pipe.InitState();
    pipe.shader_stages_ = {pipe.vs_->GetStageCreateInfo(), fs.GetStageCreateInfo()};
    pipe.gp_ci_.layout = pipeline_layout.handle();
    pipe.CreateGraphicsPipeline();

    m_commandBuffer->begin();
    m_commandBuffer->BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_commandBuffer->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.Handle());
    vk::CmdBindDescriptorSets(m_commandBuffer->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout.handle(), 0, 1,
                              &descriptor_set.set_, 0, nullptr);
    vk::CmdDraw(m_commandBuffer->handle(), 3, 1, 0, 0);
    vk::CmdEndRenderPass(m_commandBuffer->handle());
    m_commandBuffer->end();

    m_default_queue->submit(*m_commandBuffer);
    m_default_queue->wait();
}