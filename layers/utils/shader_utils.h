/* Copyright (c) 2015-2025 The Khronos Group Inc.
 * Copyright (c) 2015-2025 Valve Corporation
 * Copyright (c) 2015-2025 LunarG, Inc.
 * Copyright (C) 2015-2025 Google Inc.
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
 *
 * The Shader Validation file is in charge of taking the Shader Module data and validating it
 */

#pragma once

#include <vulkan/vulkan_core.h>
#include "containers/custom_containers.h"
#include "utils/lock_utils.h"
#include "state_tracker/shader_module.h"

namespace spirv {
struct ResourceInterfaceVariable;
}

enum class ShaderObjectStage : uint32_t {
    VERTEX = 0u,
    TESSELLATION_CONTROL,
    TESSELLATION_EVALUATION,
    GEOMETRY,
    FRAGMENT,
    COMPUTE,
    TASK,
    MESH,

    LAST = 8u,
};

constexpr uint32_t kShaderObjectStageCount = 8u;

inline ShaderObjectStage VkShaderStageToShaderObjectStage(VkShaderStageFlagBits stage) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:
            return ShaderObjectStage::VERTEX;
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
            return ShaderObjectStage::TESSELLATION_CONTROL;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
            return ShaderObjectStage::TESSELLATION_EVALUATION;
        case VK_SHADER_STAGE_GEOMETRY_BIT:
            return ShaderObjectStage::GEOMETRY;
        case VK_SHADER_STAGE_FRAGMENT_BIT:
            return ShaderObjectStage::FRAGMENT;
        case VK_SHADER_STAGE_COMPUTE_BIT:
            return ShaderObjectStage::COMPUTE;
        case VK_SHADER_STAGE_TASK_BIT_EXT:
            return ShaderObjectStage::TASK;
        case VK_SHADER_STAGE_MESH_BIT_EXT:
            return ShaderObjectStage::MESH;
        default:
            break;
    }
    return ShaderObjectStage::LAST;
}

class ValidationCache {
  public:
    static VkValidationCacheEXT Create(VkValidationCacheCreateInfoEXT const *pCreateInfo, uint32_t spirv_val_option_hash) {
        auto cache = new ValidationCache(spirv_val_option_hash);
        cache->Load(pCreateInfo);
        return VkValidationCacheEXT(cache);
    }

    void Load(VkValidationCacheCreateInfoEXT const *pCreateInfo);
    void Write(size_t *pDataSize, void *pData);
    void Merge(ValidationCache const *other);

    bool Contains(uint32_t hash) {
        auto guard = ReadLock();
        return good_shader_hashes_.count(hash) != 0;
    }

    void Insert(uint32_t hash) {
        auto guard = WriteLock();
        good_shader_hashes_.insert(hash);
    }

  private:
    ValidationCache(uint32_t spirv_val_option_hash) : spirv_val_option_hash_(spirv_val_option_hash) {}
    ReadLockGuard ReadLock() const { return ReadLockGuard(lock_); }
    WriteLockGuard WriteLock() { return WriteLockGuard(lock_); }

    void GetUUID(uint8_t *uuid);

    // Can hit cases where error appear/disappear if spirv-val settings are adjusted
    // see https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8031
    uint32_t spirv_val_option_hash_;

    // hashes of shaders that have passed validation before, and can be skipped.
    // we don't store negative results, as we would have to also store what was
    // wrong with them; also, we expect they will get fixed, so we're less
    // likely to see them again.
    vvl::unordered_set<uint32_t> good_shader_hashes_;
    mutable std::shared_mutex lock_;
};

void DumpSpirvToFile(const std::string &file_name, const uint32_t *spirv, size_t spirv_dwords_count);

bool ResourceTypeMatchesBinding(VkSpirvResourceTypeFlagsEXT resource_type,
                                const spirv::ResourceInterfaceVariable &resource_interface_variable);

inline uint32_t GetSpvCompatibleFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_UNORM:
            return spv::ImageFormatR8;
        case VK_FORMAT_R8_SNORM:
            return spv::ImageFormatR8Snorm;
        case VK_FORMAT_R8_UINT:
            return spv::ImageFormatR8ui;
        case VK_FORMAT_R8_SINT:
            return spv::ImageFormatR8i;
        case VK_FORMAT_R8G8_UNORM:
            return spv::ImageFormatRg8;
        case VK_FORMAT_R8G8_SNORM:
            return spv::ImageFormatRg8Snorm;
        case VK_FORMAT_R8G8_UINT:
            return spv::ImageFormatRg8ui;
        case VK_FORMAT_R8G8_SINT:
            return spv::ImageFormatRg8i;
        case VK_FORMAT_R8G8B8A8_UNORM:
            return spv::ImageFormatRgba8;
        case VK_FORMAT_R8G8B8A8_SNORM:
            return spv::ImageFormatRgba8Snorm;
        case VK_FORMAT_R8G8B8A8_UINT:
            return spv::ImageFormatRgba8ui;
        case VK_FORMAT_R8G8B8A8_SINT:
            return spv::ImageFormatRgba8i;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return spv::ImageFormatRgb10A2;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32:
            return spv::ImageFormatRgb10a2ui;
        case VK_FORMAT_R16_UNORM:
            return spv::ImageFormatR16;
        case VK_FORMAT_R16_SNORM:
            return spv::ImageFormatR16Snorm;
        case VK_FORMAT_R16_UINT:
            return spv::ImageFormatR16ui;
        case VK_FORMAT_R16_SINT:
            return spv::ImageFormatR16i;
        case VK_FORMAT_R16_SFLOAT:
            return spv::ImageFormatR16f;
        case VK_FORMAT_R16G16_UNORM:
            return spv::ImageFormatRg16;
        case VK_FORMAT_R16G16_SNORM:
            return spv::ImageFormatRg16Snorm;
        case VK_FORMAT_R16G16_UINT:
            return spv::ImageFormatRg16ui;
        case VK_FORMAT_R16G16_SINT:
            return spv::ImageFormatRg16i;
        case VK_FORMAT_R16G16_SFLOAT:
            return spv::ImageFormatRg16f;
        case VK_FORMAT_R16G16B16A16_UNORM:
            return spv::ImageFormatRgba16;
        case VK_FORMAT_R16G16B16A16_SNORM:
            return spv::ImageFormatRgba16Snorm;
        case VK_FORMAT_R16G16B16A16_UINT:
            return spv::ImageFormatRgba16ui;
        case VK_FORMAT_R16G16B16A16_SINT:
            return spv::ImageFormatRgba16i;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return spv::ImageFormatRgba16f;
        case VK_FORMAT_R32_UINT:
            return spv::ImageFormatR32ui;
        case VK_FORMAT_R32_SINT:
            return spv::ImageFormatR32i;
        case VK_FORMAT_R32_SFLOAT:
            return spv::ImageFormatR32f;
        case VK_FORMAT_R32G32_UINT:
            return spv::ImageFormatRg32ui;
        case VK_FORMAT_R32G32_SINT:
            return spv::ImageFormatRg32i;
        case VK_FORMAT_R32G32_SFLOAT:
            return spv::ImageFormatRg32f;
        case VK_FORMAT_R32G32B32A32_UINT:
            return spv::ImageFormatRgba32ui;
        case VK_FORMAT_R32G32B32A32_SINT:
            return spv::ImageFormatRgba32i;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return spv::ImageFormatRgba32f;
        case VK_FORMAT_R64_UINT:
            return spv::ImageFormatR64ui;
        case VK_FORMAT_R64_SINT:
            return spv::ImageFormatR64i;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
            return spv::ImageFormatR11fG11fB10f;
        default:
            break;
    }
    return spv::ImageFormatUnknown;
}
