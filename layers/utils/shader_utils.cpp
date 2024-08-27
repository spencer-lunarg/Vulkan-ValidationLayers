/* Copyright (c) 2015-2024 The Khronos Group Inc.
 * Copyright (c) 2015-2024 Valve Corporation
 * Copyright (c) 2015-2024 LunarG, Inc.
 * Copyright (C) 2015-2023 Google Inc.
 * Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "shader_utils.h"

#include "generated/device_features.h"
#include "utils/hash_util.h"

#include "generated/spirv_tools_commit_id.h"

void ValidationCache::SetUUID(uint8_t *uuid) {
    // Currently not using the last 8 bytes, so make sure they are zero
    std::memset(uuid, 0, VK_UUID_SIZE);

    std::string_view spirv_tool_commit{SPIRV_TOOLS_COMMIT_ID};
    // just make use of the VUID hash as it is the same format
    uint32_t spirv_tool_commit_hash = hash_util::VuidHash(spirv_tool_commit);
    std::memcpy(uuid, &spirv_tool_commit_hash, sizeof(uint32_t));

    std::memcpy(uuid + sizeof(uint32_t), &spirv_val_option_hash_, sizeof(uint32_t));
}

void ValidationCache::Load(VkValidationCacheCreateInfoEXT const *pCreateInfo) {
    const auto headerSize = 2 * sizeof(uint32_t) + VK_UUID_SIZE;
    auto size = headerSize;
    if (!pCreateInfo->pInitialData || pCreateInfo->initialDataSize < size) return;

    uint32_t const *data = (uint32_t const *)pCreateInfo->pInitialData;
    if (data[0] != size) return;
    if (data[1] != VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT) return;
    uint8_t expected_uuid[VK_UUID_SIZE];
    SetUUID(expected_uuid);
    if (memcmp(&data[2], expected_uuid, VK_UUID_SIZE) != 0) return;  // different version

    data = (uint32_t const *)(reinterpret_cast<uint8_t const *>(data) + headerSize);

    auto guard = WriteLock();
    for (; size < pCreateInfo->initialDataSize; data++, size += sizeof(uint32_t)) {
        good_shader_hashes_.insert(*data);
    }
}

void ValidationCache::Write(size_t *pDataSize, void *pData) {
    const auto header_size = 2 * sizeof(uint32_t) + VK_UUID_SIZE;  // 4 bytes for header size + 4 bytes for version number + UUID
    if (!pData) {
        *pDataSize = header_size + good_shader_hashes_.size() * sizeof(uint32_t);
        return;
    }

    if (*pDataSize < header_size) {
        *pDataSize = 0;
        return;  // Too small for even the header!
    }

    uint32_t *out = (uint32_t *)pData;
    size_t actual_size = header_size;

    // Write the header
    *out++ = header_size;
    *out++ = VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT;
    SetUUID(reinterpret_cast<uint8_t *>(out));
    out = (uint32_t *)(reinterpret_cast<uint8_t *>(out) + VK_UUID_SIZE);

    {
        auto guard = ReadLock();
        for (auto it = good_shader_hashes_.begin(); it != good_shader_hashes_.end() && actual_size < *pDataSize;
             it++, out++, actual_size += sizeof(uint32_t)) {
            *out = *it;
        }
    }

    *pDataSize = actual_size;
}

void ValidationCache::Merge(ValidationCache const *other) {
    // self-merging is invalid, but avoid deadlock below just in case.
    if (other == this) {
        return;
    }
    auto other_guard = other->ReadLock();
    auto guard = WriteLock();
    good_shader_hashes_.reserve(good_shader_hashes_.size() + other->good_shader_hashes_.size());
    for (auto h : other->good_shader_hashes_) good_shader_hashes_.insert(h);
}

spv_target_env PickSpirvEnv(const APIVersion &api_version, bool spirv_1_4) {
    if (api_version >= VK_API_VERSION_1_3) {
        return SPV_ENV_VULKAN_1_3;
    } else if (api_version >= VK_API_VERSION_1_2) {
        return SPV_ENV_VULKAN_1_2;
    } else if (api_version >= VK_API_VERSION_1_1) {
        if (spirv_1_4) {
            return SPV_ENV_VULKAN_1_1_SPIRV_1_4;
        } else {
            return SPV_ENV_VULKAN_1_1;
        }
    }
    return SPV_ENV_VULKAN_1_0;
}

// Some Vulkan extensions/features are just all done in spirv-val behind optional settings
void AdjustValidatorOptions(const DeviceExtensions &device_extensions, const DeviceFeatures &enabled_features,
                            spvtools::ValidatorOptions &out_options, uint32_t *out_hash) {
    struct Settings {
        bool relax_block_layout;
        bool uniform_buffer_standard_layout;
        bool scalar_block_layout;
        bool workgroup_scalar_block_layout;
        bool allow_local_size_id;
    } settings;

    // VK_KHR_relaxed_block_layout never had a feature bit so just enabling the extension allows relaxed layout
    // Was promotoed in Vulkan 1.1 so anyone using Vulkan 1.1 also gets this for free
    settings.relax_block_layout = IsExtEnabled(device_extensions.vk_khr_relaxed_block_layout);
    // The rest of the settings are controlled from a feature bit, which are set correctly in the state tracking. Regardless of
    // Vulkan version used, the feature bit is needed (also described in the spec).
    settings.uniform_buffer_standard_layout = enabled_features.uniformBufferStandardLayout == VK_TRUE;
    settings.scalar_block_layout = enabled_features.scalarBlockLayout == VK_TRUE;
    settings.workgroup_scalar_block_layout = enabled_features.workgroupMemoryExplicitLayoutScalarBlockLayout == VK_TRUE;
    settings.allow_local_size_id = enabled_features.maintenance4 == VK_TRUE;

    if (settings.relax_block_layout) {
        // --relax-block-layout
        out_options.SetRelaxBlockLayout(true);
    }
    if (settings.uniform_buffer_standard_layout) {
        // --uniform-buffer-standard-layout
        out_options.SetUniformBufferStandardLayout(true);
    }
    if (settings.scalar_block_layout) {
        // --scalar-block-layout
        out_options.SetScalarBlockLayout(true);
    }
    if (settings.workgroup_scalar_block_layout) {
        // --workgroup-scalar-block-layout
        out_options.SetWorkgroupScalarBlockLayout(true);
    }
    if (settings.allow_local_size_id) {
        // --allow-localsizeid
        out_options.SetAllowLocalSizeId(true);
    }

    // Faster validation without friendly names.
    out_options.SetFriendlyNames(false);

    // The spv_validator_options_t in libspirv.h is hidden so we can't just hash that struct, so instead need to create our own.
    if (out_hash) {
        *out_hash = hash_util::ShaderHash(&settings, sizeof(Settings));
    }
}
