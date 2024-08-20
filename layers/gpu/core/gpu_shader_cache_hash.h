
/* Copyright (c) 2020-2024 The Khronos Group Inc.
 * Copyright (c) 2020-2024 Valve Corporation
 * Copyright (c) 2020-2024 LunarG, Inc.
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
#pragma once
// Default values for those settings should match layers/VkLayer_khronos_validation.json.in

#include "generated/gpu_av_shader_hash.h"
#include "gpu/core/gpu_settings.h"

// This structure is included in the shader cache to make sure is invalidated if a setting
// or a GPUAV shader changes.
#pragma pack(push, 1)
struct ShaderCacheHash {
    ShaderCacheHash(const GpuAVSettings& gpuav_settings)
        : validate_descriptors_(gpuav_settings.validate_descriptors),
          warn_on_robust_oob_(gpuav_settings.warn_on_robust_oob),
          validate_bda_(gpuav_settings.validate_bda),
          validate_ray_query_(gpuav_settings.validate_ray_query) {}
    // Settings that are part of shader instrumentation that would need us to invalidate the cache
    bool validate_descriptors_;
    bool warn_on_robust_oob_;
    bool validate_bda_;
    bool validate_ray_query_;
    const char gpu_av_shader_git_hash[sizeof(GPU_AV_SHADER_GIT_HASH)] = GPU_AV_SHADER_GIT_HASH;
};
#pragma pack(pop)

