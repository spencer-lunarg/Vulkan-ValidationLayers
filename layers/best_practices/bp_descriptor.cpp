/* Copyright (c) 2015-2024 The Khronos Group Inc.
 * Copyright (c) 2015-2024 Valve Corporation
 * Copyright (c) 2015-2024 LunarG, Inc.
 * Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (C) 2022 RasterGrid Kft.
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

#include "best_practices/best_practices_validation.h"
#include "best_practices/best_practices_error_enums.h"
#include "best_practices/bp_state.h"

bool BestPractices::PreCallValidateAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                                          VkDescriptorSet* pDescriptorSets, const ErrorObject& error_obj,
                                                          vvl::AllocateDescriptorSetsData& ads_state_data) const {
    bool skip = false;
    skip |= ValidationStateTracker::PreCallValidateAllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets, error_obj,
                                                                          ads_state_data);
    if (skip) return skip;

    const auto pool_state = Get<bp_state::DescriptorPool>(pAllocateInfo->descriptorPool);
    if (!pool_state) return skip;

    // if the number of freed sets > 0, it implies they could be recycled instead if desirable
    // this warning is specific to Arm
    if (VendorCheckEnabled(kBPVendorArm) && (pool_state->freed_count > 0)) {
        skip |= LogPerformanceWarning(
            kVUID_BestPractices_AllocateDescriptorSets_SuboptimalReuse, device, error_obj.location,
            "%s Descriptor set memory was allocated via vkAllocateDescriptorSets() for sets which were previously freed in the "
            "same logical device. On some drivers or architectures it may be most optimal to re-use existing descriptor sets.",
            VendorSpecificTag(kBPVendorArm));
    }

    if (IsExtEnabled(device_extensions.vk_khr_maintenance1)) {
        // Track number of descriptorSets allowable in this pool
        if (pool_state->GetAvailableSets() < pAllocateInfo->descriptorSetCount) {
            skip |=
                LogWarning(kVUID_BestPractices_EmptyDescriptorPool, pool_state->Handle(), error_obj.location,
                           "Unable to allocate %" PRIu32
                           " descriptorSets from %s"
                           ". This pool only has %" PRIu32 " descriptorSets remaining.",
                           pAllocateInfo->descriptorSetCount, FormatHandle(*pool_state).c_str(), pool_state->GetAvailableSets());
        }
    }

    return skip;
}

void BestPractices::ManualPostCallRecordAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                                               VkDescriptorSet* pDescriptorSets, const RecordObject& record_obj,
                                                               vvl::AllocateDescriptorSetsData& ads_state) {
    if (record_obj.result == VK_SUCCESS) {
        if (auto pool_state = Get<bp_state::DescriptorPool>(pAllocateInfo->descriptorPool)) {
            // we record successful allocations by subtracting the allocation count from the last recorded free count
            const auto alloc_count = pAllocateInfo->descriptorSetCount;
            // clamp the unsigned subtraction to the range [0, last_free_count]
            if (pool_state->freed_count > alloc_count) {
                pool_state->freed_count -= alloc_count;
            } else {
                pool_state->freed_count = 0;
            }
        }
    }
}

void BestPractices::PostCallRecordFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount,
                                                     const VkDescriptorSet* pDescriptorSets, const RecordObject& record_obj) {
    ValidationStateTracker::PostCallRecordFreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets,
                                                             record_obj);
    if (record_obj.result == VK_SUCCESS) {
        // we want to track frees because we're interested in suggesting re-use
        if (auto pool_state = Get<bp_state::DescriptorPool>(descriptorPool)) {
            pool_state->freed_count += descriptorSetCount;
        }
    }
}

bool BestPractices::PreCallValidateCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo,
                                                 const VkAllocationCallbacks* pAllocator, VkSampler* pSampler,
                                                 const ErrorObject& error_obj) const {
    bool skip = false;

    if (VendorCheckEnabled(kBPVendorArm)) {
        if ((pCreateInfo->addressModeU != pCreateInfo->addressModeV) || (pCreateInfo->addressModeV != pCreateInfo->addressModeW)) {
            skip |= LogPerformanceWarning(
                kVUID_BestPractices_CreateSampler_DifferentWrappingModes, device, error_obj.location,
                "%s Creating a sampler object with wrapping modes which do not match (U = %u, V = %u, W = %u). "
                "This may cause reduced performance even if only U (1D image) or U/V wrapping modes (2D "
                "image) are actually used. If you need different wrapping modes, disregard this warning.",
                VendorSpecificTag(kBPVendorArm), pCreateInfo->addressModeU, pCreateInfo->addressModeV, pCreateInfo->addressModeW);
        }

        if ((pCreateInfo->minLod != 0.0f) || (pCreateInfo->maxLod < VK_LOD_CLAMP_NONE)) {
            skip |= LogPerformanceWarning(
                kVUID_BestPractices_CreateSampler_LodClamping, device, error_obj.location,
                "%s Creating a sampler object with LOD clamping (minLod = %f, maxLod = %f). This may cause reduced performance. "
                "Instead of clamping LOD in the sampler, consider using an VkImageView which restricts the mip-levels, set minLod "
                "to 0.0, and maxLod to VK_LOD_CLAMP_NONE.",
                VendorSpecificTag(kBPVendorArm), pCreateInfo->minLod, pCreateInfo->maxLod);
        }

        if (pCreateInfo->mipLodBias != 0.0f) {
            skip |=
                LogPerformanceWarning(kVUID_BestPractices_CreateSampler_LodBias, device, error_obj.location,
                                      "%s Creating a sampler object with LOD bias != 0.0 (%f). This will lead to less efficient "
                                      "descriptors being created and may cause reduced performance.",
                                      VendorSpecificTag(kBPVendorArm), pCreateInfo->mipLodBias);
        }

        if ((pCreateInfo->addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
             pCreateInfo->addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
             pCreateInfo->addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) &&
            (pCreateInfo->borderColor != VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK)) {
            skip |= LogPerformanceWarning(
                kVUID_BestPractices_CreateSampler_BorderClampColor, device, error_obj.location,
                "%s Creating a sampler object with border clamping and borderColor != VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK. "
                "This will lead to less efficient descriptors being created and may cause reduced performance. "
                "If possible, use VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK as the border color.",
                VendorSpecificTag(kBPVendorArm));
        }

        if (pCreateInfo->unnormalizedCoordinates) {
            skip |= LogPerformanceWarning(
                kVUID_BestPractices_CreateSampler_UnnormalizedCoordinates, device, error_obj.location,
                "%s Creating a sampler object with unnormalized coordinates. This will lead to less efficient "
                "descriptors being created and may cause reduced performance.",
                VendorSpecificTag(kBPVendorArm));
        }

        if (pCreateInfo->anisotropyEnable) {
            skip |= LogPerformanceWarning(
                kVUID_BestPractices_CreateSampler_Anisotropy, device, error_obj.location,
                "%s Creating a sampler object with anisotropy. This will lead to less efficient descriptors being created "
                "and may cause reduced performance.",
                VendorSpecificTag(kBPVendorArm));
        }
    }

    return skip;
}

bool BestPractices::PreCallValidateUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                        const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount,
                                                        const VkCopyDescriptorSet* pDescriptorCopies,
                                                        const ErrorObject& error_obj) const {
    bool skip = false;
    if (VendorCheckEnabled(kBPVendorAMD)) {
        if (descriptorCopyCount > 0) {
            skip |= LogPerformanceWarning(kVUID_BestPractices_UpdateDescriptors_AvoidCopyingDescriptors, device, error_obj.location,
                                          "%s copying descriptor sets is not recommended", VendorSpecificTag(kBPVendorAMD));
        }
    }

    return skip;
}

bool BestPractices::PreCallValidateCreateDescriptorUpdateTemplate(VkDevice device,
                                                                  const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo,
                                                                  const VkAllocationCallbacks* pAllocator,
                                                                  VkDescriptorUpdateTemplate* pDescriptorUpdateTemplate,
                                                                  const ErrorObject& error_obj) const {
    bool skip = false;
    if (VendorCheckEnabled(kBPVendorAMD)) {
        skip |= LogPerformanceWarning(kVUID_BestPractices_UpdateDescriptors_PreferNonTemplate, device, error_obj.location,
                                      "%s using DescriptorSetWithTemplate is not recommended. Prefer using "
                                      "vkUpdateDescriptorSet instead",
                                      VendorSpecificTag(kBPVendorAMD));
    }

    return skip;
}

std::shared_ptr<vvl::DescriptorPool> BestPractices::CreateDescriptorPoolState(VkDescriptorPool handle,
                                                                              const VkDescriptorPoolCreateInfo* pCreateInfo) {
    return std::static_pointer_cast<vvl::DescriptorPool>(std::make_shared<bp_state::DescriptorPool>(*this, handle, pCreateInfo));
}