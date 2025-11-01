/* Copyright (c) 2025-2026 The Khronos Group Inc.
 * Copyright (c) 2025-2026 Valve Corporation
 * Copyright (c) 2025-2026 LunarG, Inc.
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

#include "descriptor_heap_validator.h"
#include "state_tracker/pipeline_state.h"
#include "drawdispatch/drawdispatch_vuids.h"
#include "gpuav/core/gpuav.h"
#include "gpuav/resources/gpuav_state_trackers.h"
#include "state_tracker/shader_module.h"
#include "utils/math_utils.h"
#include "utils/shader_utils.h"
#include "../layers/containers/container_utils.h"
#include "../layers/core_checks/cc_buffer_address.h"
#include <xxhash.h>

namespace vvl {

DescriptorHeapValidator::DescriptorHeapValidator(vvl::DeviceProxy &dev, gpuav::CommandBufferSubState &cb_state, uint32_t set_index,
                                                 const LogObjectList *objlist, const Location &loc)
    : Logger(dev.debug_report),
      dev_proxy(dev),
      cb_state(cb_state),
      loc(loc),
      vuids(&GetDrawDispatchVuid(loc.function)),
      original_spirv(nullptr),  // chance might not find
      instruction_position_offset(0),
      objlist(objlist) {}

static std::string DescribeOffset(VkDescriptorSetAndBindingMappingEXT mapping, bool sampler, uint32_t index = 0) {
    std::stringstream msg;

    if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT) {
        msg << "heapOffset (" << mapping.sourceData.constantOffset.heapOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT) {
        msg << "heapOffset (" << mapping.sourceData.pushIndex.heapOffset << ") + pushIndex (" << index << ") * heapIndexStride ("
            << mapping.sourceData.pushIndex.heapIndexStride << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
        msg << "heapOffset (" << mapping.sourceData.indirectIndex.heapOffset << ") + indirectIndex (" << index
            << ") * heapIndexStride (" << mapping.sourceData.indirectIndex.heapIndexStride << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
        msg << "heapOffset (" << mapping.sourceData.indirectIndex.heapOffset << ") + indirectIndex (" << index
            << ") * heapIndexStride (" << mapping.sourceData.indirectIndex.heapIndexStride << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT) {
        msg << "heapOffset (" << mapping.sourceData.heapData.heapOffset << ", pushOffset ("
            << mapping.sourceData.heapData.pushOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT) {
        msg << "pushDataOffset (" << mapping.sourceData.pushDataOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT) {
        msg << "pushAddressOffset (" << mapping.sourceData.pushAddressOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT) {
        msg << "pushOffset (" << mapping.sourceData.indirectAddress.pushOffset << "), addressOffset ("
            << mapping.sourceData.indirectAddress.addressOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT) {
        msg << "heapOffset (" << mapping.sourceData.shaderRecordIndex.heapOffset << ") + shaderRecordIndex (" << index
            << ") * heapIndexStride (" << mapping.sourceData.shaderRecordIndex.heapIndexStride << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_DATA_EXT) {
        msg << "shaderRecordDataOffset (" << mapping.sourceData.shaderRecordDataOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT) {
        msg << "shaderRecordAddressOffset (" << mapping.sourceData.shaderRecordAddressOffset << ")";
    }

    return msg.str();
}

static std::string DescribeSamplerOffset(VkDescriptorSetAndBindingMappingEXT mapping, uint32_t index = 0) {
    std::stringstream msg;

    if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT) {
        msg << "samplerHeapOffset (" << mapping.sourceData.constantOffset.samplerHeapOffset << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT) {
        msg << "samplerHeapOffset (" << mapping.sourceData.pushIndex.samplerHeapOffset << ") + pushIndex (" << index
            << ") * samplerHeapIndexStride (" << mapping.sourceData.pushIndex.samplerHeapIndexStride << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
        msg << "samplerHeapOffset (" << mapping.sourceData.indirectIndex.samplerHeapOffset << ") + indirectIndex (" << index
            << ") * samplerHeapIndexStride (" << mapping.sourceData.indirectIndex.samplerHeapIndexStride << ")";
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
        msg << "samplerHeapOffset (" << mapping.sourceData.indirectIndex.samplerHeapOffset << ") + indirectIndex (" << index
            << ") * samplerHeapIndexStride (" << mapping.sourceData.indirectIndex.samplerHeapIndexStride << ")";
    } else {
        return DescribeOffset(mapping, index);
    }

    return msg.str();
}

static uint32_t GetHeapOffset(VkDescriptorSetAndBindingMappingEXT mapping, bool sampler) {
    if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT) {
        return sampler ? mapping.sourceData.constantOffset.samplerHeapOffset : mapping.sourceData.constantOffset.heapOffset;
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT) {
        return sampler ? mapping.sourceData.pushIndex.samplerHeapOffset : mapping.sourceData.pushIndex.heapOffset;
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
        return sampler ? mapping.sourceData.indirectIndex.samplerHeapOffset : mapping.sourceData.indirectIndex.heapOffset;
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
        return sampler ? mapping.sourceData.indirectIndex.samplerHeapOffset : mapping.sourceData.indirectIndex.heapOffset;
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT) {
        return mapping.sourceData.heapData.heapOffset;
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT) {
        return sampler ? mapping.sourceData.shaderRecordIndex.samplerHeapOffset : mapping.sourceData.shaderRecordIndex.heapOffset;
    }
    return 0;
}

bool DescriptorHeapValidator::ValidateBinding(gpuav::Validator &gpuav, const spirv::ResourceInterfaceVariable &resource_variable,
                                              const VkDescriptorSetAndBindingMappingEXT &mapping,
                                              const gpuav::vko::IndirectAccessMap &indirect_access, uint32_t byte_offset,
                                              bool pipeline, bool robustness) {
    bool skip = false;

    const bool sampler = resource_variable.base_type.Opcode() == spv::OpTypeSampler;

    if (IsValueIn(mapping.source, {VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT,
                                   VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT,
                                   VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT,
                                   VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT})) {
        uint32_t offset = GetHeapOffset(mapping, sampler);
        uint32_t index = 0;
        if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT) {
            index = cb_state.GetPushData<uint32_t>(mapping.sourceData.pushIndex.pushOffset);
            // We don't need to factor in heapArrayStride, because it is guaranteed to be a multiple of heapIndexStride by 11252
            offset += index * mapping.sourceData.pushIndex.heapIndexStride;
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT) {
            gpuav::vko::IndirectKey key = {true, 0, mapping.sourceData.shaderRecordIndex.shaderRecordOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                uint32_t *index_ptr = static_cast<uint32_t *>(buffer->second.GetHostBufferPtr());
                index = *index_ptr;
                offset += index * mapping.sourceData.shaderRecordIndex.heapIndexStride;
            }
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
            gpuav::vko::IndirectKey key = {false, mapping.sourceData.indirectIndex.pushOffset,
                                           mapping.sourceData.indirectIndex.addressOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                uint32_t *indirect_offset = static_cast<uint32_t *>(buffer->second.GetHostBufferPtr());
                offset += *indirect_offset * mapping.sourceData.indirectIndex.heapIndexStride;
            }
        } else {
            index = cb_state.GetPushData<uint32_t>(mapping.sourceData.indirectIndexArray.pushOffset);
            gpuav::vko::IndirectKey key = {false, mapping.sourceData.indirectIndexArray.pushOffset,
                                           mapping.sourceData.indirectIndexArray.addressOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                uint32_t *indirect_offset_array = static_cast<uint32_t *>(buffer->second.GetHostBufferPtr());
                offset += indirect_offset_array[index] * mapping.sourceData.indirectIndexArray.heapIndexStride;
            }
        }
        if ((resource_variable.is_uniform_buffer || resource_variable.is_storage_buffer) &&
            !IsIntegerMultipleOf(offset, gpuav.phys_dev_ext_props.descriptor_heap_props.bufferDescriptorAlignment)) {
            skip |=
                LogError(vuids->descriptor_heap_11297, objlist, loc.Get(),
                         "The offset at which the buffer %s is accessed is %" PRIu32
                         " [%s], which is not aligned to bufferDescriptorAlignment (%" PRIu64 ")",
                         resource_variable.DescribeDescriptor().c_str(), offset, DescribeOffset(mapping, sampler, index).c_str(),
                         gpuav.phys_dev_ext_props.descriptor_heap_props.bufferDescriptorAlignment);
        } else if (resource_variable.IsImage() &&
                   !IsIntegerMultipleOf(offset, gpuav.phys_dev_ext_props.descriptor_heap_props.imageDescriptorAlignment)) {
            skip |=
                LogError(vuids->descriptor_heap_11298, objlist, loc.Get(),
                         "The offset at which the image %s is accessed is %" PRIu32
                         " [%s], which is not aligned to imageDescriptorAlignment (%" PRIu64 ")",
                         resource_variable.DescribeDescriptor().c_str(), offset, DescribeOffset(mapping, sampler, index).c_str(),
                         gpuav.phys_dev_ext_props.descriptor_heap_props.imageDescriptorAlignment);

        } else if (sampler &&
                   !IsIntegerMultipleOf(offset, gpuav.phys_dev_ext_props.descriptor_heap_props.samplerDescriptorAlignment)) {
            skip |= LogError(vuids->descriptor_heap_11299, objlist, loc.Get(),
                             "The offset at which the sampler %s is accessed is %" PRIu32
                             " [%s], which is not aligned to samplerDescriptorAlignment (%" PRIu64 ")",
                             resource_variable.DescribeDescriptor().c_str(), offset, DescribeSamplerOffset(mapping, index).c_str(),
                             gpuav.phys_dev_ext_props.descriptor_heap_props.samplerDescriptorAlignment);
        } else if (resource_variable.is_storage_tensor &&
                   !IsIntegerMultipleOf(offset, gpuav.phys_dev_ext_props.descriptor_heap_tensor_props.tensorDescriptorAlignment)) {
            skip |=
                LogError(vuids->descriptor_heap_11397, objlist, loc.Get(),
                         "The offset at which the tensor %s is accessed is %" PRIu32
                         " [%s], which is not aligned to tensorDescriptorAlignment (%" PRIu64 ")",
                         resource_variable.DescribeDescriptor().c_str(), offset, DescribeOffset(mapping, sampler, index).c_str(),
                         gpuav.phys_dev_ext_props.descriptor_heap_tensor_props.tensorDescriptorAlignment);
        }
    }
    if (IsValueIn(mapping.source,
                  {VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT, VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT,
                   VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT})) {
        VkDeviceAddress address = 0;
        if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT) {
            address = cb_state.GetPushData<VkDeviceAddress>(mapping.sourceData.pushAddressOffset);
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT) {
            gpuav::vko::IndirectKey key = {false, mapping.sourceData.indirectAddress.pushOffset,
                                           mapping.sourceData.indirectAddress.addressOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                VkDeviceAddress *indirect_address = static_cast<VkDeviceAddress *>(buffer->second.GetHostBufferPtr());
                address = *indirect_address;
            }
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT) {
            gpuav::vko::IndirectKey key = {true, 0, mapping.sourceData.shaderRecordAddressOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                address = *static_cast<VkDeviceAddress *>(buffer->second.GetHostBufferPtr());
            }
        }
        const char *usage_vuid = nullptr;
        VkBufferUsageFlagBits2 required_usage_bit = 0;

        if (resource_variable.is_uniform_buffer) {
            usage_vuid = vuids->descriptor_heap_11438;
            required_usage_bit = VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT;

            if (!IsIntegerMultipleOf(address, gpuav.phys_dev_props.limits.minUniformBufferOffsetAlignment)) {
                skip |= LogError(
                    vuids->descriptor_heap_11441, objlist, loc.Get(),
                    "Buffer at addess 0x%" PRIxLEAST64
                    " was accessed as a uniform buffer %s, but is not aligned to minUniformBufferOffsetAlignment (%" PRIu64 ")",
                    address, resource_variable.DescribeDescriptor().c_str(),
                    gpuav.phys_dev_props.limits.minUniformBufferOffsetAlignment);
            }
        } else if (resource_variable.is_storage_buffer) {
            usage_vuid = vuids->descriptor_heap_11439;
            required_usage_bit = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT;

            if (!IsIntegerMultipleOf(address, gpuav.phys_dev_props.limits.minStorageBufferOffsetAlignment)) {
                skip |= LogError(
                    vuids->descriptor_heap_11442, objlist, loc.Get(),
                    "Buffer at addess 0x%" PRIxLEAST64
                    " was accessed as a storage buffer %s, but is not aligned to minStorageBufferOffsetAlignment (%" PRIu64 ")",
                    address, resource_variable.DescribeDescriptor().c_str(),
                    gpuav.phys_dev_props.limits.minStorageBufferOffsetAlignment);
            }
        } else if (resource_variable.base_type.Opcode() == spv::OpTypeAccelerationStructureKHR) {
            usage_vuid = vuids->descriptor_heap_11440;
            required_usage_bit = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

            if (address % 256 != 0) {
                skip |=
                    LogError(vuids->descriptor_heap_11443, objlist, loc.Get(),
                             "Acceleration structure %s uses mapping %s, but the address (0x%" PRIxLEAST64
                             ") [%s] is not aligned to 256 bytes",
                             resource_variable.DescribeDescriptor().c_str(), string_VkDescriptorMappingSourceEXT(mapping.source),
                             address, DescribeOffset(mapping, sampler).c_str());
            }
        }

        if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT ||
            mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT) {
            const char *buffer_address_vuid = mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT
                                                  ? vuids->descriptor_heap_11306
                                                  : vuids->descriptor_heap_11302;
            if (usage_vuid) {
                std::string msg =
                    "The following buffers are missing " + std::string(string_VkBufferUsageFlagBits2(required_usage_bit));
                BufferAddressValidation<1> buffer_address_validator = {{{{usage_vuid,
                                                                          [required_usage_bit](const vvl::Buffer &buffer_state) {
                                                                              return (buffer_state.usage & required_usage_bit) == 0;
                                                                          },
                                                                          [msg]() { return msg; }, kUsageErrorMsgBuffer}}}};

                skip |=
                    buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, address, 0u, buffer_address_vuid);
            } else {
                BufferAddressValidation<0> buffer_address_validator = {};
                skip |=
                    buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, address, 0u, buffer_address_vuid);
            }

            if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT) {
                VkDeviceAddress indirect_address = cb_state.GetPushData<VkDeviceAddress>(mapping.sourceData.pushAddressOffset);
                BufferAddressValidation<0> buffer_address_validator = {};
                skip |= buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, indirect_address, 0u,
                                                                       vuids->descriptor_heap_11302);
            }
        }

        if (!skip && !gpuav.phys_dev_props_core11.protectedNoFault) {
            BufferAddressValidation<1> buffer_address_validator = {
                {{{vuids->descriptor_heap_11455,
                   [](const vvl::Buffer &buffer_state) { return (buffer_state.usage & VK_BUFFER_CREATE_PROTECTED_BIT) != 0; },
                   []() { return "Buffer was created with VK_BUFFER_CREATE_PROTECTED_BIT"; }, kUsageErrorMsgBuffer}}}};
            skip |= buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, address);
        }

        const auto buffers = gpuav.GetBuffersByAddress(address);
        if (buffers.size() == 1) {
            if (byte_offset >= buffers[0]->create_info.size) {
                skip |= LogError(vuids->descriptor_heap_11398, objlist, loc.Get(),
                                 "%s accessed %" PRIu32
                                 " bytes and uses %s mapping, but the buffer "
                                 "(%s) found at %s has size of %" PRIu64 ".",
                                 resource_variable.DescribeDescriptor().c_str(), byte_offset + 1,
                                 string_VkDescriptorMappingSourceEXT(mapping.source), FormatHandle(buffers[0]->Handle()).c_str(),
                                 DescribeOffset(mapping, false).c_str(), buffers[0]->create_info.size);
            }
        }
    }

    if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT) {
        uint32_t offset = mapping.sourceData.pushDataOffset + byte_offset;
        if (offset > gpuav.push_data_offset_) {
            skip |= LogError(vuids->descriptor_heap_11398, objlist, loc.Get(),
                             "%s accessed byte at offset %" PRIu32
                             " using VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT mapping, "
                             "but maxPushDataSize is %" PRIu32 ".",
                             resource_variable.DescribeDescriptor().c_str(), offset, gpuav.push_data_offset_);
        }
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_DATA_EXT) {
        /*uint32_t offset = mapping.sourceData.shaderRecordDataOffset + byte_offset;
        if (offset > shader_record_size) {
            skip |= LogError(vuids->descriptor_heap_11398, objlist, loc.Get(),
                             "%s accessed byte at offset %" PRIu32
                             " using VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_DATA_EXT mapping, "
                             "but size of bound shader record data is %" PRIu32 ".",
                             resource_variable.DescribeDescriptor().c_str(), offset, shader_record_size);
        }*/
    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT) {
        uint32_t offset = mapping.sourceData.heapData.heapOffset;
        offset += cb_state.GetPushData<uint32_t>(mapping.sourceData.heapData.pushOffset);
        if (offset > gpuav.resource_heap_size_) {
            skip |= LogError(vuids->descriptor_heap_11398, objlist, loc.Get(),
                             "%s accessed byte at offset %" PRIu32
                             " using VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT mapping, "
                             "but size of bound resource heap is %" PRIu32 ".",
                             resource_variable.DescribeDescriptor().c_str(), offset, gpuav.resource_heap_size_);
        } else if (offset >= gpuav.resource_heap_reserved_offset_ &&
                   offset < gpuav.resource_heap_reserved_offset_ + gpuav.resource_heap_reserved_range_size_) {
            skip |= LogError(vuids->descriptor_heap_11398, objlist, loc.Get(),
                             "%s accessed byte offset %" PRIu32
                             " using VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT mapping, "
                             "which is within the reserved range [%"
                             PRIu32 ", %" PRIu32 ") of the bound resource heap.",
                             resource_variable.DescribeDescriptor().c_str(), offset, gpuav.resource_heap_reserved_offset_,
                             gpuav.resource_heap_reserved_offset_ + gpuav.resource_heap_reserved_range_size_);
        }
    }

    if (IsValueIn(mapping.source, {VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT,
                                   VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT,
                                   VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT})) {
        uint32_t offset = 0;
        if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
            offset = mapping.sourceData.indirectIndexArray.pushOffset;
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
            offset = mapping.sourceData.indirectIndex.pushOffset;
        } else {
            offset = mapping.sourceData.indirectAddress.pushOffset;
        }
        VkDeviceAddress address = cb_state.GetPushData<VkDeviceAddress>(offset);

        BufferAddressValidation<1> buffer_address_validator = {
            {{{vuids->descriptor_heap_11437,
               [](const vvl::Buffer &buffer_state) { return (buffer_state.usage & VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT) == 0; },
               []() { return "The following buffers are missing VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT"; }, kUsageErrorMsgBuffer}}}};

        const char *vuid = mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT ? vuids->descriptor_heap_11305
                                                                                               : vuids->descriptor_heap_11301;
        skip |= buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, address, 0u, vuid);
        if (!skip) {
            if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT ||
                mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
                if (address % 4 != 0) {
                    skip |= LogError(vuids->descriptor_heap_11300, objlist, loc.Get(),
                                     "Resource %s uses mapping %s, but the address (0x%" PRIxLEAST64
                                     ") in push data at offset %" PRIu32 " is not aligned to 4 bytes",
                                     resource_variable.DescribeDescriptor().c_str(),
                                     string_VkDescriptorMappingSourceEXT(mapping.source), address,
                                     mapping.sourceData.indirectIndex.pushOffset);
                }
                if (!gpuav.phys_dev_props_core11.protectedNoFault) {
                    BufferAddressValidation<1> protected_buffer_address_validator = {
                        {{{vuids->descriptor_heap_11456,
                           [](const vvl::Buffer &buffer_state) {
                               return (buffer_state.usage & VK_BUFFER_CREATE_PROTECTED_BIT) != 0;
                           },
                           []() { return "Buffer was created with VK_BUFFER_CREATE_PROTECTED_BIT"; }, kUsageErrorMsgBuffer}}}};
                    skip |= protected_buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, address);
                }
            } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT) {
                if (address % 8 != 0) {
                    skip |= LogError(vuids->descriptor_heap_11304, objlist, loc.Get(),
                                     "Resource %s uses mapping VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT, but the address "
                                     "(0x%" PRIxLEAST64 ") in push data at offset %" PRIu32 " is not aligned to 8 bytes",
                                     resource_variable.DescribeDescriptor().c_str(), address,
                                     mapping.sourceData.indirectIndex.pushOffset);
                }
            }
        }
    }

    if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT) {
        gpuav::vko::IndirectKey key = {false, mapping.sourceData.indirectAddress.pushOffset,
                                       mapping.sourceData.indirectAddress.addressOffset};
        if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
            VkDeviceAddress *address = static_cast<VkDeviceAddress *>(buffer->second.GetHostBufferPtr());
            BufferAddressValidation<0> buffer_address_validator = {};
            skip |= buffer_address_validator.ValidateDeviceAddress(gpuav, loc.Get(), *objlist, *address, 0u,
                                                                   vuids->descriptor_heap_11302);
        }
    }
    {
        uint32_t offset = GetHeapOffset(mapping, sampler);
        if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT) {
            uint32_t index = cb_state.GetPushData<uint32_t>(mapping.sourceData.pushIndex.pushOffset);
            offset += index * mapping.sourceData.pushIndex.heapIndexStride;
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
            gpuav::vko::IndirectKey key = {false, mapping.sourceData.indirectIndex.pushOffset,
                                           mapping.sourceData.indirectIndex.addressOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                uint32_t *indirect_offset = static_cast<uint32_t *>(buffer->second.GetHostBufferPtr());
                offset += *indirect_offset * mapping.sourceData.indirectIndex.heapIndexStride;
            }
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
            uint32_t index = cb_state.GetPushData<uint32_t>(mapping.sourceData.indirectIndexArray.pushOffset);
            gpuav::vko::IndirectKey key = {false, mapping.sourceData.indirectIndexArray.pushOffset,
                                           mapping.sourceData.indirectIndexArray.addressOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                uint32_t *indirect_offset_array = static_cast<uint32_t *>(buffer->second.GetHostBufferPtr());
                offset += indirect_offset_array[index] * mapping.sourceData.indirectIndexArray.heapIndexStride;
                // Todo factor in array index
            }
        } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT) {
            gpuav::vko::IndirectKey key = {true, 0, mapping.sourceData.shaderRecordIndex.shaderRecordOffset};
            if (auto buffer = indirect_access->find(key); buffer != indirect_access->end()) {
                uint32_t *index_ptr = static_cast<uint32_t *>(buffer->second.GetHostBufferPtr());
                offset += *index_ptr * mapping.sourceData.shaderRecordIndex.heapIndexStride;
            }
        }

        bool oob_access = false;
        std::ostringstream msg;
        if (sampler) {
            if (offset > gpuav.sampler_heap_size_) {
                oob_access = true;
                msg << "Sampler heap size is " << gpuav.sampler_heap_size_ << " bytes, but the sampler "
                    << resource_variable.DescribeDescriptor() << " is accessed at offset " << offset << " [" 
                    << DescribeSamplerOffset(mapping) << "].";
            } else if (offset >= gpuav.sampler_heap_reserved_offset_ &&
                       offset < gpuav.sampler_heap_reserved_offset_ + gpuav.sampler_heap_reserved_range_size_) {
                oob_access = true;
                msg << "Sampler heap reserved range is [" << gpuav.sampler_heap_reserved_offset_ << ", "
                    << gpuav.sampler_heap_reserved_offset_ + gpuav.sampler_heap_reserved_range_size_
                    << "], but the sampler " << resource_variable.DescribeDescriptor() << " is accessed at offset " << offset << " ["
                    << DescribeSamplerOffset(mapping) << "].";
            }
        } else {
            if (offset > gpuav.resource_heap_size_) {
                oob_access = true;
                msg << "Resource heap size is " << gpuav.resource_heap_size_ << " bytes, but the resource "
                    << resource_variable.DescribeDescriptor() << " is accessed at offset " << offset << " ["
                    << DescribeOffset(mapping, sampler) << "].";
            } else if (offset >= gpuav.resource_heap_reserved_offset_ &&
                       offset < gpuav.resource_heap_reserved_offset_ + gpuav.resource_heap_reserved_range_size_) {
                oob_access = true;
                msg << "Resource heap reserved range is [" << gpuav.resource_heap_reserved_offset_ << ", "
                    << gpuav.resource_heap_reserved_offset_ + gpuav.resource_heap_reserved_range_size_
                    << "], but the resource " << resource_variable.DescribeDescriptor() << " is accessed at offset " << offset << " ["
                    << DescribeOffset(mapping, sampler) << "].";
            }
        }
        if (oob_access) {
            skip |= LogError(vuids->descriptor_heap_11309, objlist, loc.Get(), "%s", msg.str().c_str());
        } else if (!robustness && !gpuav.enabled_features.robustBufferAccess && !gpuav.enabled_features.robustBufferAccess2) {
            if (resource_variable.is_storage_buffer || resource_variable.is_uniform_buffer ||
                resource_variable.is_storage_texel_buffer) {
                void *resource_heap_memory = nullptr;
                bool need_to_unmap = false;
                if (gpuav.resource_heap_buffer_state_) {
                    const auto memory_state = gpuav.resource_heap_buffer_state_->MemoryState();
                    if (memory_state->p_driver_data) {
                        resource_heap_memory = memory_state->p_driver_data;
                    } else {
                        const VkMemoryType memoryType =
                            gpuav.phys_dev_mem_props.memoryTypes[memory_state->allocate_info.memoryTypeIndex];
                        if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
                            need_to_unmap = true;
                            DispatchMapMemory(gpuav.device, memory_state->VkHandle(), 0, VK_WHOLE_SIZE, 0, &resource_heap_memory);
                        }
                    }
                }
                if (resource_heap_memory) {
                    uint8_t *data = static_cast<uint8_t *>(resource_heap_memory) + offset;
                    const size_t n = gpuav.buffer_descriptor_size_;
                    gpuav::Validator::HeapKey key = {XXH3_64bits(data, n), n};
                    auto it = gpuav.heap_buffers.find(key);
                    if (it != gpuav.heap_buffers.end()) {
                        const auto &bucket = it->second;
                        VkDeviceSize size = 0;
                        for (const auto &resource : bucket) {
                            if (resource.bytes.size() == n && std::memcmp(resource.bytes.data(), data, n) == 0) {
                                size = resource.address_range.size;
                                break;
                            }
                        }
                        if (byte_offset >= size) {
                            const char *vuid;
                            if (!pipeline) {
                                vuid = vuids->descriptor_heap_11374;
                            } else if (resource_variable.is_uniform_buffer) {
                                vuid = vuids->descriptor_heap_11372;
                            } else {
                                vuid = vuids->descriptor_heap_11373;
                            }
                            skip |=
                                LogError(vuid, objlist, loc.Get(),
                                         "%s accessed %" PRIu32
                                         " bytes and uses %s mapping, but the descriptor in resource heap at offset %" PRIu32
                                         " was written with size %" PRIu32 ".",
                                         resource_variable.DescribeDescriptor().c_str(), byte_offset + 1,
                                         string_VkDescriptorMappingSourceEXT(mapping.source), offset, static_cast<uint32_t>(size));
                        }
                    }
                    if (need_to_unmap) {
                        DispatchUnmapMemory(gpuav.device, gpuav.resource_heap_buffer_state_->MemoryState()->VkHandle());
                    }
                }
            }
        }
    }

    return skip;
}

bool DescriptorHeapValidator::ValidateBinding(gpuav::Validator &gpuav, const spirv::ResourceInterfaceVariable &resource_variable,
                                              const VkShaderDescriptorSetAndBindingMappingInfoEXT &mappings,
                                              const gpuav::vko::IndirectAccessMap &indirect_access, uint32_t byte_offset,
                                              bool pipeline, bool robustness) {
    bool skip = false;

    for (uint32_t i = 0; i < mappings.mappingCount; ++i) {
        const auto &mapping = mappings.pMappings[i];
        if (mapping.descriptorSet == resource_variable.decorations.set &&
            mapping.firstBinding == resource_variable.decorations.binding &&
            ResourceTypeMatchesBinding(mapping.resourceMask, resource_variable)) {
            skip |= ValidateBinding(gpuav, resource_variable, mapping, indirect_access, byte_offset, pipeline, robustness);
            break;
        }
    }

    return skip;
}

}  // namespace vvl