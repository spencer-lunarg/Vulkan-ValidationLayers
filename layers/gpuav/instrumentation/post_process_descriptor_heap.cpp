/* Copyright (c) 2025 LunarG, Inc.
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

#include "gpuav/instrumentation/post_process_descriptor_heap.h"

#include "gpuav/core/gpuav.h"
#include "gpuav/shaders/gpuav_shaders_constants.h"
#include "gpuav/resources/gpuav_state_trackers.h"
#include "drawdispatch/descriptor_heap_validator.h"
#include "state_tracker/shader_module.h"
#include "gpuav/shaders/gpuav_error_codes.h"
#include "gpuav/shaders/gpuav_error_header.h"
#include "generated/spirv_validation_helper.h"
#include "utils/math_utils.h"

namespace gpuav {

struct PostProcessingCbState {
    vko::BufferRange post_process_variables;
};

struct HeapSizes {
    uint32_t resource_heap_size;
    uint32_t resource_reserved_offset;
    uint32_t resource_reserved_size;
    uint32_t sampler_heap_size;
    uint32_t sampler_reserved_offset;
    uint32_t sampler_reserved_size;
};

struct ValidDescriptors {
    vko::BufferRange descriptors;
    uint8_t* resource_heap_memory;
    uint8_t* sampler_heap_memory;
};

struct Slot {
    // see gpuav_shaders_constants.h for how we split this metadata up
    uint32_t meta_data;
    // OpVariable ID of descriptor accessed.
    // This is required to distinguish between 2 aliased descriptors
    uint32_t variable_id;
    // Used in order to print out information about which instruction caused the issue
    uint32_t instruction_position_offset;
    // Last accessed byte
    uint32_t byte_offset;
};

void RegisterPostProcessingDescriptorHeap(Validator& gpuav, CommandBufferSubState& cb) {
    if (!gpuav.gpuav_settings.shader_instrumentation.descriptor_heap) {
        return;
    }

    cb.on_instrumentation_desc_heap_update_functions.emplace_back(
        [](CommandBufferSubState& cb, VkPipelineBindPoint pipeline_bind_point, uint8_t*, const Location& loc,
           VkDeviceAddressRangeEXT& out_address_range, uint32_t& out_dst_binding) mutable {
            PostProcessingCbState& pp_cb_state = cb.shared_resources_cache.GetOrCreate<PostProcessingCbState>();
            pp_cb_state.post_process_variables =
                cb.gpu_resources_manager.GetHostCoherentBufferRange(sizeof(Slot) * kDebugMaxDescSetAndBindings);

            out_address_range.address = pp_cb_state.post_process_variables.offset_address;
            out_address_range.size = pp_cb_state.post_process_variables.size;
            out_dst_binding = glsl::kBindingInstDescriptorHeapPostProcess;
            memset(pp_cb_state.post_process_variables.offset_mapped_ptr, 0, sizeof(VkDeviceAddressRangeEXT));
        });

    cb.on_instrumentation_desc_heap_descriptors_functions.emplace_back(
        [](Validator& gpuav, CommandBufferSubState& cb, VkPipelineBindPoint pipeline_bind_point, uint8_t* resource_heap_memory,
           uint8_t* sampler_heap_memory, const Location& loc, VkDeviceAddressRangeEXT& out_address_range,
           uint32_t& out_dst_binding) {
            ValidDescriptors& valid_descriptors = cb.shared_resources_cache.GetOrCreate<ValidDescriptors>();
            valid_descriptors.resource_heap_memory = resource_heap_memory;
            valid_descriptors.sampler_heap_memory = sampler_heap_memory;
            const VkDeviceSize alignment = std::min(gpuav.buffer_descriptor_alignment_, gpuav.image_descriptor_alignment_);
            VkDeviceSize size =
                sizeof(HeapSizes) + std::max(gpuav.resource_heap_size_ / alignment, VkDeviceSize(256)) * sizeof(uint32_t);
            size = Align(size, gpuav.phys_dev_props.limits.minStorageBufferOffsetAlignment);
            valid_descriptors.descriptors = cb.gpu_resources_manager.GetHostCoherentBufferRange(size);

            out_address_range.address = valid_descriptors.descriptors.offset_address;
            out_address_range.size = valid_descriptors.descriptors.size;
            out_dst_binding = glsl::kBindingInstDescriptorHeapValidDescriptors;
        });

    cb.on_pre_cb_submission_functions.emplace_back([](Validator& gpuav, CommandBufferSubState& cb,
                                                      VkCommandBuffer per_pre_submission_cb) {
        ValidDescriptors* valid_descriptors = cb.shared_resources_cache.TryGet<ValidDescriptors>();
        if (valid_descriptors && valid_descriptors->descriptors.size > 0 && valid_descriptors->resource_heap_memory) {
            const VkDeviceSize size = valid_descriptors->descriptors.size;
            uint8_t* data = reinterpret_cast<uint8_t*>(valid_descriptors->descriptors.offset_mapped_ptr);
            memset(data, 0, size);
            HeapSizes* heap_sizes = reinterpret_cast<HeapSizes*>(data);
            heap_sizes->resource_heap_size = static_cast<uint32_t>(gpuav.resource_heap_size_);
            heap_sizes->resource_reserved_offset = static_cast<uint32_t>(gpuav.resource_heap_reserved_offset_);
            heap_sizes->resource_reserved_size = static_cast<uint32_t>(gpuav.resource_heap_reserved_range_size_);
            heap_sizes->sampler_heap_size = static_cast<uint32_t>(gpuav.sampler_heap_size_);
            heap_sizes->sampler_reserved_offset = static_cast<uint32_t>(gpuav.sampler_heap_reserved_offset_);
            heap_sizes->sampler_reserved_size = static_cast<uint32_t>(gpuav.sampler_heap_reserved_range_size_);
            uint32_t* descriptors = reinterpret_cast<uint32_t*>(data + sizeof(HeapSizes));
            static const uint8_t zeros[1024] = {};
            for (uint32_t i = 0; i < gpuav.resource_heap_size_ / gpuav.buffer_descriptor_alignment_; ++i) {
                if (i == gpuav.resource_heap_reserved_offset_ / gpuav.buffer_descriptor_alignment_) {
                    i += static_cast<uint32_t>(gpuav.resource_heap_reserved_range_size_ / gpuav.buffer_descriptor_alignment_) - 1;
                    continue;
                }
                if (std::memcmp(valid_descriptors->resource_heap_memory + i * gpuav.buffer_descriptor_size_, zeros,
                                gpuav.buffer_descriptor_size_) == 0) {
                    continue;
                }
                descriptors[i] |= gpuav.IsValidBuffer(valid_descriptors->resource_heap_memory + i * gpuav.buffer_descriptor_size_,
                                                      gpuav.buffer_descriptor_size_);
            }
            for (uint32_t i = 0; i < gpuav.resource_heap_size_ / gpuav.image_descriptor_alignment_; ++i) {
                if (i == gpuav.resource_heap_reserved_offset_ / gpuav.image_descriptor_alignment_) {
                    i += static_cast<uint32_t>(gpuav.resource_heap_reserved_range_size_ / gpuav.image_descriptor_alignment_) - 1;
                    continue;
                }
                if (std::memcmp(valid_descriptors->resource_heap_memory + i * gpuav.image_descriptor_size_, zeros,
                                gpuav.image_descriptor_size_) == 0) {
                    continue;
                }
                descriptors[i] |= gpuav.IsValidImage(valid_descriptors->resource_heap_memory + i * gpuav.image_descriptor_size_,
                                                     gpuav.image_descriptor_size_);
            }
            for (uint32_t i = 0; i < gpuav.sampler_heap_size_ / gpuav.sampler_descriptor_alignment_; ++i) {
                if (i == gpuav.sampler_heap_reserved_offset_ / gpuav.sampler_descriptor_alignment_) {
                    i += static_cast<uint32_t>(gpuav.sampler_heap_reserved_range_size_ / gpuav.sampler_descriptor_alignment_) - 1;
                    continue;
                }
                if (std::memcmp(valid_descriptors->sampler_heap_memory + i * gpuav.sampler_descriptor_size_, zeros,
                                gpuav.sampler_descriptor_size_) == 0) {
                    continue;
                }
                descriptors[i] |= gpuav.IsValidSampler(valid_descriptors->sampler_heap_memory + i * gpuav.sampler_descriptor_size_,
                                                       gpuav.sampler_descriptor_size_);
            }
        }
    });

    gpuav::vko::IndirectAccessMap indirect_access_map =
        std::make_shared<vvl::unordered_map<vko::IndirectKey, vko::StagingBuffer, gpuav::vko::IndirectKeyHash>>();
    cb.on_post_cb_submission_functions.emplace_back([indirect_access_map](Validator& gpuav, CommandBufferSubState& cb,
                                                                          VkCommandBuffer per_post_submission_cb) {
        PostProcessingCbState* pp_cb_state = cb.shared_resources_cache.TryGet<PostProcessingCbState>();
        if (pp_cb_state) {
            DispatchDeviceWaitIdle(gpuav.device);  // TODO Todo
            const Slot* data = reinterpret_cast<Slot*>(pp_cb_state->post_process_variables.offset_mapped_ptr);
            if (!data) {
                return;
            }
            for (uint32_t i = 0; i < kDebugMaxDescSetAndBindings; ++i) {
                const Slot& slot = data[i];
                if ((slot.meta_data & glsl::kPostProcessMetaMaskAccessed) == 0) {
                    break;
                }

                uint32_t shader_id = slot.meta_data & glsl::kShaderIdMask;

                auto it = gpuav.instrumented_shaders_map_.find(shader_id);
                if (it == gpuav.instrumented_shaders_map_.end()) {
                    assert(false);
                    continue;
                }

                const ::spirv::ResourceInterfaceVariable* resource_variable = nullptr;
                const VkShaderDescriptorSetAndBindingMappingInfoEXT* mappings = nullptr;
                VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL;
                if (it->second.pipeline != VK_NULL_HANDLE) {
                    // We use pipeline over vkShaderModule as likely they will have been destroyed by now
                    const vvl::Pipeline* pipeline_state = gpuav.Get<vvl::Pipeline>(it->second.pipeline).get();
                    for (const ShaderStageState& stage_state : pipeline_state->stage_states) {
                        auto variable_it = stage_state.entrypoint->resource_interface_variable_map.find(slot.variable_id);
                        if (variable_it != stage_state.entrypoint->resource_interface_variable_map.end()) {
                            mappings =
                                vku::FindStructInPNextChain<VkShaderDescriptorSetAndBindingMappingInfoEXT>(stage_state.GetPNext());
                            resource_variable = variable_it->second;
                            stage = stage_state.GetStage();
                            break;
                        }
                    }
                } else if (it->second.shader_object != VK_NULL_HANDLE) {
                    const vvl::ShaderObject* shader_object_state = gpuav.Get<vvl::ShaderObject>(it->second.shader_object).get();
                    ASSERT_AND_CONTINUE(shader_object_state->entrypoint);
                    mappings = vku::FindStructInPNextChain<VkShaderDescriptorSetAndBindingMappingInfoEXT>(
                        shader_object_state->create_info.pNext);
                    auto variable_it = shader_object_state->entrypoint->resource_interface_variable_map.find(slot.variable_id);
                    if (variable_it != shader_object_state->entrypoint->resource_interface_variable_map.end()) {
                        resource_variable = variable_it->second;
                        stage = shader_object_state->create_info.stage;
                    }
                }
                ASSERT_AND_CONTINUE(resource_variable);
                ASSERT_AND_CONTINUE(mappings);

                for (uint32_t j = 0; j < mappings->mappingCount; ++j) {
                    const auto& mapping = mappings->pMappings[j];
                    if (mapping.descriptorSet != resource_variable->decorations.set ||
                        mapping.firstBinding != resource_variable->decorations.binding) {
                        continue;
                    }
                    bool shader_record = false;
                    uint32_t push_offset = 0u;
                    uint32_t address_offset = 0u;
                    uint32_t size = 0u;
                    if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT) {
                        if (mapping.resourceMask == VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT) {
                            push_offset = mapping.sourceData.indirectIndex.samplerPushOffset;
                            address_offset = mapping.sourceData.indirectIndex.samplerAddressOffset;
                        } else {
                            push_offset = mapping.sourceData.indirectIndex.pushOffset;
                            address_offset = mapping.sourceData.indirectIndex.addressOffset;
                        }
                        size = sizeof(uint32_t);
                    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT) {
                        // Todo, factor in array index
                        if (mapping.resourceMask == VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT) {
                            push_offset = mapping.sourceData.indirectIndexArray.samplerPushOffset;
                            address_offset = mapping.sourceData.indirectIndexArray.samplerAddressOffset;
                        } else {
                            push_offset = mapping.sourceData.indirectIndexArray.pushOffset;
                            address_offset = mapping.sourceData.indirectIndexArray.addressOffset;
                        }
                        size = sizeof(uint32_t);
                    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT) {
                        push_offset = mapping.sourceData.indirectAddress.pushOffset;
                        address_offset = mapping.sourceData.indirectAddress.addressOffset;
                        size = sizeof(VkDeviceAddress);
                    } else if (mapping.source == VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT) {
                        shader_record = true;
                        address_offset = mapping.sourceData.shaderRecordAddressOffset;
                        size = sizeof(VkDeviceAddress);
                    } else {
                        continue;
                    }

                    VkDeviceAddress device_address = {};
                    if (!shader_record) {
                        device_address = cb.GetPushData<VkDeviceAddress>(push_offset);
                    } else {
                        if (stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
                            device_address = cb.base.raygen_shader_binding_table.deviceAddress;
                        } else if (stage == VK_SHADER_STAGE_MISS_BIT_KHR) {
                            device_address = cb.base.miss_shader_binding_table.deviceAddress;
                        } else if (stage == VK_SHADER_STAGE_ANY_HIT_BIT_KHR || stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR ||
                                   stage == VK_SHADER_STAGE_INTERSECTION_BIT_KHR) {
                            device_address = cb.base.hit_shader_binding_table.deviceAddress;
                        } else if (stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR) {
                            device_address = cb.base.callable_shader_binding_table.deviceAddress;
                        } else {
                            assert(0);
                            continue;
                        }
                        if (device_address == 0) {
                            continue;
                        }
                    }

                    vko::IndirectKey key = {shader_record, push_offset, address_offset};
                    vko::StagingBuffer staging_buffer(cb.gpu_resources_manager, size, per_post_submission_cb);
                    indirect_access_map->insert({key, staging_buffer});

                    if (shader_record) {
                        address_offset += gpuav.phys_dev_ext_props.ray_tracing_props_khr.shaderGroupHandleSize;
                    }

                    const auto buffers = gpuav.GetBuffersByAddress(device_address);
                    if (buffers.size() == 1) {
                        VkBufferMemoryBarrier barrier_write_after_read = vku::InitStructHelper();
                        barrier_write_after_read.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                        barrier_write_after_read.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier_write_after_read.buffer = buffers[0]->VkHandle();
                        barrier_write_after_read.offset = address_offset;
                        barrier_write_after_read.size = size;

                        DispatchCmdPipelineBarrier(per_post_submission_cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier_write_after_read, 0,
                                                   nullptr);
                        VkBufferCopy copy;
                        copy.srcOffset = address_offset;
                        copy.dstOffset = staging_buffer.GetBufferRange().offset;
                        copy.size = size;
                        DispatchCmdCopyBuffer(per_post_submission_cb, buffers[0]->VkHandle(),
                                              staging_buffer.GetBufferRange().buffer, 1, &copy);

                        VkBufferMemoryBarrier barrier_read_before_write = vku::InitStructHelper();
                        barrier_read_before_write.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier_read_before_write.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                        barrier_read_before_write.buffer = buffers[0]->VkHandle();
                        barrier_read_before_write.offset = address_offset;
                        barrier_read_before_write.size = size;

                        DispatchCmdPipelineBarrier(per_post_submission_cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier_read_before_write,
                                                   0, nullptr);

                        staging_buffer.CmdCopyDeviceToHost(per_post_submission_cb);
                    }
                }
            }
        }
    });

    cb.on_cb_completion_functions.emplace_back([indirect_access_map](Validator& gpuav, CommandBufferSubState& cb,
                                                                     const CommandBufferSubState::LabelLogging& label_logging,
                                                                     const Location& submission_loc) {
        PostProcessingCbState* pp_cb_state = cb.shared_resources_cache.TryGet<PostProcessingCbState>();
        if (pp_cb_state) {
            const Slot* data = reinterpret_cast<Slot*>(pp_cb_state->post_process_variables.offset_mapped_ptr);
            if (!data) {
                return false;
            }
            std::vector<const ::spirv::ResourceInterfaceVariable*> validated_resource_variables;
            for (uint32_t i = 0; i < kDebugMaxDescSetAndBindings; ++i) {
                const Slot& slot = data[i];
                if ((slot.meta_data & glsl::kPostProcessMetaMaskAccessed) == 0) {
                    break;
                }

                uint32_t shader_id = slot.meta_data & glsl::kShaderIdMask;

                const uint32_t error_logger_i =
                    (slot.meta_data & glsl::kPostProcessMetaMaskErrorLoggerIndex) >> glsl::kPostProcessMetaShiftErrorLoggerIndex;
                const CommandBufferSubState::CommandErrorLogger& cmd_error_logger = cb.GetErrorLogger(error_logger_i);
                std::string debug_region_name =
                    cb.GetDebugLabelRegion(cmd_error_logger.label_cmd_i, label_logging.initial_label_stack);

                Location access_loc(cmd_error_logger.loc.Get(), debug_region_name);
                vvl::DescriptorHeapValidator context(gpuav, cb, 0, nullptr, access_loc);
                context.SetObjlistForGpuAv(&cmd_error_logger.objlist);

                const ::spirv::ResourceInterfaceVariable* resource_variable = nullptr;

                const vvl::Pipeline* pipeline_state = nullptr;
                const vvl::ShaderObject* shader_object_state = nullptr;

                auto it = gpuav.instrumented_shaders_map_.find(shader_id);
                if (it == gpuav.instrumented_shaders_map_.end()) {
                    assert(false);
                    continue;
                }

                if (it->second.pipeline != VK_NULL_HANDLE) {
                    // We use pipeline over vkShaderModule as likely they will have been destroyed by now
                    pipeline_state = gpuav.Get<vvl::Pipeline>(it->second.pipeline).get();
                } else if (it->second.shader_object != VK_NULL_HANDLE) {
                    shader_object_state = gpuav.Get<vvl::ShaderObject>(it->second.shader_object).get();
                    ASSERT_AND_CONTINUE(shader_object_state->entrypoint);
                } else {
                    assert(false);
                    continue;
                }

                const VkShaderDescriptorSetAndBindingMappingInfoEXT* mappings = nullptr;
                bool pipeline = false;
                bool robustness = false;

                if (pipeline_state) {
                    for (const ShaderStageState& stage_state : pipeline_state->stage_states) {
                        ASSERT_AND_CONTINUE(stage_state.entrypoint);
                        auto variable_it = stage_state.entrypoint->resource_interface_variable_map.find(slot.variable_id);
                        if (variable_it != stage_state.entrypoint->resource_interface_variable_map.end()) {
                            mappings =
                                vku::FindStructInPNextChain<VkShaderDescriptorSetAndBindingMappingInfoEXT>(stage_state.GetPNext());
                            resource_variable = variable_it->second;
                            pipeline = true;
                            if (pipeline_state->uses_pipeline_robustness) {
                                robustness = true;
                            }
                            break;  // Only need to find a single entry point
                        }
                    }
                } else if (shader_object_state) {
                    ASSERT_AND_CONTINUE(shader_object_state->entrypoint);
                    mappings = vku::FindStructInPNextChain<VkShaderDescriptorSetAndBindingMappingInfoEXT>(
                        shader_object_state->create_info.pNext);
                    auto variable_it = shader_object_state->entrypoint->resource_interface_variable_map.find(slot.variable_id);
                    if (variable_it != shader_object_state->entrypoint->resource_interface_variable_map.end()) {
                        resource_variable = variable_it->second;
                    }
                }
                ASSERT_AND_CONTINUE(resource_variable);
                ASSERT_AND_CONTINUE(mappings);
                if (std::find(validated_resource_variables.begin(), validated_resource_variables.end(), resource_variable) !=
                    validated_resource_variables.end()) {
                    continue;
                }

                context.ValidateBinding(gpuav, *resource_variable, *mappings, indirect_access_map, slot.byte_offset, pipeline,
                                        robustness);
                validated_resource_variables.push_back(resource_variable);
            }
        }
        return true;
    });

    cb.on_instrumentation_error_logger_register_functions.emplace_back([](Validator& gpuav, CommandBufferSubState& cb,
                                                                          const LastBound& last_bound) {
        CommandBufferSubState::InstrumentationErrorLogger inst_error_logger = [](Validator& gpuav, const Location& loc,
                                                                                 const uint32_t* error_record,
                                                                                 std::string& out_error_msg,
                                                                                 std::string& out_vuid_msg) {
            using namespace glsl;
            bool error_found = false;
            if (GetErrorGroup(error_record) != kErrorGroupInstDescriptorHeap) {
                return error_found;
            }
            error_found = true;

            std::ostringstream strm;

            const uint32_t param1 = error_record[kInstLogErrorParameterOffset_0];
            const uint32_t param2 = error_record[kInstLogErrorParameterOffset_1];
            const uint32_t param3 = error_record[kInstLogErrorParameterOffset_2];

            const uint32_t error_sub_code = GetSubError(error_record);
            switch (error_sub_code) {
                case kErrorSubCodeHeapBufferPointerAlignment: {
                    out_vuid_msg = "VUID-RuntimeSpirv-bufferDescriptorAlignment-11384";
                    strm << "ResourceHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2 << "), but bufferDescriptorAlignment is " << param3;
                } break;
                case kErrorSubCodeHeapSamplerPointerAlignment: {
                    out_vuid_msg = "VUID-RuntimeSpirv-samplerDescriptorAlignment-11348";
                    strm << "SamplerHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2 << "), but samplerDescriptorAlignment is " << param3;
                } break;
                case kErrorSubCodeHeapImagePointerAlignment: {
                    out_vuid_msg = "VUID-RuntimeSpirv-imageDescriptorAlignment-11349";
                    strm << "ResourceHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2 << "), but imageDescriptorAlignment is " << param3;
                } break;
                case kErrorSubCodeHeapImageTexelPointerAlignment: {
                    out_vuid_msg = "VUID-RuntimeSpirv-imageDescriptorAlignment-11383";
                    strm << "ResourceHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2 << "), but imageDescriptorAlignment is " << param3;
                } break;
                case kErrorSubCodeHeapAccelerationStructureAlignment: {
                    out_vuid_msg = "VUID-RuntimeSpirv-Result-11350";
                    strm << "ResourceHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2 << "), but bufferDescriptorAlignment is " << param3;
                } break;
                case kErrorSubCodeHeapTensorAlignment: {
                    out_vuid_msg = "VUID-RuntimeSpirv-Result-11481";
                    strm << "ResourceHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2 << "), but tensorDescriptorAlignment is " << param3;
                } break;
                case kErrorSubCodeHeapInvalidBufferDescriptor:
                    out_vuid_msg = "VUID-RuntimeSpirv-Result-11342";
                    strm << "ResourceHeapEXT is being accessed with index " << param1 << " and stride " << param2
                         << " (offset = " << param1 * param2
                         << "), but the bound descriptor heap at the offset does not contain a valid buffer descriptor";
                    break;
                case kErrorSubCodeHeapInvalidImageDescriptor:
                    if ((param3 & glsl::kDescriptorHeapImage) == 0) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Result-11341";
                        strm << "ResourceHeapEXT is being accessed at index " << param1
                             << ", but the bound descriptor heap at the offset does not contain a valid image descriptor";
                    } else if ((param2 ^ param3) & glsl::kDescriptorHeapImageArrayed) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Result-11345";
                        strm << "OpTypeImage is "
                             << (((param2 & glsl::kDescriptorHeapImageArrayed) != 0) ? "arrayed" : "not arrayed")
                             << ", but the descriptor in resource heap at offset " << param1 << " is "
                             << (((param3 & glsl::kDescriptorHeapImageArrayed) != 0) ? "arrayed" : "not arrayed") << ".";
                    } else if ((param2 ^ param3) & glsl::kDescriptorHeapImageMultiSampled) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Result-11345";
                        strm << "OpTypeImage is "
                             << (((param2 & glsl::kDescriptorHeapImageMultiSampled) != 0) ? "multisampled" : "not multisampled")
                             << ", but the descriptor in resource heap at offset " << param1 << " is "
                             << (((param3 & glsl::kDescriptorHeapImageMultiSampled) != 0) ? "multisampled" : "not multisampled")
                             << ".";
                    } else if (((param2 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask) !=
                               ((param3 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask)) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Result-11345";
                        strm << "OpTypeImage is "
                             << (((param2 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask) != 0
                                     ? "sampled"
                                     : "storage/subpass data")
                             << ", but the descriptor in resource heap at offset " << param1 << " is "
                             << (((param3 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask) != 0
                                     ? "sampled"
                                     : "storage/subpass data")
                             << ".";
                    } else if (((param2 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask) !=
                               ((param3 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask)) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Result-11345";
                        strm << "OpTypeImage has dimension of "
                             << ((param2 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask)
                             << ", but the descriptor in resource heap at offset " << param1 << " has dimension of "
                             << ((param3 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask) << ".";
                    } else if (((param2 >> glsl::kDescriptorHeapImageFormatShift) & glsl::kDescriptorHeapImageFormatMask) !=
                               ((param3 >> glsl::kDescriptorHeapImageFormatShift) & glsl::kDescriptorHeapImageFormatMask)) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Result-11343";
                        strm << "OpTypeImage uses format "
                             << string_SpirvImageFormat(CompatibleSpirvImageFormat(
                                    (param2 >> glsl::kDescriptorHeapImageFormatShift) & glsl::kDescriptorHeapImageFormatMask))
                             << ", which is not compatible with the format "
                             << string_VkFormat(static_cast<VkFormat>((param3 >> glsl::kDescriptorHeapImageFormatShift) &
                                                                      glsl::kDescriptorHeapImageFormatMask))
                             << ", that the descriptor in resource heap at offset " << param1 << " uses.";
                    }
                    break;
                case kErrorSubCodeHeapInvalidTexelPointer:
                    if ((param3 & glsl::kDescriptorHeapTexelPointer) == 0) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Image-11379";
                        strm << "ResourceHeapEXT is being accessed at index " << param1
                             << ", but the bound descriptor heap at the offset does not contain a valid texel buffer "
                                "descriptor";
                    } else if ((param2 ^ param3) & glsl::kDescriptorHeapImageArrayed) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Image-11382";
                        strm << "OpTypeImage is "
                             << (((param2 & glsl::kDescriptorHeapImageArrayed) != 0) ? "arrayed" : "not arrayed")
                             << ", but the descriptor in resource heap at offset " << param1 << " is "
                             << (((param3 & glsl::kDescriptorHeapImageArrayed) != 0) ? "arrayed" : "not arrayed") << ".";
                    } else if ((param2 ^ param3) & glsl::kDescriptorHeapImageMultiSampled) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Image-11382";
                        strm << "OpTypeImage is "
                             << (((param2 & glsl::kDescriptorHeapImageMultiSampled) != 0) ? "multisampled" : "not multisampled")
                             << ", but the descriptor in resource heap at offset " << param1 << " is "
                             << (((param3 & glsl::kDescriptorHeapImageMultiSampled) != 0) ? "multisampled" : "not multisampled")
                             << ".";
                    } else if (((param2 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask) !=
                               ((param3 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask)) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Image-11382";
                        strm << "OpTypeImage is "
                             << (((param2 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask) != 0
                                     ? "sampled"
                                     : "storage/subpass data")
                             << ", but the descriptor in resource heap at offset " << param1 << " is "
                             << (((param3 >> glsl::kDescriptorHeapImageSampledShift) & glsl::kDescriptorHeapImageSampledMask) != 0
                                     ? "sampled"
                                     : "storage/subpass data")
                             << ".";
                    } else if (((param2 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask) !=
                               ((param3 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask)) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Image-11382";
                        strm << "OpTypeImage has dimension of "
                             << ((param2 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask)
                             << ", but the descriptor in resource heap at offset " << param1 << " has dimension of "
                             << ((param3 >> glsl::kDescriptorHeapImageDimShift) & glsl::kDescriptorHeapImageDimMask) << ".";
                    } else if (((param2 >> glsl::kDescriptorHeapImageFormatShift) & glsl::kDescriptorHeapImageFormatMask) !=
                               ((param3 >> glsl::kDescriptorHeapImageFormatShift) & glsl::kDescriptorHeapImageFormatMask)) {
                        out_vuid_msg = "VUID-RuntimeSpirv-Image-11380";
                        strm << "OpTypeImage uses format "
                             << string_SpirvImageFormat(CompatibleSpirvImageFormat(
                                    (param2 >> glsl::kDescriptorHeapImageFormatShift) & glsl::kDescriptorHeapImageFormatMask))
                             << ", which is not compatible with the format "
                             << string_VkFormat(static_cast<VkFormat>((param3 >> glsl::kDescriptorHeapImageFormatShift) &
                                                                      glsl::kDescriptorHeapImageFormatMask))
                             << ", that the descriptor in resource heap at offset " << param1 << " uses.";
                    }
                    break;
                case kErrorSubCodeHeapInvalidSamplerDescriptor:
                    out_vuid_msg = "VUID-RuntimeSpirv-Result-11340";
                    strm << "SamplerHeapEXT is being accessed at index " << param1
                         << ", but the bound descriptor heap at the offset does not contain a valid sampler descriptor";
                    break;
                case kErrorSubCodeHeapUnregisteredCustomBorderColor:
                case kErrorSubCodeHeapDifferentCustomBorderColor:
                    switch (loc.function) {
                        case vvl::Func::vkCmdDraw:
                            out_vuid_msg = "VUID-vkCmdDraw-index-11450";
                            break;
                        case vvl::Func::vkCmdDispatch:
                            out_vuid_msg = "VUID-vkCmdDispatch-index-11450";
                            break;
                        case vvl::Func::vkCmdDispatchBase:
                            out_vuid_msg = "VUID-vkCmdDispatchBase-index-11450";
                            break;
                        case vvl::Func::vkCmdDispatchIndirect:
                            out_vuid_msg = "VUID-vkCmdDispatchIndirect-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawIndexed:
                            out_vuid_msg = "VUID-vkCmdDrawIndexed-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawIndexedIndirect:
                            out_vuid_msg = "VUID-vkCmdDrawIndexedIndirect-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawIndexedIndirectCount:
                            out_vuid_msg = "VUID-vkCmdDrawIndexedIndirectCount-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawIndirect:
                            out_vuid_msg = "VUID-vkCmdDrawIndirect-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawIndirectByteCountEXT:
                            out_vuid_msg = "VUID-vkCmdDrawIndirectByteCountEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawIndirectCount:
                            out_vuid_msg = "VUID-vkCmdDrawIndirectCount-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectCountEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectCountEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectCountNV:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectCountNV-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectNV:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectNV-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksNV:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksNV-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMultiEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMultiEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdDrawMultiIndexedEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMultiIndexedEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdExecuteGeneratedCommandsEXT:
                            out_vuid_msg = "VUID-vkCmdExecuteGeneratedCommandsEXT-index-11450";
                            break;
                        case vvl::Func::vkCmdTraceRaysIndirect2KHR:
                            out_vuid_msg = "VUID-vkCmdTraceRaysIndirect2KHR-index-11450";
                            break;
                        case vvl::Func::vkCmdTraceRaysIndirectKHR:
                            out_vuid_msg = "VUID-vkCmdTraceRaysIndirectKHR-index-11450";
                            break;
                        case vvl::Func::vkCmdTraceRaysKHR:
                            out_vuid_msg = "VUID-vkCmdTraceRaysKHR-index-11450";
                            break;
                        case vvl::Func::vkCmdTraceRaysNV:
                            out_vuid_msg = "VUID-vkCmdTraceRaysNV-index-11450";
                            break;
                        default:
                            return false;
                    }
                    if (error_sub_code == kErrorSubCodeHeapUnregisteredCustomBorderColor) {
                        strm << "Sampler is being accessed in sampler heap at offset " << param1
                             << ", but it was created with a custom border color index that is not registered at the time "
                                "of "
                                "command buffer execution.";
                    } else {
                        strm << "Sampler is being accessed in sampler heap at index " << param1
                             << ", but the custom border color index was created with a different color than the color "
                                "that is "
                                "currently registered.";
                    }
                    break;
                case kErrorSubCodeHeapResourceOOB:
                case kErrorSubCodeHeapResourceReservedRange:
                case kErrorSubCodeHeapSamplerOOB:
                case kErrorSubCodeHeapSamplerReservedRange:
                    switch (loc.function) {
                        case vvl::Func::vkCmdDispatch:
                            out_vuid_msg = "VUID-vkCmdDispatch-None-11309";
                            break;
                        case vvl::Func::vkCmdDispatchBase:
                            out_vuid_msg = "VUID-vkCmdDispatchBase-None-11309";
                            break;
                        case vvl::Func::vkCmdDispatchIndirect:
                            out_vuid_msg = "VUID-vkCmdDispatchIndirect-None-11309";
                            break;
                        case vvl::Func::vkCmdDraw:
                            out_vuid_msg = "VUID-vkCmdDraw-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawIndexed:
                            out_vuid_msg = "VUID-vkCmdDrawIndexed-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawIndexedIndirect:
                            out_vuid_msg = "VUID-vkCmdDrawIndexedIndirect-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawIndexedIndirectCount:
                            out_vuid_msg = "VUID-vkCmdDrawIndexedIndirectCount-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawIndirect:
                            out_vuid_msg = "VUID-vkCmdDrawIndirect-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawIndirectByteCountEXT:
                            out_vuid_msg = "VUID-vkCmdDrawIndirectByteCountEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawIndirectCount:
                            out_vuid_msg = "VUID-vkCmdDrawIndirectCount-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectCountEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectCountEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectCountNV:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectCountNV-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksIndirectNV:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksIndirectNV-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMeshTasksNV:
                            out_vuid_msg = "VUID-vkCmdDrawMeshTasksNV-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMultiEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMultiEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdDrawMultiIndexedEXT:
                            out_vuid_msg = "VUID-vkCmdDrawMultiIndexedEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdExecuteGeneratedCommandsEXT:
                            out_vuid_msg = "VUID-vkCmdExecuteGeneratedCommandsEXT-None-11309";
                            break;
                        case vvl::Func::vkCmdTraceRaysIndirect2KHR:
                            out_vuid_msg = "VUID-vkCmdTraceRaysIndirect2KHR-None-11309";
                            break;
                        case vvl::Func::vkCmdTraceRaysIndirectKHR:
                            out_vuid_msg = "VUID-vkCmdTraceRaysIndirectKHR-None-11309";
                            break;
                        case vvl::Func::vkCmdTraceRaysKHR:
                            out_vuid_msg = "VUID-vkCmdTraceRaysKHR-None-11309";
                            break;
                        case vvl::Func::vkCmdTraceRaysNV:
                            out_vuid_msg = "VUID-vkCmdTraceRaysNV-None-11309";
                            break;
                        default:
                            return false;
                    }
                    if (error_sub_code == kErrorSubCodeHeapResourceOOB) {
                        strm << "Resource heap size is " << param2 << " bytes, but resource attempted to access the heap at "
                             << param1 << " bytes.";
                    } else if (error_sub_code == kErrorSubCodeHeapResourceReservedRange) {
                        strm << "Reserved range of the resource heap size is [" << param2 << ", " << param2 + param3
                             << ") bytes, but resource attempted to access the heap at byte " << param1 << ".";
                    } else if (error_sub_code == kErrorSubCodeHeapSamplerOOB) {
                        strm << "Sampler heap size is " << param2 << " bytes, but sampled attempted to access the heap at "
                             << param1 << " bytes.";
                    } else {
                        strm << "Reserved range of the sampler heap size is [" << param2 << ", " << param2 + param3
                             << ") bytes, but sampler attempted to access the heap at byte " << param1 << ".";
                    }
                    break;
                default:
                    error_found = false;
                    break;
            }
            out_error_msg += strm.str();
            return error_found;
        };

        return inst_error_logger;
    });
}

}  // namespace gpuav
