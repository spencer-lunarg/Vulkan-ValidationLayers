/* Copyright (c) 2024-2025 LunarG, Inc.
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

#include "buffer_device_address_pass.h"
#include "generated/spirv_grammar_helper.h"
#include "module.h"
#include <cstdint>
#include <spirv/unified1/spirv.hpp>
#include <iostream>
#include "type_manager.h"
#include "utils/math_utils.h"
#include "gpuav/shaders/gpuav_error_header.h"

#include "generated/instrumentation_buffer_device_address_comp.h"
#include "generated/instrumentation_buffer_device_address_alignment_check_comp.h"
#include "generated/instrumentation_buffer_device_address_alignment_report_comp.h"

namespace gpuav {
namespace spirv {

const static OfflineLinkInfo link_info = {instrumentation_buffer_device_address_comp,
                                          instrumentation_buffer_device_address_comp_size, "inst_buffer_device_address",
                                          ZeroInitializeUintPrivateVariables};

const static OfflineLinkInfo link_info_check = {instrumentation_buffer_device_address_alignment_check_comp,
                                                  instrumentation_buffer_device_address_alignment_check_comp_size,
                                                  "inst_buffer_device_address_alignment_check", SwapPrivateVariable};
const static OfflineLinkInfo link_info_report = {instrumentation_buffer_device_address_alignment_report_comp,
                                                   instrumentation_buffer_device_address_alignment_report_comp_size,
                                                   "inst_buffer_device_address_alignment_report", SwapPrivateVariable};

BufferDeviceAddressPass::BufferDeviceAddressPass(Module& module) : Pass(module) { module.use_bda_ = true; }

// By appending the LinkInfo, it will attempt at linking stage to add the function.
uint32_t BufferDeviceAddressPass::GetLinkFunctionId() { return module_.GetLinkFunction(link_function_id_, link_info); }

uint32_t BufferDeviceAddressPass::GetLinkFunctionIdAlignmentCheck() {
    if (link_alignment_check_id_ == 0) {
        link_alignment_check_id_ = module_.TakeNextId();
        assert(invalid_alignment_priavte_variable_id_ != 0);  // should have been set at top of function
        module_.link_info_.emplace_back(
            LinkInfo{link_info_check, link_alignment_check_id_, invalid_alignment_priavte_variable_id_});
    }
    return link_alignment_check_id_;
}

uint32_t BufferDeviceAddressPass::GetLinkFunctionIdAlignmentReport() {
    if (link_alignment_report_id_ == 0) {
        link_alignment_report_id_ = module_.TakeNextId();
        assert(invalid_alignment_priavte_variable_id_ != 0);  // should have been set at top of function
        module_.link_info_.emplace_back(
            LinkInfo{link_info_report, link_alignment_report_id_, invalid_alignment_priavte_variable_id_});
    }
    return link_alignment_report_id_;
}

uint32_t BufferDeviceAddressPass::CreateFunctionCall(BasicBlock& block, InstructionIt* inst_it, const InjectionData& injection_data,
                                                     const InstructionMeta& meta) {
    // The Pointer ID Operand is always the first operand for Load/Store/Atomics
    // We can just take it and cast to a uint64 here to examine the ptr value
    const uint32_t pointer_id = meta.target_instruction->Operand(0);

    // Convert reference pointer to uint64
    const Type& uint64_type = module_.type_manager_.GetTypeInt(64, 0);
    const uint32_t addr_id = module_.TakeNextId();
    block.CreateInstruction(spv::OpConvertPtrToU, {uint64_type.Id(), addr_id, pointer_id}, inst_it);

    uint32_t function_result = 0;

    {
        function_result = module_.TakeNextId();
        const uint32_t function_def = GetLinkFunctionIdAlignmentCheck();
        const uint32_t void_type = module_.type_manager_.GetTypeVoid().Id();
        const Constant& alignment_constant = module_.type_manager_.GetConstantUInt32(meta.alignment_literal);

        block.CreateInstruction(
            spv::OpFunctionCall,
            {void_type, function_result, function_def, addr_id, alignment_constant.Id(), injection_data.inst_position_id}, inst_it);
        function_alignment_checks_++;
    }

    {
        const Constant& length_constant = module_.type_manager_.GetConstantUInt32(meta.type_length);
        const uint32_t opcode = meta.target_instruction->Opcode();

        uint32_t access_type_value = 0;
        if (opcode == spv::OpStore) {
            access_type_value |= 1 << glsl::kInstBuffAddrAccessPayloadShiftIsWrite;
        }
        if (meta.type_is_struct) {
            access_type_value |= 1 << glsl::kInstBuffAddrAccessPayloadShiftIsStruct;
        }
        const Constant& access_type = module_.type_manager_.GetConstantUInt32(access_type_value);

        function_result = module_.TakeNextId();
        const uint32_t function_def = GetLinkFunctionId();
        const uint32_t bool_type = module_.type_manager_.GetTypeBool().Id();

        block.CreateInstruction(spv::OpFunctionCall,
                                {bool_type, function_result, function_def, injection_data.inst_position_id,
                                 injection_data.stage_info_id, addr_id, length_constant.Id(), access_type.Id()},
                                inst_it);
    }

    return function_result;
}

void BufferDeviceAddressPass::ClearPrivateVariable(BasicBlock& block, InstructionIt* inst_it) {
    // First time we clear, we add the variable
    // TODO - Find out if we leave in if there are no instructions instrumented (Does CI or traces break?)
    if (invalid_alignment_priavte_variable_id_ == 0) {
        invalid_alignment_priavte_variable_id_ = module_.TakeNextId();
        const Type& uint32_type = module_.type_manager_.GetTypeInt(32, false);
        const Type& uvec4_type = module_.type_manager_.GetTypeVector(uint32_type, 4);
        const Type& pointer_type = module_.type_manager_.GetTypePointer(spv::StorageClassPrivate, uvec4_type);
        auto new_inst = std::make_unique<Instruction>(4, spv::OpVariable);
        new_inst->Fill({pointer_type.Id(), invalid_alignment_priavte_variable_id_, spv::StorageClassPrivate});
        module_.type_manager_.AddVariable(std::move(new_inst), pointer_type);
    }

    const uint32_t uvec4_zero_id = module_.type_manager_.GetConstantZeroUvec4().Id();
    block.CreateInstruction(spv::OpStore, {invalid_alignment_priavte_variable_id_, uvec4_zero_id}, inst_it);

    // Reset here instead at report time (because functions can have multiple returnZ)
    function_alignment_checks_ = 0;
}

void BufferDeviceAddressPass::CreateFunctionCallAlignmentReport(BasicBlock& block, InstructionIt* inst_it) {
    if (function_alignment_checks_ == 0) return;

    uint32_t function_result = module_.TakeNextId();
    const uint32_t function_def = GetLinkFunctionIdAlignmentReport();
    const uint32_t void_type = module_.type_manager_.GetTypeVoid().Id();
    // TODO - Add in the Stage Info
    block.CreateInstruction(spv::OpFunctionCall, {void_type, function_result, function_def}, inst_it);
}

uint32_t BufferDeviceAddressPass::FindLastByteOffset(uint32_t bda_struct_id,
                                                     const std::vector<const Instruction*>& access_chain_insts) const {
    assert(!access_chain_insts.empty());
    uint32_t last_byte_offset = 0;
    const uint32_t reset_ac_word = 4;  // points to first "Index" operand of an OpAccessChain
    uint32_t ac_word_index = reset_ac_word;

    uint32_t matrix_stride = 0;
    bool col_major = false;
    bool in_matrix = false;

    auto access_chain_iter = access_chain_insts.rbegin();

    uint32_t current_type_id = bda_struct_id;
    // Walk down access chains to build up the offset
    while (access_chain_iter != access_chain_insts.rend()) {
        const uint32_t ac_index_id = (*access_chain_iter)->Word(ac_word_index);
        const Constant* index_constant = module_.type_manager_.FindConstantById(ac_index_id);
        if (!index_constant || index_constant->inst_.Opcode() != spv::OpConstant) {
            return 0;  // Access Chain has dynamic value
        }
        const uint32_t constant_value = index_constant->GetValueUint32();

        uint32_t current_offset = 0;

        const Type* current_type = module_.type_manager_.FindTypeById(current_type_id);
        switch (current_type->spv_type_) {
            case SpvType::kArray:
            case SpvType::kRuntimeArray: {
                // Get array stride and multiply by current index
                const uint32_t array_stride = GetDecoration(current_type_id, spv::DecorationArrayStride)->Word(3);
                current_offset = constant_value * array_stride;

                current_type_id = current_type->inst_.Operand(0);  // Get element type for next step
            } break;
            case SpvType::kMatrix: {
                if (matrix_stride == 0) {
                    module_.InternalError(Name(), "FindLastByteOffset is missing matrix stride");
                }
                in_matrix = true;
                uint32_t vec_type_id = current_type->inst_.Operand(0);

                // If column major, multiply column index by matrix stride, otherwise by vector component size and save matrix
                // stride for vector (row) index
                uint32_t col_stride = 0;
                if (col_major) {
                    col_stride = matrix_stride;
                } else {
                    const uint32_t component_type_id = module_.type_manager_.FindTypeById(vec_type_id)->inst_.Operand(0);
                    col_stride = FindTypeByteSize(component_type_id);
                }

                current_offset = constant_value * col_stride;

                current_type_id = vec_type_id;  // Get element type for next step
            } break;
            case SpvType::kVector: {
                // If inside a row major matrix type, multiply index by matrix stride,
                // else multiply by component size
                const uint32_t component_type_id = current_type->inst_.Operand(0);

                if (in_matrix && !col_major) {
                    current_offset = constant_value * matrix_stride;
                } else {
                    const uint32_t component_type_size = FindTypeByteSize(component_type_id);
                    current_offset = constant_value * component_type_size;
                }

                current_type_id = component_type_id;  // Get element type for next step
            } break;
            case SpvType::kStruct: {
                // Get buffer byte offset for the referenced member
                current_offset = GetMemberDecoration(current_type_id, constant_value, spv::DecorationOffset)->Word(4);

                // Look for matrix stride for this member if there is one. The matrix
                // stride is not on the matrix type, but in a OpMemberDecorate on the
                // enclosing struct type at the member index. If none is found, reset
                // stride to 0.
                const Instruction* decoration_matrix_stride =
                    GetMemberDecoration(current_type_id, constant_value, spv::DecorationMatrixStride);
                matrix_stride = decoration_matrix_stride ? decoration_matrix_stride->Word(4) : 0;

                const Instruction* decoration_col_major =
                    GetMemberDecoration(current_type_id, constant_value, spv::DecorationColMajor);
                col_major = decoration_col_major != nullptr;

                current_type_id = current_type->inst_.Operand(constant_value);  // Get element type for next step
            } break;
            default: {
                module_.InternalError(Name(), "FindLastByteOffset has unexpected non-composite type");
            } break;
        }

        last_byte_offset += current_offset;

        ac_word_index++;
        if (ac_word_index >= (*access_chain_iter)->Length()) {
            ++access_chain_iter;
            ac_word_index = reset_ac_word;
        }
    }

    // Add in offset of last byte of referenced object
    const uint32_t accessed_type_size = FindTypeByteSize(current_type_id, matrix_stride, col_major, in_matrix);
    const uint32_t last_byte_index = accessed_type_size - 1;
    last_byte_offset += last_byte_index;

    return last_byte_offset;
}

bool BufferDeviceAddressPass::RequiresInstrumentation(const Function& function, const Instruction& inst, InstructionMeta& meta,
                                                      bool pre_pass) {
    const uint32_t opcode = inst.Opcode();
    if (opcode == spv::OpLoad || opcode == spv::OpStore) {
        // We only care if there is an Aligned Memory Operands
        // VUID-StandaloneSpirv-PhysicalStorageBuffer64-04708 requires there to be an Aligned operand
        const uint32_t memory_operand_index = opcode == spv::OpLoad ? 4 : 3;
        const uint32_t alignment_word_index = opcode == spv::OpLoad ? 5 : 4;  // OpStore is at [4]
        if (inst.Length() < alignment_word_index) {
            return false;
        }
        const uint32_t memory_operands = inst.Word(memory_operand_index);
        if ((memory_operands & spv::MemoryAccessAlignedMask) == 0) {
            return false;
        }
        // Even if they are other Memory Operands the spec says it is ordered by smallest bit first,
        // Luckily |Aligned| is the smallest bit that can have an operand so we know it is here
        meta.alignment_literal = inst.Word(alignment_word_index);

        // Aligned 0 was not being validated (https://github.com/KhronosGroup/glslang/issues/3893)
        // This is nonsense and we should skip (as it should be validated in spirv-val)
        if (!IsPowerOfTwo(meta.alignment_literal)) return false;
    } else if (AtomicOperation(opcode)) {
        // Atomics are naturally aligned and by setting this to 1, it will always pass the alignment check
        meta.alignment_literal = 1;
    } else {
        return false;
    }

    // While the Pointer Id might not be an OpAccessChain (can be OpLoad, OpCopyObject, etc), we can just examine its result type to
    // see if it is a PhysicalStorageBuffer pointer or not
    const uint32_t pointer_id = inst.Operand(0);
    const Instruction* pointer_inst = function.FindInstruction(pointer_id);
    if (!pointer_inst) {
        return false;  // Can be pointing to a Workgroup variable out of the function
    }

    // Get the OpTypePointer
    const Type* op_type_pointer = module_.type_manager_.FindTypeById(pointer_inst->TypeId());
    if (!op_type_pointer || op_type_pointer->spv_type_ != SpvType::kPointer ||
        op_type_pointer->inst_.Operand(0) != spv::StorageClassPhysicalStorageBuffer) {
        return false;
    }

    // The OpTypePointer's type
    uint32_t accessed_type_id = op_type_pointer->inst_.Operand(1);
    const Type* accessed_type = module_.type_manager_.FindTypeById(accessed_type_id);
    if (!accessed_type) {
        assert(false);
        return false;
    }

    // This might be an OpTypeStruct, even if some compilers are smart enough (know Mesa is) to detect only the first part of a
    // struct is loaded, we have to assume the entire struct is loaded and the entire memory is accessed (see
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8089)
    meta.type_length = module_.type_manager_.TypeLength(*accessed_type);
    // Will mark this is a struct acess to inform the user
    meta.type_is_struct = accessed_type->spv_type_ == SpvType::kStruct;

    // opaccesschain -> OpLoad/OpBitcast -> OpTypePointer (PSB) -> OpTypeStruct
    if (pre_pass && pointer_inst->IsAccessChain()) {
        std::vector<const Instruction*> access_chain_insts;

        const Instruction* next_inst = pointer_inst;
        // First walk back to the outer most access chain
        while (next_inst && next_inst->IsAccessChain()) {
            access_chain_insts.push_back(next_inst);
            const uint32_t access_chain_base_id = next_inst->Operand(0);
            next_inst = function.FindInstruction(access_chain_base_id);
        }
        if (!access_chain_insts.empty() && next_inst) {
            const Type* load_type_pointer = module_.type_manager_.FindTypeById(next_inst->TypeId());
            if (load_type_pointer && load_type_pointer->spv_type_ == SpvType::kPointer &&
                load_type_pointer->inst_.StorageClass() == spv::StorageClassPhysicalStorageBuffer) {
                const Type* struct_type = module_.type_manager_.FindTypeById(load_type_pointer->inst_.Operand(1));
                if (struct_type && struct_type->spv_type_ == SpvType::kStruct) {
                    const uint32_t struct_offset = FindLastByteOffset(struct_type->Id(), access_chain_insts);
                    // printf("struct_offset = %u\n", struct_offset);
                    if (struct_offset != 0) {
                        Range& range = block_struct_range_map_[struct_type->Id()];
                        range.begin = std::min(range.begin, struct_offset);
                        range.end = std::max(range.end, struct_offset);
                        range.instructions.insert(inst.GetPositionIndex());
                        block_skip_list_.insert(inst.GetPositionIndex());
                    }
                }
            }
        }
    }

    meta.target_instruction = &inst;
    return true;
}

bool BufferDeviceAddressPass::Instrument() {
    // Can safely loop function list as there is no injecting of new Functions until linking time
    for (const auto& function : module_.functions_) {
        if (function->instrumentation_added_) continue;

        BasicBlock& first_block = function->GetFirstBlock();
        InstructionIt first_injectable_instruction = first_block.GetFirstInjectableInstrution();
        ClearPrivateVariable(first_block, &first_injectable_instruction);

        for (auto block_it = function->blocks_.begin(); block_it != function->blocks_.end(); ++block_it) {
            BasicBlock& current_block = **block_it;

            cf_.Update(current_block);
            if (debug_disable_loops_ && cf_.in_loop) continue;

            if (current_block.IsLoopHeader()) {
                continue;  // Currently can't properly handle injecting CFG logic into a loop header block
            }
            auto& block_instructions = current_block.instructions_;

            block_struct_range_map_.clear();
            block_skip_list_.clear();

            for (auto inst_it = block_instructions.begin(); inst_it != block_instructions.end(); ++inst_it) {
                InstructionMeta meta;
                if (!RequiresInstrumentation(*function, *(inst_it->get()), meta, true)) continue;
            }
            printf("---[ %u ] skip %zu ---\n", module_.settings_.shader_id, block_skip_list_.size());
            for (const auto& [struct_id, range] : block_struct_range_map_) {
                printf("\t[%u] | (%zu) | %u - %u\n", struct_id, range.instructions.size(), range.begin, range.end);
            }

            for (auto inst_it = block_instructions.begin(); inst_it != block_instructions.end(); ++inst_it) {
                InstructionMeta meta;
                // Every instruction is analyzed by the specific pass and lets us know if we need to inject a function or not
                if (!RequiresInstrumentation(*function, *(inst_it->get()), meta, false)) {
                    if (function->IsReturn(*(inst_it->get()))) {
                        CreateFunctionCallAlignmentReport(current_block, &inst_it);
                    }
                    continue;
                }

                if (IsMaxInstrumentationsCount()) continue;
                instrumentations_count_++;

                InjectionData injection_data = GetInjectionData(*function, current_block, inst_it, *meta.target_instruction);

                if (module_.settings_.unsafe_mode) {
                    CreateFunctionCall(current_block, &inst_it, injection_data, meta);
                } else {
                    InjectConditionalData ic_data = InjectFunctionPre(*function.get(), block_it, inst_it);
                    ic_data.function_result_id = CreateFunctionCall(current_block, nullptr, injection_data, meta);
                    InjectFunctionPost(current_block, ic_data);
                    // Skip the newly added valid and invalid block. Start searching again from newly split merge block
                    block_it++;
                    block_it++;
                    break;
                }
            }
        }
    }

    return instrumentations_count_ != 0;
}

void BufferDeviceAddressPass::PrintDebugInfo() const {
    std::cout << "BufferDeviceAddressPass instrumentation count: " << instrumentations_count_ << '\n';
}

}  // namespace spirv
}  // namespace gpuav