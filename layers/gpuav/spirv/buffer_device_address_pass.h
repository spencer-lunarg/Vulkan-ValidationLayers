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
#pragma once

#include <stdint.h>
#include "containers/custom_containers.h"
#include "containers/limits.h"
#include "pass.h"

namespace gpuav {
namespace spirv {

// Create a pass to instrument physical buffer address checking
// This pass instruments all physical buffer address references to check that
// all referenced bytes fall in a valid buffer.
class BufferDeviceAddressPass : public Pass {
  public:
    BufferDeviceAddressPass(Module& module);
    const char* Name() const final { return "BufferDeviceAddressPass"; }
    bool Instrument() final;
    void PrintDebugInfo() const final;

  private:
    // This is metadata tied to a single instruction gathered during RequiresInstrumentation() to be used later
    struct InstructionMeta {
        const Instruction* target_instruction = nullptr;
        uint32_t alignment_literal = 0;
        uint32_t type_length = 0;
        bool type_is_struct = false;
    };

    bool RequiresInstrumentation(const Function& function, const Instruction& inst, InstructionMeta& meta, bool pre_pass);
    uint32_t CreateFunctionCall(BasicBlock& block, InstructionIt* inst_it, const InjectionData& injection_data,
                                const InstructionMeta& meta);

    // Called when a Function starts
    void ClearPrivateVariable(BasicBlock& block, InstructionIt* inst_it);
    // Called when a Function ends
    void CreateFunctionCallAlignmentReport(BasicBlock& block, InstructionIt* inst_it);

    uint32_t FindLastByteOffset(uint32_t bda_struct_id, const std::vector<const Instruction*>& access_chain_insts) const;

    uint32_t GetLinkFunctionId();
    uint32_t GetLinkFunctionIdAlignmentCheck();
    uint32_t GetLinkFunctionIdAlignmentReport();

    uint32_t invalid_alignment_priavte_variable_id_ = 0;
    uint32_t function_alignment_checks_ = 0;

    // Function IDs to link in
    uint32_t link_function_id_ = 0;
    uint32_t link_alignment_check_id_ = 0;
    uint32_t link_alignment_report_id_ = 0;

    struct Range {
        uint32_t begin = vvl::kU32Max;
        uint32_t begin_instruction = 0;
        uint32_t end = 0;
        uint32_t end_instruction = 0;
        vvl::unordered_set<uint32_t> instructions;
    };

    vvl::unordered_map<uint32_t, Range> block_struct_range_map_;
    vvl::unordered_set<uint32_t> block_skip_list_;
};

}  // namespace spirv
}  // namespace gpuav