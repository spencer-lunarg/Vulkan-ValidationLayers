// *** THIS FILE IS GENERATED - DO NOT EDIT ***
// See extension_helper_generator.py for modifications

/***************************************************************************
 *
 * Copyright (c) 2015-2024 The Khronos Group Inc.
 * Copyright (c) 2015-2024 Valve Corporation
 * Copyright (c) 2015-2024 LunarG, Inc.
 * Copyright (c) 2015-2024 Google Inc.
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
 ****************************************************************************/

// NOLINTBEGIN

#pragma once

#include <string>
#include <utility>
#include <array>
#include <vector>
#include <cassert>

#include <vulkan/vulkan.h>
#include "containers/custom_containers.h"
#include "generated/vk_api_version.h"
#include "generated/error_location_helper.h"

// Extensions (unlike functions, struct, etc) are passed in as strings.
// The goal is to turn the string to a enum and pass that around the layers.
// Only when we need to print an error do we try and turn it back into a string
vvl::Extension GetExtension(std::string extension);

enum ExtEnabled : unsigned char {
    kNotEnabled,
    kEnabledByCreateinfo,
    kEnabledByApiLevel,
    kEnabledByInteraction,
};

// Map of promoted extension information per version (a separate map exists for instance and device extensions).
// The map is keyed by the version number (e.g. VK_API_VERSION_1_1) and each value is a pair consisting of the
// version string (e.g. "VK_VERSION_1_1") and the set of name of the promoted extensions.
typedef vvl::unordered_map<uint32_t, std::pair<const char *, vvl::unordered_set<vvl::Extension>>> PromotedExtensionInfoMap;

/*
This function is a helper to know if the extension is enabled.

Times to use it
- To determine the VUID
- The VU mentions the use of the extension
- Extension exposes property limits being validated
- Checking not enabled
    - if (!IsExtEnabled(...)) { }
- Special extensions that being EXPOSED alters the VUs
    - IsExtEnabled(device_extensions.vk_khr_portability_subset)
- Special extensions that alter behaviour of enabled
    - IsExtEnabled(device_extensions.vk_khr_maintenance*)

Times to NOT use it
    - If checking if a struct or enum is being used. There are a stateless checks
      to make sure the new Structs/Enums are not being used without this enabled.
    - If checking if the extension's feature enable status, because if the feature
      is enabled, then we already validated that extension is enabled.
    - Some variables (ex. viewMask) require the extension to be used if non-zero
*/
[[maybe_unused]] static bool IsExtEnabled(ExtEnabled extension) { return (extension != kNotEnabled); }

[[maybe_unused]] static bool IsExtEnabledByCreateinfo(ExtEnabled extension) { return (extension == kEnabledByCreateinfo); }

struct InstanceExtensions {
    ExtEnabled vk_feature_version_1_1{kNotEnabled};
    ExtEnabled vk_feature_version_1_2{kNotEnabled};
    ExtEnabled vk_feature_version_1_3{kNotEnabled};
    ExtEnabled vk_khr_surface{kNotEnabled};
    ExtEnabled vk_khr_display{kNotEnabled};
    ExtEnabled vk_khr_xlib_surface{kNotEnabled};
    ExtEnabled vk_khr_xcb_surface{kNotEnabled};
    ExtEnabled vk_khr_wayland_surface{kNotEnabled};
    ExtEnabled vk_khr_android_surface{kNotEnabled};
    ExtEnabled vk_khr_win32_surface{kNotEnabled};
    ExtEnabled vk_khr_get_physical_device_properties2{kNotEnabled};
    ExtEnabled vk_khr_device_group_creation{kNotEnabled};
    ExtEnabled vk_khr_external_memory_capabilities{kNotEnabled};
    ExtEnabled vk_khr_external_semaphore_capabilities{kNotEnabled};
    ExtEnabled vk_khr_external_fence_capabilities{kNotEnabled};
    ExtEnabled vk_khr_get_surface_capabilities2{kNotEnabled};
    ExtEnabled vk_khr_get_display_properties2{kNotEnabled};
    ExtEnabled vk_khr_surface_protected_capabilities{kNotEnabled};
    ExtEnabled vk_khr_portability_enumeration{kNotEnabled};
    ExtEnabled vk_ext_debug_report{kNotEnabled};
    ExtEnabled vk_ggp_stream_descriptor_surface{kNotEnabled};
    ExtEnabled vk_nv_external_memory_capabilities{kNotEnabled};
    ExtEnabled vk_ext_validation_flags{kNotEnabled};
    ExtEnabled vk_nn_vi_surface{kNotEnabled};
    ExtEnabled vk_ext_direct_mode_display{kNotEnabled};
    ExtEnabled vk_ext_acquire_xlib_display{kNotEnabled};
    ExtEnabled vk_ext_display_surface_counter{kNotEnabled};
    ExtEnabled vk_ext_swapchain_colorspace{kNotEnabled};
    ExtEnabled vk_mvk_ios_surface{kNotEnabled};
    ExtEnabled vk_mvk_macos_surface{kNotEnabled};
    ExtEnabled vk_ext_debug_utils{kNotEnabled};
    ExtEnabled vk_fuchsia_imagepipe_surface{kNotEnabled};
    ExtEnabled vk_ext_metal_surface{kNotEnabled};
    ExtEnabled vk_ext_validation_features{kNotEnabled};
    ExtEnabled vk_ext_headless_surface{kNotEnabled};
    ExtEnabled vk_ext_surface_maintenance1{kNotEnabled};
    ExtEnabled vk_ext_acquire_drm_display{kNotEnabled};
    ExtEnabled vk_ext_directfb_surface{kNotEnabled};
    ExtEnabled vk_qnx_screen_surface{kNotEnabled};
    ExtEnabled vk_google_surfaceless_query{kNotEnabled};
    ExtEnabled vk_lunarg_direct_driver_loading{kNotEnabled};
    ExtEnabled vk_ext_layer_settings{kNotEnabled};

    static const PromotedExtensionInfoMap &get_promotion_info_map() {
        static const PromotedExtensionInfoMap promoted_map = {
            {VK_API_VERSION_1_1,
             {"VK_VERSION_1_1",
              {
                  vvl::Extension::KHR_get_physical_device_properties2,
                  vvl::Extension::KHR_device_group_creation,
                  vvl::Extension::KHR_external_memory_capabilities,
                  vvl::Extension::KHR_external_semaphore_capabilities,
                  vvl::Extension::KHR_external_fence_capabilities,
              }}},

        };
        return promoted_map;
    }

    struct InstanceReq {
        const ExtEnabled InstanceExtensions::*enabled;
        const char *name;
    };
    typedef std::vector<InstanceReq> InstanceReqVec;
    struct InstanceInfo {
        InstanceInfo(ExtEnabled InstanceExtensions::*state_, const InstanceReqVec requirements_)
            : state(state_), requirements(requirements_) {}
        ExtEnabled InstanceExtensions::*state;
        InstanceReqVec requirements;
    };

    typedef vvl::unordered_map<vvl::Extension, InstanceInfo> InstanceInfoMap;
    static const InstanceInfoMap &get_info_map() {
        static const InstanceInfoMap info_map = {
            {vvl::Extension::KHR_surface, InstanceInfo(&InstanceExtensions::vk_khr_surface, {})},
            {vvl::Extension::KHR_display, InstanceInfo(&InstanceExtensions::vk_khr_display,
                                                       {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_XLIB_KHR
            {vvl::Extension::KHR_xlib_surface,
             InstanceInfo(&InstanceExtensions::vk_khr_xlib_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_XLIB_KHR
#ifdef VK_USE_PLATFORM_XCB_KHR
            {vvl::Extension::KHR_xcb_surface,
             InstanceInfo(&InstanceExtensions::vk_khr_xcb_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_XCB_KHR
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
            {vvl::Extension::KHR_wayland_surface,
             InstanceInfo(&InstanceExtensions::vk_khr_wayland_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WAYLAND_KHR
#ifdef VK_USE_PLATFORM_ANDROID_KHR
            {vvl::Extension::KHR_android_surface,
             InstanceInfo(&InstanceExtensions::vk_khr_android_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_ANDROID_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_win32_surface,
             InstanceInfo(&InstanceExtensions::vk_khr_win32_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_get_physical_device_properties2,
             InstanceInfo(&InstanceExtensions::vk_khr_get_physical_device_properties2, {})},
            {vvl::Extension::KHR_device_group_creation, InstanceInfo(&InstanceExtensions::vk_khr_device_group_creation, {})},
            {vvl::Extension::KHR_external_memory_capabilities,
             InstanceInfo(&InstanceExtensions::vk_khr_external_memory_capabilities,
                          {{{&InstanceExtensions::vk_khr_get_physical_device_properties2,
                             VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_external_semaphore_capabilities,
             InstanceInfo(&InstanceExtensions::vk_khr_external_semaphore_capabilities,
                          {{{&InstanceExtensions::vk_khr_get_physical_device_properties2,
                             VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_external_fence_capabilities,
             InstanceInfo(&InstanceExtensions::vk_khr_external_fence_capabilities,
                          {{{&InstanceExtensions::vk_khr_get_physical_device_properties2,
                             VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_get_surface_capabilities2,
             InstanceInfo(&InstanceExtensions::vk_khr_get_surface_capabilities2,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_get_display_properties2,
             InstanceInfo(&InstanceExtensions::vk_khr_get_display_properties2,
                          {{{&InstanceExtensions::vk_khr_display, VK_KHR_DISPLAY_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_surface_protected_capabilities,
             InstanceInfo(
                 &InstanceExtensions::vk_khr_surface_protected_capabilities,
                 {{{&InstanceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                   {&InstanceExtensions::vk_khr_get_surface_capabilities2, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_portability_enumeration, InstanceInfo(&InstanceExtensions::vk_khr_portability_enumeration, {})},
            {vvl::Extension::EXT_debug_report, InstanceInfo(&InstanceExtensions::vk_ext_debug_report, {})},
#ifdef VK_USE_PLATFORM_GGP
            {vvl::Extension::GGP_stream_descriptor_surface,
             InstanceInfo(&InstanceExtensions::vk_ggp_stream_descriptor_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_GGP
            {vvl::Extension::NV_external_memory_capabilities,
             InstanceInfo(&InstanceExtensions::vk_nv_external_memory_capabilities, {})},
            {vvl::Extension::EXT_validation_flags, InstanceInfo(&InstanceExtensions::vk_ext_validation_flags, {})},
#ifdef VK_USE_PLATFORM_VI_NN
            {vvl::Extension::NN_vi_surface, InstanceInfo(&InstanceExtensions::vk_nn_vi_surface,
                                                         {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_VI_NN
            {vvl::Extension::EXT_direct_mode_display,
             InstanceInfo(&InstanceExtensions::vk_ext_direct_mode_display,
                          {{{&InstanceExtensions::vk_khr_display, VK_KHR_DISPLAY_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
            {vvl::Extension::EXT_acquire_xlib_display,
             InstanceInfo(&InstanceExtensions::vk_ext_acquire_xlib_display,
                          {{{&InstanceExtensions::vk_ext_direct_mode_display, VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_XLIB_XRANDR_EXT
            {vvl::Extension::EXT_display_surface_counter,
             InstanceInfo(&InstanceExtensions::vk_ext_display_surface_counter,
                          {{{&InstanceExtensions::vk_khr_display, VK_KHR_DISPLAY_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_swapchain_colorspace,
             InstanceInfo(&InstanceExtensions::vk_ext_swapchain_colorspace,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_IOS_MVK
            {vvl::Extension::MVK_ios_surface,
             InstanceInfo(&InstanceExtensions::vk_mvk_ios_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_IOS_MVK
#ifdef VK_USE_PLATFORM_MACOS_MVK
            {vvl::Extension::MVK_macos_surface,
             InstanceInfo(&InstanceExtensions::vk_mvk_macos_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_MACOS_MVK
            {vvl::Extension::EXT_debug_utils, InstanceInfo(&InstanceExtensions::vk_ext_debug_utils, {})},
#ifdef VK_USE_PLATFORM_FUCHSIA
            {vvl::Extension::FUCHSIA_imagepipe_surface,
             InstanceInfo(&InstanceExtensions::vk_fuchsia_imagepipe_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_FUCHSIA
#ifdef VK_USE_PLATFORM_METAL_EXT
            {vvl::Extension::EXT_metal_surface,
             InstanceInfo(&InstanceExtensions::vk_ext_metal_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_METAL_EXT
            {vvl::Extension::EXT_validation_features, InstanceInfo(&InstanceExtensions::vk_ext_validation_features, {})},
            {vvl::Extension::EXT_headless_surface,
             InstanceInfo(&InstanceExtensions::vk_ext_headless_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_surface_maintenance1,
             InstanceInfo(
                 &InstanceExtensions::vk_ext_surface_maintenance1,
                 {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME},
                   {&InstanceExtensions::vk_khr_get_surface_capabilities2, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_acquire_drm_display,
             InstanceInfo(&InstanceExtensions::vk_ext_acquire_drm_display,
                          {{{&InstanceExtensions::vk_ext_direct_mode_display, VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
            {vvl::Extension::EXT_directfb_surface,
             InstanceInfo(&InstanceExtensions::vk_ext_directfb_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_DIRECTFB_EXT
#ifdef VK_USE_PLATFORM_SCREEN_QNX
            {vvl::Extension::QNX_screen_surface,
             InstanceInfo(&InstanceExtensions::vk_qnx_screen_surface,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_SCREEN_QNX
            {vvl::Extension::GOOGLE_surfaceless_query,
             InstanceInfo(&InstanceExtensions::vk_google_surfaceless_query,
                          {{{&InstanceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
            {vvl::Extension::LUNARG_direct_driver_loading, InstanceInfo(&InstanceExtensions::vk_lunarg_direct_driver_loading, {})},
            {vvl::Extension::EXT_layer_settings, InstanceInfo(&InstanceExtensions::vk_ext_layer_settings, {})},

        };
        return info_map;
    }

    static const InstanceInfo &get_version_map(const char *version) {
        static const InstanceInfo empty_info{nullptr, InstanceReqVec()};
        static const vvl::unordered_map<std::string, InstanceInfo> version_map = {
            {"VK_VERSION_1_1", InstanceInfo(&InstanceExtensions::vk_feature_version_1_1, {})},
            {"VK_VERSION_1_2", InstanceInfo(&InstanceExtensions::vk_feature_version_1_2, {})},
            {"VK_VERSION_1_3", InstanceInfo(&InstanceExtensions::vk_feature_version_1_3, {})},
        };
        const auto info = version_map.find(version);
        return (info != version_map.cend()) ? info->second : empty_info;
    }

    static const InstanceInfo &get_info(vvl::Extension extension_name) {
        static const InstanceInfo empty_info{nullptr, InstanceReqVec()};
        const auto &ext_map = InstanceExtensions::get_info_map();
        const auto info = ext_map.find(extension_name);
        return (info != ext_map.cend()) ? info->second : empty_info;
    }

    APIVersion InitFromInstanceCreateInfo(APIVersion requested_api_version, const VkInstanceCreateInfo *pCreateInfo);
};

struct DeviceExtensions : public InstanceExtensions {
    ExtEnabled vk_feature_version_1_1{kNotEnabled};
    ExtEnabled vk_feature_version_1_2{kNotEnabled};
    ExtEnabled vk_feature_version_1_3{kNotEnabled};
    ExtEnabled vk_khr_swapchain{kNotEnabled};
    ExtEnabled vk_khr_display_swapchain{kNotEnabled};
    ExtEnabled vk_khr_sampler_mirror_clamp_to_edge{kNotEnabled};
    ExtEnabled vk_khr_video_queue{kNotEnabled};
    ExtEnabled vk_khr_video_decode_queue{kNotEnabled};
    ExtEnabled vk_khr_video_encode_h264{kNotEnabled};
    ExtEnabled vk_khr_video_encode_h265{kNotEnabled};
    ExtEnabled vk_khr_video_decode_h264{kNotEnabled};
    ExtEnabled vk_khr_dynamic_rendering{kNotEnabled};
    ExtEnabled vk_khr_multiview{kNotEnabled};
    ExtEnabled vk_khr_device_group{kNotEnabled};
    ExtEnabled vk_khr_shader_draw_parameters{kNotEnabled};
    ExtEnabled vk_khr_maintenance1{kNotEnabled};
    ExtEnabled vk_khr_external_memory{kNotEnabled};
    ExtEnabled vk_khr_external_memory_win32{kNotEnabled};
    ExtEnabled vk_khr_external_memory_fd{kNotEnabled};
    ExtEnabled vk_khr_win32_keyed_mutex{kNotEnabled};
    ExtEnabled vk_khr_external_semaphore{kNotEnabled};
    ExtEnabled vk_khr_external_semaphore_win32{kNotEnabled};
    ExtEnabled vk_khr_external_semaphore_fd{kNotEnabled};
    ExtEnabled vk_khr_push_descriptor{kNotEnabled};
    ExtEnabled vk_khr_shader_float16_int8{kNotEnabled};
    ExtEnabled vk_khr_16bit_storage{kNotEnabled};
    ExtEnabled vk_khr_incremental_present{kNotEnabled};
    ExtEnabled vk_khr_descriptor_update_template{kNotEnabled};
    ExtEnabled vk_khr_imageless_framebuffer{kNotEnabled};
    ExtEnabled vk_khr_create_renderpass2{kNotEnabled};
    ExtEnabled vk_khr_shared_presentable_image{kNotEnabled};
    ExtEnabled vk_khr_external_fence{kNotEnabled};
    ExtEnabled vk_khr_external_fence_win32{kNotEnabled};
    ExtEnabled vk_khr_external_fence_fd{kNotEnabled};
    ExtEnabled vk_khr_performance_query{kNotEnabled};
    ExtEnabled vk_khr_maintenance2{kNotEnabled};
    ExtEnabled vk_khr_variable_pointers{kNotEnabled};
    ExtEnabled vk_khr_dedicated_allocation{kNotEnabled};
    ExtEnabled vk_khr_storage_buffer_storage_class{kNotEnabled};
    ExtEnabled vk_khr_relaxed_block_layout{kNotEnabled};
    ExtEnabled vk_khr_get_memory_requirements2{kNotEnabled};
    ExtEnabled vk_khr_image_format_list{kNotEnabled};
    ExtEnabled vk_khr_sampler_ycbcr_conversion{kNotEnabled};
    ExtEnabled vk_khr_bind_memory2{kNotEnabled};
    ExtEnabled vk_khr_portability_subset{kNotEnabled};
    ExtEnabled vk_khr_maintenance3{kNotEnabled};
    ExtEnabled vk_khr_draw_indirect_count{kNotEnabled};
    ExtEnabled vk_khr_shader_subgroup_extended_types{kNotEnabled};
    ExtEnabled vk_khr_8bit_storage{kNotEnabled};
    ExtEnabled vk_khr_shader_atomic_int64{kNotEnabled};
    ExtEnabled vk_khr_shader_clock{kNotEnabled};
    ExtEnabled vk_khr_video_decode_h265{kNotEnabled};
    ExtEnabled vk_khr_global_priority{kNotEnabled};
    ExtEnabled vk_khr_driver_properties{kNotEnabled};
    ExtEnabled vk_khr_shader_float_controls{kNotEnabled};
    ExtEnabled vk_khr_depth_stencil_resolve{kNotEnabled};
    ExtEnabled vk_khr_swapchain_mutable_format{kNotEnabled};
    ExtEnabled vk_khr_timeline_semaphore{kNotEnabled};
    ExtEnabled vk_khr_vulkan_memory_model{kNotEnabled};
    ExtEnabled vk_khr_shader_terminate_invocation{kNotEnabled};
    ExtEnabled vk_khr_fragment_shading_rate{kNotEnabled};
    ExtEnabled vk_khr_dynamic_rendering_local_read{kNotEnabled};
    ExtEnabled vk_khr_shader_quad_control{kNotEnabled};
    ExtEnabled vk_khr_spirv_1_4{kNotEnabled};
    ExtEnabled vk_khr_separate_depth_stencil_layouts{kNotEnabled};
    ExtEnabled vk_khr_present_wait{kNotEnabled};
    ExtEnabled vk_khr_uniform_buffer_standard_layout{kNotEnabled};
    ExtEnabled vk_khr_buffer_device_address{kNotEnabled};
    ExtEnabled vk_khr_deferred_host_operations{kNotEnabled};
    ExtEnabled vk_khr_pipeline_executable_properties{kNotEnabled};
    ExtEnabled vk_khr_map_memory2{kNotEnabled};
    ExtEnabled vk_khr_shader_integer_dot_product{kNotEnabled};
    ExtEnabled vk_khr_pipeline_library{kNotEnabled};
    ExtEnabled vk_khr_shader_non_semantic_info{kNotEnabled};
    ExtEnabled vk_khr_present_id{kNotEnabled};
    ExtEnabled vk_khr_video_encode_queue{kNotEnabled};
    ExtEnabled vk_khr_synchronization2{kNotEnabled};
    ExtEnabled vk_khr_fragment_shader_barycentric{kNotEnabled};
    ExtEnabled vk_khr_shader_subgroup_uniform_control_flow{kNotEnabled};
    ExtEnabled vk_khr_zero_initialize_workgroup_memory{kNotEnabled};
    ExtEnabled vk_khr_workgroup_memory_explicit_layout{kNotEnabled};
    ExtEnabled vk_khr_copy_commands2{kNotEnabled};
    ExtEnabled vk_khr_format_feature_flags2{kNotEnabled};
    ExtEnabled vk_khr_ray_tracing_maintenance1{kNotEnabled};
    ExtEnabled vk_khr_maintenance4{kNotEnabled};
    ExtEnabled vk_khr_shader_subgroup_rotate{kNotEnabled};
    ExtEnabled vk_khr_shader_maximal_reconvergence{kNotEnabled};
    ExtEnabled vk_khr_maintenance5{kNotEnabled};
    ExtEnabled vk_khr_ray_tracing_position_fetch{kNotEnabled};
    ExtEnabled vk_khr_cooperative_matrix{kNotEnabled};
    ExtEnabled vk_khr_video_maintenance1{kNotEnabled};
    ExtEnabled vk_khr_vertex_attribute_divisor{kNotEnabled};
    ExtEnabled vk_khr_load_store_op_none{kNotEnabled};
    ExtEnabled vk_khr_shader_float_controls2{kNotEnabled};
    ExtEnabled vk_khr_index_type_uint8{kNotEnabled};
    ExtEnabled vk_khr_line_rasterization{kNotEnabled};
    ExtEnabled vk_khr_calibrated_timestamps{kNotEnabled};
    ExtEnabled vk_khr_shader_expect_assume{kNotEnabled};
    ExtEnabled vk_khr_maintenance6{kNotEnabled};
    ExtEnabled vk_nv_glsl_shader{kNotEnabled};
    ExtEnabled vk_ext_depth_range_unrestricted{kNotEnabled};
    ExtEnabled vk_img_filter_cubic{kNotEnabled};
    ExtEnabled vk_amd_rasterization_order{kNotEnabled};
    ExtEnabled vk_amd_shader_trinary_minmax{kNotEnabled};
    ExtEnabled vk_amd_shader_explicit_vertex_parameter{kNotEnabled};
    ExtEnabled vk_ext_debug_marker{kNotEnabled};
    ExtEnabled vk_amd_gcn_shader{kNotEnabled};
    ExtEnabled vk_nv_dedicated_allocation{kNotEnabled};
    ExtEnabled vk_ext_transform_feedback{kNotEnabled};
    ExtEnabled vk_nvx_binary_import{kNotEnabled};
    ExtEnabled vk_nvx_image_view_handle{kNotEnabled};
    ExtEnabled vk_amd_draw_indirect_count{kNotEnabled};
    ExtEnabled vk_amd_negative_viewport_height{kNotEnabled};
    ExtEnabled vk_amd_gpu_shader_half_float{kNotEnabled};
    ExtEnabled vk_amd_shader_ballot{kNotEnabled};
    ExtEnabled vk_amd_texture_gather_bias_lod{kNotEnabled};
    ExtEnabled vk_amd_shader_info{kNotEnabled};
    ExtEnabled vk_amd_shader_image_load_store_lod{kNotEnabled};
    ExtEnabled vk_nv_corner_sampled_image{kNotEnabled};
    ExtEnabled vk_img_format_pvrtc{kNotEnabled};
    ExtEnabled vk_nv_external_memory{kNotEnabled};
    ExtEnabled vk_nv_external_memory_win32{kNotEnabled};
    ExtEnabled vk_nv_win32_keyed_mutex{kNotEnabled};
    ExtEnabled vk_ext_shader_subgroup_ballot{kNotEnabled};
    ExtEnabled vk_ext_shader_subgroup_vote{kNotEnabled};
    ExtEnabled vk_ext_texture_compression_astc_hdr{kNotEnabled};
    ExtEnabled vk_ext_astc_decode_mode{kNotEnabled};
    ExtEnabled vk_ext_pipeline_robustness{kNotEnabled};
    ExtEnabled vk_ext_conditional_rendering{kNotEnabled};
    ExtEnabled vk_nv_clip_space_w_scaling{kNotEnabled};
    ExtEnabled vk_ext_display_control{kNotEnabled};
    ExtEnabled vk_google_display_timing{kNotEnabled};
    ExtEnabled vk_nv_sample_mask_override_coverage{kNotEnabled};
    ExtEnabled vk_nv_geometry_shader_passthrough{kNotEnabled};
    ExtEnabled vk_nv_viewport_array2{kNotEnabled};
    ExtEnabled vk_nvx_multiview_per_view_attributes{kNotEnabled};
    ExtEnabled vk_nv_viewport_swizzle{kNotEnabled};
    ExtEnabled vk_ext_discard_rectangles{kNotEnabled};
    ExtEnabled vk_ext_conservative_rasterization{kNotEnabled};
    ExtEnabled vk_ext_depth_clip_enable{kNotEnabled};
    ExtEnabled vk_ext_hdr_metadata{kNotEnabled};
    ExtEnabled vk_img_relaxed_line_rasterization{kNotEnabled};
    ExtEnabled vk_ext_external_memory_dma_buf{kNotEnabled};
    ExtEnabled vk_ext_queue_family_foreign{kNotEnabled};
    ExtEnabled vk_android_external_memory_android_hardware_buffer{kNotEnabled};
    ExtEnabled vk_ext_sampler_filter_minmax{kNotEnabled};
    ExtEnabled vk_amd_gpu_shader_int16{kNotEnabled};
    ExtEnabled vk_amdx_shader_enqueue{kNotEnabled};
    ExtEnabled vk_amd_mixed_attachment_samples{kNotEnabled};
    ExtEnabled vk_amd_shader_fragment_mask{kNotEnabled};
    ExtEnabled vk_ext_inline_uniform_block{kNotEnabled};
    ExtEnabled vk_ext_shader_stencil_export{kNotEnabled};
    ExtEnabled vk_ext_sample_locations{kNotEnabled};
    ExtEnabled vk_ext_blend_operation_advanced{kNotEnabled};
    ExtEnabled vk_nv_fragment_coverage_to_color{kNotEnabled};
    ExtEnabled vk_nv_framebuffer_mixed_samples{kNotEnabled};
    ExtEnabled vk_nv_fill_rectangle{kNotEnabled};
    ExtEnabled vk_nv_shader_sm_builtins{kNotEnabled};
    ExtEnabled vk_ext_post_depth_coverage{kNotEnabled};
    ExtEnabled vk_ext_image_drm_format_modifier{kNotEnabled};
    ExtEnabled vk_ext_validation_cache{kNotEnabled};
    ExtEnabled vk_ext_descriptor_indexing{kNotEnabled};
    ExtEnabled vk_ext_shader_viewport_index_layer{kNotEnabled};
    ExtEnabled vk_nv_shading_rate_image{kNotEnabled};
    ExtEnabled vk_nv_ray_tracing{kNotEnabled};
    ExtEnabled vk_nv_representative_fragment_test{kNotEnabled};
    ExtEnabled vk_ext_filter_cubic{kNotEnabled};
    ExtEnabled vk_qcom_render_pass_shader_resolve{kNotEnabled};
    ExtEnabled vk_ext_global_priority{kNotEnabled};
    ExtEnabled vk_ext_external_memory_host{kNotEnabled};
    ExtEnabled vk_amd_buffer_marker{kNotEnabled};
    ExtEnabled vk_amd_pipeline_compiler_control{kNotEnabled};
    ExtEnabled vk_ext_calibrated_timestamps{kNotEnabled};
    ExtEnabled vk_amd_shader_core_properties{kNotEnabled};
    ExtEnabled vk_amd_memory_overallocation_behavior{kNotEnabled};
    ExtEnabled vk_ext_vertex_attribute_divisor{kNotEnabled};
    ExtEnabled vk_ggp_frame_token{kNotEnabled};
    ExtEnabled vk_ext_pipeline_creation_feedback{kNotEnabled};
    ExtEnabled vk_nv_shader_subgroup_partitioned{kNotEnabled};
    ExtEnabled vk_nv_compute_shader_derivatives{kNotEnabled};
    ExtEnabled vk_nv_mesh_shader{kNotEnabled};
    ExtEnabled vk_nv_fragment_shader_barycentric{kNotEnabled};
    ExtEnabled vk_nv_shader_image_footprint{kNotEnabled};
    ExtEnabled vk_nv_scissor_exclusive{kNotEnabled};
    ExtEnabled vk_nv_device_diagnostic_checkpoints{kNotEnabled};
    ExtEnabled vk_intel_shader_integer_functions2{kNotEnabled};
    ExtEnabled vk_intel_performance_query{kNotEnabled};
    ExtEnabled vk_ext_pci_bus_info{kNotEnabled};
    ExtEnabled vk_amd_display_native_hdr{kNotEnabled};
    ExtEnabled vk_ext_fragment_density_map{kNotEnabled};
    ExtEnabled vk_ext_scalar_block_layout{kNotEnabled};
    ExtEnabled vk_google_hlsl_functionality1{kNotEnabled};
    ExtEnabled vk_google_decorate_string{kNotEnabled};
    ExtEnabled vk_ext_subgroup_size_control{kNotEnabled};
    ExtEnabled vk_amd_shader_core_properties2{kNotEnabled};
    ExtEnabled vk_amd_device_coherent_memory{kNotEnabled};
    ExtEnabled vk_ext_shader_image_atomic_int64{kNotEnabled};
    ExtEnabled vk_ext_memory_budget{kNotEnabled};
    ExtEnabled vk_ext_memory_priority{kNotEnabled};
    ExtEnabled vk_nv_dedicated_allocation_image_aliasing{kNotEnabled};
    ExtEnabled vk_ext_buffer_device_address{kNotEnabled};
    ExtEnabled vk_ext_tooling_info{kNotEnabled};
    ExtEnabled vk_ext_separate_stencil_usage{kNotEnabled};
    ExtEnabled vk_nv_cooperative_matrix{kNotEnabled};
    ExtEnabled vk_nv_coverage_reduction_mode{kNotEnabled};
    ExtEnabled vk_ext_fragment_shader_interlock{kNotEnabled};
    ExtEnabled vk_ext_ycbcr_image_arrays{kNotEnabled};
    ExtEnabled vk_ext_provoking_vertex{kNotEnabled};
    ExtEnabled vk_ext_full_screen_exclusive{kNotEnabled};
    ExtEnabled vk_ext_line_rasterization{kNotEnabled};
    ExtEnabled vk_ext_shader_atomic_float{kNotEnabled};
    ExtEnabled vk_ext_host_query_reset{kNotEnabled};
    ExtEnabled vk_ext_index_type_uint8{kNotEnabled};
    ExtEnabled vk_ext_extended_dynamic_state{kNotEnabled};
    ExtEnabled vk_ext_host_image_copy{kNotEnabled};
    ExtEnabled vk_ext_shader_atomic_float2{kNotEnabled};
    ExtEnabled vk_ext_swapchain_maintenance1{kNotEnabled};
    ExtEnabled vk_ext_shader_demote_to_helper_invocation{kNotEnabled};
    ExtEnabled vk_nv_device_generated_commands{kNotEnabled};
    ExtEnabled vk_nv_inherited_viewport_scissor{kNotEnabled};
    ExtEnabled vk_ext_texel_buffer_alignment{kNotEnabled};
    ExtEnabled vk_qcom_render_pass_transform{kNotEnabled};
    ExtEnabled vk_ext_depth_bias_control{kNotEnabled};
    ExtEnabled vk_ext_device_memory_report{kNotEnabled};
    ExtEnabled vk_ext_robustness2{kNotEnabled};
    ExtEnabled vk_ext_custom_border_color{kNotEnabled};
    ExtEnabled vk_google_user_type{kNotEnabled};
    ExtEnabled vk_nv_present_barrier{kNotEnabled};
    ExtEnabled vk_ext_private_data{kNotEnabled};
    ExtEnabled vk_ext_pipeline_creation_cache_control{kNotEnabled};
    ExtEnabled vk_nv_device_diagnostics_config{kNotEnabled};
    ExtEnabled vk_qcom_render_pass_store_ops{kNotEnabled};
    ExtEnabled vk_nv_cuda_kernel_launch{kNotEnabled};
    ExtEnabled vk_nv_low_latency{kNotEnabled};
    ExtEnabled vk_ext_metal_objects{kNotEnabled};
    ExtEnabled vk_ext_descriptor_buffer{kNotEnabled};
    ExtEnabled vk_ext_graphics_pipeline_library{kNotEnabled};
    ExtEnabled vk_amd_shader_early_and_late_fragment_tests{kNotEnabled};
    ExtEnabled vk_nv_fragment_shading_rate_enums{kNotEnabled};
    ExtEnabled vk_nv_ray_tracing_motion_blur{kNotEnabled};
    ExtEnabled vk_ext_ycbcr_2plane_444_formats{kNotEnabled};
    ExtEnabled vk_ext_fragment_density_map2{kNotEnabled};
    ExtEnabled vk_qcom_rotated_copy_commands{kNotEnabled};
    ExtEnabled vk_ext_image_robustness{kNotEnabled};
    ExtEnabled vk_ext_image_compression_control{kNotEnabled};
    ExtEnabled vk_ext_attachment_feedback_loop_layout{kNotEnabled};
    ExtEnabled vk_ext_4444_formats{kNotEnabled};
    ExtEnabled vk_ext_device_fault{kNotEnabled};
    ExtEnabled vk_arm_rasterization_order_attachment_access{kNotEnabled};
    ExtEnabled vk_ext_rgba10x6_formats{kNotEnabled};
    ExtEnabled vk_nv_acquire_winrt_display{kNotEnabled};
    ExtEnabled vk_valve_mutable_descriptor_type{kNotEnabled};
    ExtEnabled vk_ext_vertex_input_dynamic_state{kNotEnabled};
    ExtEnabled vk_ext_physical_device_drm{kNotEnabled};
    ExtEnabled vk_ext_device_address_binding_report{kNotEnabled};
    ExtEnabled vk_ext_depth_clip_control{kNotEnabled};
    ExtEnabled vk_ext_primitive_topology_list_restart{kNotEnabled};
    ExtEnabled vk_fuchsia_external_memory{kNotEnabled};
    ExtEnabled vk_fuchsia_external_semaphore{kNotEnabled};
    ExtEnabled vk_fuchsia_buffer_collection{kNotEnabled};
    ExtEnabled vk_huawei_subpass_shading{kNotEnabled};
    ExtEnabled vk_huawei_invocation_mask{kNotEnabled};
    ExtEnabled vk_nv_external_memory_rdma{kNotEnabled};
    ExtEnabled vk_ext_pipeline_properties{kNotEnabled};
    ExtEnabled vk_ext_frame_boundary{kNotEnabled};
    ExtEnabled vk_ext_multisampled_render_to_single_sampled{kNotEnabled};
    ExtEnabled vk_ext_extended_dynamic_state2{kNotEnabled};
    ExtEnabled vk_ext_color_write_enable{kNotEnabled};
    ExtEnabled vk_ext_primitives_generated_query{kNotEnabled};
    ExtEnabled vk_ext_global_priority_query{kNotEnabled};
    ExtEnabled vk_ext_image_view_min_lod{kNotEnabled};
    ExtEnabled vk_ext_multi_draw{kNotEnabled};
    ExtEnabled vk_ext_image_2d_view_of_3d{kNotEnabled};
    ExtEnabled vk_ext_shader_tile_image{kNotEnabled};
    ExtEnabled vk_ext_opacity_micromap{kNotEnabled};
    ExtEnabled vk_nv_displacement_micromap{kNotEnabled};
    ExtEnabled vk_ext_load_store_op_none{kNotEnabled};
    ExtEnabled vk_huawei_cluster_culling_shader{kNotEnabled};
    ExtEnabled vk_ext_border_color_swizzle{kNotEnabled};
    ExtEnabled vk_ext_pageable_device_local_memory{kNotEnabled};
    ExtEnabled vk_arm_shader_core_properties{kNotEnabled};
    ExtEnabled vk_arm_scheduling_controls{kNotEnabled};
    ExtEnabled vk_ext_image_sliced_view_of_3d{kNotEnabled};
    ExtEnabled vk_valve_descriptor_set_host_mapping{kNotEnabled};
    ExtEnabled vk_ext_depth_clamp_zero_one{kNotEnabled};
    ExtEnabled vk_ext_non_seamless_cube_map{kNotEnabled};
    ExtEnabled vk_arm_render_pass_striped{kNotEnabled};
    ExtEnabled vk_qcom_fragment_density_map_offset{kNotEnabled};
    ExtEnabled vk_nv_copy_memory_indirect{kNotEnabled};
    ExtEnabled vk_nv_memory_decompression{kNotEnabled};
    ExtEnabled vk_nv_device_generated_commands_compute{kNotEnabled};
    ExtEnabled vk_nv_linear_color_attachment{kNotEnabled};
    ExtEnabled vk_ext_image_compression_control_swapchain{kNotEnabled};
    ExtEnabled vk_qcom_image_processing{kNotEnabled};
    ExtEnabled vk_ext_nested_command_buffer{kNotEnabled};
    ExtEnabled vk_ext_external_memory_acquire_unmodified{kNotEnabled};
    ExtEnabled vk_ext_extended_dynamic_state3{kNotEnabled};
    ExtEnabled vk_ext_subpass_merge_feedback{kNotEnabled};
    ExtEnabled vk_ext_shader_module_identifier{kNotEnabled};
    ExtEnabled vk_ext_rasterization_order_attachment_access{kNotEnabled};
    ExtEnabled vk_nv_optical_flow{kNotEnabled};
    ExtEnabled vk_ext_legacy_dithering{kNotEnabled};
    ExtEnabled vk_ext_pipeline_protected_access{kNotEnabled};
    ExtEnabled vk_android_external_format_resolve{kNotEnabled};
    ExtEnabled vk_ext_shader_object{kNotEnabled};
    ExtEnabled vk_qcom_tile_properties{kNotEnabled};
    ExtEnabled vk_sec_amigo_profiling{kNotEnabled};
    ExtEnabled vk_qcom_multiview_per_view_viewports{kNotEnabled};
    ExtEnabled vk_nv_ray_tracing_invocation_reorder{kNotEnabled};
    ExtEnabled vk_nv_extended_sparse_address_space{kNotEnabled};
    ExtEnabled vk_ext_mutable_descriptor_type{kNotEnabled};
    ExtEnabled vk_arm_shader_core_builtins{kNotEnabled};
    ExtEnabled vk_ext_pipeline_library_group_handles{kNotEnabled};
    ExtEnabled vk_ext_dynamic_rendering_unused_attachments{kNotEnabled};
    ExtEnabled vk_nv_low_latency2{kNotEnabled};
    ExtEnabled vk_qcom_multiview_per_view_render_areas{kNotEnabled};
    ExtEnabled vk_nv_per_stage_descriptor_set{kNotEnabled};
    ExtEnabled vk_qcom_image_processing2{kNotEnabled};
    ExtEnabled vk_qcom_filter_cubic_weights{kNotEnabled};
    ExtEnabled vk_qcom_ycbcr_degamma{kNotEnabled};
    ExtEnabled vk_qcom_filter_cubic_clamp{kNotEnabled};
    ExtEnabled vk_ext_attachment_feedback_loop_dynamic_state{kNotEnabled};
    ExtEnabled vk_qnx_external_memory_screen_buffer{kNotEnabled};
    ExtEnabled vk_msft_layered_driver{kNotEnabled};
    ExtEnabled vk_nv_descriptor_pool_overallocation{kNotEnabled};
    ExtEnabled vk_khr_acceleration_structure{kNotEnabled};
    ExtEnabled vk_khr_ray_tracing_pipeline{kNotEnabled};
    ExtEnabled vk_khr_ray_query{kNotEnabled};
    ExtEnabled vk_ext_mesh_shader{kNotEnabled};

    static const PromotedExtensionInfoMap &get_promotion_info_map() {
        static const PromotedExtensionInfoMap promoted_map = {
            {VK_API_VERSION_1_1,
             {"VK_VERSION_1_1",
              {
                  vvl::Extension::KHR_multiview,
                  vvl::Extension::KHR_device_group,
                  vvl::Extension::KHR_shader_draw_parameters,
                  vvl::Extension::KHR_maintenance1,
                  vvl::Extension::KHR_external_memory,
                  vvl::Extension::KHR_external_semaphore,
                  vvl::Extension::KHR_16bit_storage,
                  vvl::Extension::KHR_descriptor_update_template,
                  vvl::Extension::KHR_external_fence,
                  vvl::Extension::KHR_maintenance2,
                  vvl::Extension::KHR_variable_pointers,
                  vvl::Extension::KHR_dedicated_allocation,
                  vvl::Extension::KHR_storage_buffer_storage_class,
                  vvl::Extension::KHR_relaxed_block_layout,
                  vvl::Extension::KHR_get_memory_requirements2,
                  vvl::Extension::KHR_sampler_ycbcr_conversion,
                  vvl::Extension::KHR_bind_memory2,
                  vvl::Extension::KHR_maintenance3,
              }}},
            {VK_API_VERSION_1_2,
             {"VK_VERSION_1_2",
              {
                  vvl::Extension::KHR_sampler_mirror_clamp_to_edge,
                  vvl::Extension::KHR_shader_float16_int8,
                  vvl::Extension::KHR_imageless_framebuffer,
                  vvl::Extension::KHR_create_renderpass2,
                  vvl::Extension::KHR_image_format_list,
                  vvl::Extension::KHR_draw_indirect_count,
                  vvl::Extension::KHR_shader_subgroup_extended_types,
                  vvl::Extension::KHR_8bit_storage,
                  vvl::Extension::KHR_shader_atomic_int64,
                  vvl::Extension::KHR_driver_properties,
                  vvl::Extension::KHR_shader_float_controls,
                  vvl::Extension::KHR_depth_stencil_resolve,
                  vvl::Extension::KHR_timeline_semaphore,
                  vvl::Extension::KHR_vulkan_memory_model,
                  vvl::Extension::KHR_spirv_1_4,
                  vvl::Extension::KHR_separate_depth_stencil_layouts,
                  vvl::Extension::KHR_uniform_buffer_standard_layout,
                  vvl::Extension::KHR_buffer_device_address,
                  vvl::Extension::EXT_sampler_filter_minmax,
                  vvl::Extension::EXT_descriptor_indexing,
                  vvl::Extension::EXT_shader_viewport_index_layer,
                  vvl::Extension::EXT_scalar_block_layout,
                  vvl::Extension::EXT_separate_stencil_usage,
                  vvl::Extension::EXT_host_query_reset,
              }}},
            {VK_API_VERSION_1_3,
             {"VK_VERSION_1_3",
              {
                  vvl::Extension::KHR_dynamic_rendering,
                  vvl::Extension::KHR_shader_terminate_invocation,
                  vvl::Extension::KHR_shader_integer_dot_product,
                  vvl::Extension::KHR_shader_non_semantic_info,
                  vvl::Extension::KHR_synchronization2,
                  vvl::Extension::KHR_zero_initialize_workgroup_memory,
                  vvl::Extension::KHR_copy_commands2,
                  vvl::Extension::KHR_format_feature_flags2,
                  vvl::Extension::KHR_maintenance4,
                  vvl::Extension::EXT_texture_compression_astc_hdr,
                  vvl::Extension::EXT_inline_uniform_block,
                  vvl::Extension::EXT_pipeline_creation_feedback,
                  vvl::Extension::EXT_subgroup_size_control,
                  vvl::Extension::EXT_tooling_info,
                  vvl::Extension::EXT_extended_dynamic_state,
                  vvl::Extension::EXT_shader_demote_to_helper_invocation,
                  vvl::Extension::EXT_texel_buffer_alignment,
                  vvl::Extension::EXT_private_data,
                  vvl::Extension::EXT_pipeline_creation_cache_control,
                  vvl::Extension::EXT_ycbcr_2plane_444_formats,
                  vvl::Extension::EXT_image_robustness,
                  vvl::Extension::EXT_4444_formats,
                  vvl::Extension::EXT_extended_dynamic_state2,
              }}},

        };
        return promoted_map;
    }

    struct DeviceReq {
        const ExtEnabled DeviceExtensions::*enabled;
        const char *name;
    };
    typedef std::vector<DeviceReq> DeviceReqVec;
    struct DeviceInfo {
        DeviceInfo(ExtEnabled DeviceExtensions::*state_, const DeviceReqVec requirements_)
            : state(state_), requirements(requirements_) {}
        ExtEnabled DeviceExtensions::*state;
        DeviceReqVec requirements;
    };

    typedef vvl::unordered_map<vvl::Extension, DeviceInfo> DeviceInfoMap;
    static const DeviceInfoMap &get_info_map() {
        static const DeviceInfoMap info_map = {
            {vvl::Extension::KHR_swapchain, DeviceInfo(&DeviceExtensions::vk_khr_swapchain,
                                                       {{{&DeviceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_display_swapchain,
             DeviceInfo(&DeviceExtensions::vk_khr_display_swapchain,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_display, VK_KHR_DISPLAY_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_sampler_mirror_clamp_to_edge,
             DeviceInfo(&DeviceExtensions::vk_khr_sampler_mirror_clamp_to_edge, {})},
            {vvl::Extension::KHR_video_queue,
             DeviceInfo(&DeviceExtensions::vk_khr_video_queue,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_decode_queue,
             DeviceInfo(&DeviceExtensions::vk_khr_video_decode_queue,
                        {{{&DeviceExtensions::vk_khr_video_queue, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_encode_h264,
             DeviceInfo(&DeviceExtensions::vk_khr_video_encode_h264,
                        {{{&DeviceExtensions::vk_khr_video_encode_queue, VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_encode_h265,
             DeviceInfo(&DeviceExtensions::vk_khr_video_encode_h265,
                        {{{&DeviceExtensions::vk_khr_video_encode_queue, VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_decode_h264,
             DeviceInfo(&DeviceExtensions::vk_khr_video_decode_h264,
                        {{{&DeviceExtensions::vk_khr_video_decode_queue, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_dynamic_rendering,
             DeviceInfo(&DeviceExtensions::vk_khr_dynamic_rendering,
                        {{{&DeviceExtensions::vk_khr_depth_stencil_resolve, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_multiview,
             DeviceInfo(&DeviceExtensions::vk_khr_multiview, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_device_group,
             DeviceInfo(&DeviceExtensions::vk_khr_device_group,
                        {{{&DeviceExtensions::vk_khr_device_group_creation, VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_draw_parameters, DeviceInfo(&DeviceExtensions::vk_khr_shader_draw_parameters, {})},
            {vvl::Extension::KHR_maintenance1, DeviceInfo(&DeviceExtensions::vk_khr_maintenance1, {})},
            {vvl::Extension::KHR_external_memory,
             DeviceInfo(&DeviceExtensions::vk_khr_external_memory, {{{&DeviceExtensions::vk_khr_external_memory_capabilities,
                                                                      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_memory_win32,
             DeviceInfo(&DeviceExtensions::vk_khr_external_memory_win32,
                        {{{&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_memory_fd,
             DeviceInfo(&DeviceExtensions::vk_khr_external_memory_fd,
                        {{{&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_win32_keyed_mutex,
             DeviceInfo(&DeviceExtensions::vk_khr_win32_keyed_mutex,
                        {{{&DeviceExtensions::vk_khr_external_memory_win32, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_semaphore,
             DeviceInfo(&DeviceExtensions::vk_khr_external_semaphore, {{{&DeviceExtensions::vk_khr_external_semaphore_capabilities,
                                                                         VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_semaphore_win32,
             DeviceInfo(&DeviceExtensions::vk_khr_external_semaphore_win32,
                        {{{&DeviceExtensions::vk_khr_external_semaphore, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_semaphore_fd,
             DeviceInfo(&DeviceExtensions::vk_khr_external_semaphore_fd,
                        {{{&DeviceExtensions::vk_khr_external_semaphore, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_push_descriptor,
             DeviceInfo(&DeviceExtensions::vk_khr_push_descriptor, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_float16_int8, DeviceInfo(&DeviceExtensions::vk_khr_shader_float16_int8,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_16bit_storage,
             DeviceInfo(&DeviceExtensions::vk_khr_16bit_storage, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                                                                   {&DeviceExtensions::vk_khr_storage_buffer_storage_class,
                                                                    VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_incremental_present,
             DeviceInfo(&DeviceExtensions::vk_khr_incremental_present,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_descriptor_update_template, DeviceInfo(&DeviceExtensions::vk_khr_descriptor_update_template, {})},
            {vvl::Extension::KHR_imageless_framebuffer,
             DeviceInfo(&DeviceExtensions::vk_khr_imageless_framebuffer,
                        {{{&DeviceExtensions::vk_khr_maintenance2, VK_KHR_MAINTENANCE_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_image_format_list, VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_create_renderpass2,
             DeviceInfo(&DeviceExtensions::vk_khr_create_renderpass2,
                        {{{&DeviceExtensions::vk_khr_multiview, VK_KHR_MULTIVIEW_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_maintenance2, VK_KHR_MAINTENANCE_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shared_presentable_image,
             DeviceInfo(&DeviceExtensions::vk_khr_shared_presentable_image,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_surface_capabilities2, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_external_fence,
             DeviceInfo(&DeviceExtensions::vk_khr_external_fence, {{{&DeviceExtensions::vk_khr_external_fence_capabilities,
                                                                     VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_fence_win32,
             DeviceInfo(&DeviceExtensions::vk_khr_external_fence_win32,
                        {{{&DeviceExtensions::vk_khr_external_fence, VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::KHR_external_fence_fd,
             DeviceInfo(&DeviceExtensions::vk_khr_external_fence_fd,
                        {{{&DeviceExtensions::vk_khr_external_fence, VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_performance_query,
             DeviceInfo(&DeviceExtensions::vk_khr_performance_query, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_maintenance2, DeviceInfo(&DeviceExtensions::vk_khr_maintenance2, {})},
            {vvl::Extension::KHR_variable_pointers,
             DeviceInfo(&DeviceExtensions::vk_khr_variable_pointers, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                                                                       {&DeviceExtensions::vk_khr_storage_buffer_storage_class,
                                                                        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_dedicated_allocation,
             DeviceInfo(&DeviceExtensions::vk_khr_dedicated_allocation,
                        {{{&DeviceExtensions::vk_khr_get_memory_requirements2, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_storage_buffer_storage_class,
             DeviceInfo(&DeviceExtensions::vk_khr_storage_buffer_storage_class, {})},
            {vvl::Extension::KHR_relaxed_block_layout, DeviceInfo(&DeviceExtensions::vk_khr_relaxed_block_layout, {})},
            {vvl::Extension::KHR_get_memory_requirements2, DeviceInfo(&DeviceExtensions::vk_khr_get_memory_requirements2, {})},
            {vvl::Extension::KHR_image_format_list, DeviceInfo(&DeviceExtensions::vk_khr_image_format_list, {})},
            {vvl::Extension::KHR_sampler_ycbcr_conversion,
             DeviceInfo(&DeviceExtensions::vk_khr_sampler_ycbcr_conversion,
                        {{{&DeviceExtensions::vk_khr_maintenance1, VK_KHR_MAINTENANCE_1_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_bind_memory2, VK_KHR_BIND_MEMORY_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_memory_requirements2, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_bind_memory2, DeviceInfo(&DeviceExtensions::vk_khr_bind_memory2, {})},
#ifdef VK_ENABLE_BETA_EXTENSIONS
            {vvl::Extension::KHR_portability_subset, DeviceInfo(&DeviceExtensions::vk_khr_portability_subset,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
#endif  // VK_ENABLE_BETA_EXTENSIONS
            {vvl::Extension::KHR_maintenance3,
             DeviceInfo(&DeviceExtensions::vk_khr_maintenance3, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_draw_indirect_count, DeviceInfo(&DeviceExtensions::vk_khr_draw_indirect_count, {})},
            {vvl::Extension::KHR_shader_subgroup_extended_types,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_subgroup_extended_types,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::KHR_8bit_storage,
             DeviceInfo(&DeviceExtensions::vk_khr_8bit_storage, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                                                                  {&DeviceExtensions::vk_khr_storage_buffer_storage_class,
                                                                   VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_atomic_int64, DeviceInfo(&DeviceExtensions::vk_khr_shader_atomic_int64,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_clock,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_clock, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_decode_h265,
             DeviceInfo(&DeviceExtensions::vk_khr_video_decode_h265,
                        {{{&DeviceExtensions::vk_khr_video_decode_queue, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_global_priority,
             DeviceInfo(&DeviceExtensions::vk_khr_global_priority, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_driver_properties,
             DeviceInfo(&DeviceExtensions::vk_khr_driver_properties, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_float_controls, DeviceInfo(&DeviceExtensions::vk_khr_shader_float_controls,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_depth_stencil_resolve,
             DeviceInfo(&DeviceExtensions::vk_khr_depth_stencil_resolve,
                        {{{&DeviceExtensions::vk_khr_create_renderpass2, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_swapchain_mutable_format,
             DeviceInfo(&DeviceExtensions::vk_khr_swapchain_mutable_format,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_maintenance2, VK_KHR_MAINTENANCE_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_image_format_list, VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_timeline_semaphore, DeviceInfo(&DeviceExtensions::vk_khr_timeline_semaphore,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_vulkan_memory_model, DeviceInfo(&DeviceExtensions::vk_khr_vulkan_memory_model,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_terminate_invocation,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_terminate_invocation,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_fragment_shading_rate,
             DeviceInfo(&DeviceExtensions::vk_khr_fragment_shading_rate,
                        {{{&DeviceExtensions::vk_khr_create_renderpass2, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_dynamic_rendering_local_read,
             DeviceInfo(&DeviceExtensions::vk_khr_dynamic_rendering_local_read,
                        {{{&DeviceExtensions::vk_khr_dynamic_rendering, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_quad_control,
             DeviceInfo(
                 &DeviceExtensions::vk_khr_shader_quad_control,
                 {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                   {&DeviceExtensions::vk_khr_vulkan_memory_model, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME},
                   {&DeviceExtensions::vk_khr_shader_maximal_reconvergence, VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_spirv_1_4,
             DeviceInfo(&DeviceExtensions::vk_khr_spirv_1_4,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                          {&DeviceExtensions::vk_khr_shader_float_controls, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_separate_depth_stencil_layouts,
             DeviceInfo(&DeviceExtensions::vk_khr_separate_depth_stencil_layouts,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_create_renderpass2, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_present_wait,
             DeviceInfo(&DeviceExtensions::vk_khr_present_wait,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_present_id, VK_KHR_PRESENT_ID_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_uniform_buffer_standard_layout,
             DeviceInfo(&DeviceExtensions::vk_khr_uniform_buffer_standard_layout,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_buffer_device_address,
             DeviceInfo(&DeviceExtensions::vk_khr_buffer_device_address,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_device_group, VK_KHR_DEVICE_GROUP_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_deferred_host_operations, DeviceInfo(&DeviceExtensions::vk_khr_deferred_host_operations, {})},
            {vvl::Extension::KHR_pipeline_executable_properties,
             DeviceInfo(&DeviceExtensions::vk_khr_pipeline_executable_properties,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_map_memory2, DeviceInfo(&DeviceExtensions::vk_khr_map_memory2, {})},
            {vvl::Extension::KHR_shader_integer_dot_product,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_integer_dot_product,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_pipeline_library, DeviceInfo(&DeviceExtensions::vk_khr_pipeline_library, {})},
            {vvl::Extension::KHR_shader_non_semantic_info, DeviceInfo(&DeviceExtensions::vk_khr_shader_non_semantic_info, {})},
            {vvl::Extension::KHR_present_id, DeviceInfo(&DeviceExtensions::vk_khr_present_id,
                                                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                                                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_encode_queue,
             DeviceInfo(&DeviceExtensions::vk_khr_video_encode_queue,
                        {{{&DeviceExtensions::vk_khr_video_queue, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_synchronization2,
             DeviceInfo(&DeviceExtensions::vk_khr_synchronization2, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_fragment_shader_barycentric,
             DeviceInfo(&DeviceExtensions::vk_khr_fragment_shader_barycentric,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_subgroup_uniform_control_flow,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_subgroup_uniform_control_flow,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::KHR_zero_initialize_workgroup_memory,
             DeviceInfo(&DeviceExtensions::vk_khr_zero_initialize_workgroup_memory,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_workgroup_memory_explicit_layout,
             DeviceInfo(&DeviceExtensions::vk_khr_workgroup_memory_explicit_layout,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_copy_commands2,
             DeviceInfo(&DeviceExtensions::vk_khr_copy_commands2, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_format_feature_flags2, DeviceInfo(&DeviceExtensions::vk_khr_format_feature_flags2,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_ray_tracing_maintenance1,
             DeviceInfo(&DeviceExtensions::vk_khr_ray_tracing_maintenance1,
                        {{{&DeviceExtensions::vk_khr_acceleration_structure, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_maintenance4,
             DeviceInfo(&DeviceExtensions::vk_khr_maintenance4, {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::KHR_shader_subgroup_rotate, DeviceInfo(&DeviceExtensions::vk_khr_shader_subgroup_rotate, {})},
            {vvl::Extension::KHR_shader_maximal_reconvergence,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_maximal_reconvergence,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::KHR_maintenance5,
             DeviceInfo(&DeviceExtensions::vk_khr_maintenance5,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                          {&DeviceExtensions::vk_khr_dynamic_rendering, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_ray_tracing_position_fetch,
             DeviceInfo(&DeviceExtensions::vk_khr_ray_tracing_position_fetch,
                        {{{&DeviceExtensions::vk_khr_acceleration_structure, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_cooperative_matrix, DeviceInfo(&DeviceExtensions::vk_khr_cooperative_matrix,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_video_maintenance1,
             DeviceInfo(&DeviceExtensions::vk_khr_video_maintenance1,
                        {{{&DeviceExtensions::vk_khr_video_queue, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_vertex_attribute_divisor,
             DeviceInfo(&DeviceExtensions::vk_khr_vertex_attribute_divisor,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_load_store_op_none, DeviceInfo(&DeviceExtensions::vk_khr_load_store_op_none, {})},
            {vvl::Extension::KHR_shader_float_controls2,
             DeviceInfo(&DeviceExtensions::vk_khr_shader_float_controls2,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                          {&DeviceExtensions::vk_khr_shader_float_controls, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_index_type_uint8,
             DeviceInfo(&DeviceExtensions::vk_khr_index_type_uint8, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_line_rasterization, DeviceInfo(&DeviceExtensions::vk_khr_line_rasterization,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_calibrated_timestamps, DeviceInfo(&DeviceExtensions::vk_khr_calibrated_timestamps,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_shader_expect_assume, DeviceInfo(&DeviceExtensions::vk_khr_shader_expect_assume,
                                                                  {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_maintenance6,
             DeviceInfo(&DeviceExtensions::vk_khr_maintenance6, {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::NV_glsl_shader, DeviceInfo(&DeviceExtensions::vk_nv_glsl_shader, {})},
            {vvl::Extension::EXT_depth_range_unrestricted, DeviceInfo(&DeviceExtensions::vk_ext_depth_range_unrestricted, {})},
            {vvl::Extension::IMG_filter_cubic, DeviceInfo(&DeviceExtensions::vk_img_filter_cubic, {})},
            {vvl::Extension::AMD_rasterization_order, DeviceInfo(&DeviceExtensions::vk_amd_rasterization_order, {})},
            {vvl::Extension::AMD_shader_trinary_minmax, DeviceInfo(&DeviceExtensions::vk_amd_shader_trinary_minmax, {})},
            {vvl::Extension::AMD_shader_explicit_vertex_parameter,
             DeviceInfo(&DeviceExtensions::vk_amd_shader_explicit_vertex_parameter, {})},
            {vvl::Extension::EXT_debug_marker,
             DeviceInfo(&DeviceExtensions::vk_ext_debug_marker,
                        {{{&DeviceExtensions::vk_ext_debug_report, VK_EXT_DEBUG_REPORT_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_gcn_shader, DeviceInfo(&DeviceExtensions::vk_amd_gcn_shader, {})},
            {vvl::Extension::NV_dedicated_allocation, DeviceInfo(&DeviceExtensions::vk_nv_dedicated_allocation, {})},
            {vvl::Extension::EXT_transform_feedback, DeviceInfo(&DeviceExtensions::vk_ext_transform_feedback,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NVX_binary_import, DeviceInfo(&DeviceExtensions::vk_nvx_binary_import, {})},
            {vvl::Extension::NVX_image_view_handle, DeviceInfo(&DeviceExtensions::vk_nvx_image_view_handle, {})},
            {vvl::Extension::AMD_draw_indirect_count, DeviceInfo(&DeviceExtensions::vk_amd_draw_indirect_count, {})},
            {vvl::Extension::AMD_negative_viewport_height, DeviceInfo(&DeviceExtensions::vk_amd_negative_viewport_height, {})},
            {vvl::Extension::AMD_gpu_shader_half_float, DeviceInfo(&DeviceExtensions::vk_amd_gpu_shader_half_float, {})},
            {vvl::Extension::AMD_shader_ballot, DeviceInfo(&DeviceExtensions::vk_amd_shader_ballot, {})},
            {vvl::Extension::AMD_texture_gather_bias_lod, DeviceInfo(&DeviceExtensions::vk_amd_texture_gather_bias_lod,
                                                                     {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_shader_info, DeviceInfo(&DeviceExtensions::vk_amd_shader_info, {})},
            {vvl::Extension::AMD_shader_image_load_store_lod,
             DeviceInfo(&DeviceExtensions::vk_amd_shader_image_load_store_lod, {})},
            {vvl::Extension::NV_corner_sampled_image, DeviceInfo(&DeviceExtensions::vk_nv_corner_sampled_image,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::IMG_format_pvrtc, DeviceInfo(&DeviceExtensions::vk_img_format_pvrtc, {})},
            {vvl::Extension::NV_external_memory,
             DeviceInfo(&DeviceExtensions::vk_nv_external_memory, {{{&DeviceExtensions::vk_nv_external_memory_capabilities,
                                                                     VK_NV_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::NV_external_memory_win32,
             DeviceInfo(&DeviceExtensions::vk_nv_external_memory_win32,
                        {{{&DeviceExtensions::vk_nv_external_memory, VK_NV_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
            {vvl::Extension::NV_win32_keyed_mutex,
             DeviceInfo(&DeviceExtensions::vk_nv_win32_keyed_mutex,
                        {{{&DeviceExtensions::vk_nv_external_memory_win32, VK_NV_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::EXT_shader_subgroup_ballot, DeviceInfo(&DeviceExtensions::vk_ext_shader_subgroup_ballot, {})},
            {vvl::Extension::EXT_shader_subgroup_vote, DeviceInfo(&DeviceExtensions::vk_ext_shader_subgroup_vote, {})},
            {vvl::Extension::EXT_texture_compression_astc_hdr,
             DeviceInfo(&DeviceExtensions::vk_ext_texture_compression_astc_hdr,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_astc_decode_mode,
             DeviceInfo(&DeviceExtensions::vk_ext_astc_decode_mode, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_pipeline_robustness, DeviceInfo(&DeviceExtensions::vk_ext_pipeline_robustness,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_conditional_rendering, DeviceInfo(&DeviceExtensions::vk_ext_conditional_rendering,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_clip_space_w_scaling, DeviceInfo(&DeviceExtensions::vk_nv_clip_space_w_scaling, {})},
            {vvl::Extension::EXT_display_control,
             DeviceInfo(&DeviceExtensions::vk_ext_display_control,
                        {{{&DeviceExtensions::vk_ext_display_surface_counter, VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
            {vvl::Extension::GOOGLE_display_timing,
             DeviceInfo(&DeviceExtensions::vk_google_display_timing,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
            {vvl::Extension::NV_sample_mask_override_coverage,
             DeviceInfo(&DeviceExtensions::vk_nv_sample_mask_override_coverage, {})},
            {vvl::Extension::NV_geometry_shader_passthrough, DeviceInfo(&DeviceExtensions::vk_nv_geometry_shader_passthrough, {})},
            {vvl::Extension::NV_viewport_array2, DeviceInfo(&DeviceExtensions::vk_nv_viewport_array2, {})},
            {vvl::Extension::NVX_multiview_per_view_attributes,
             DeviceInfo(&DeviceExtensions::vk_nvx_multiview_per_view_attributes,
                        {{{&DeviceExtensions::vk_khr_multiview, VK_KHR_MULTIVIEW_EXTENSION_NAME}}})},
            {vvl::Extension::NV_viewport_swizzle, DeviceInfo(&DeviceExtensions::vk_nv_viewport_swizzle, {})},
            {vvl::Extension::EXT_discard_rectangles, DeviceInfo(&DeviceExtensions::vk_ext_discard_rectangles,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_conservative_rasterization,
             DeviceInfo(&DeviceExtensions::vk_ext_conservative_rasterization,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_depth_clip_enable,
             DeviceInfo(&DeviceExtensions::vk_ext_depth_clip_enable, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_hdr_metadata,
             DeviceInfo(&DeviceExtensions::vk_ext_hdr_metadata,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
            {vvl::Extension::IMG_relaxed_line_rasterization,
             DeviceInfo(&DeviceExtensions::vk_img_relaxed_line_rasterization,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_external_memory_dma_buf,
             DeviceInfo(&DeviceExtensions::vk_ext_external_memory_dma_buf,
                        {{{&DeviceExtensions::vk_khr_external_memory_fd, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_queue_family_foreign,
             DeviceInfo(&DeviceExtensions::vk_ext_queue_family_foreign,
                        {{{&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_ANDROID_KHR
            {vvl::Extension::ANDROID_external_memory_android_hardware_buffer,
             DeviceInfo(&DeviceExtensions::vk_android_external_memory_android_hardware_buffer,
                        {{{&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_queue_family_foreign, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_dedicated_allocation, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_ANDROID_KHR
            {vvl::Extension::EXT_sampler_filter_minmax, DeviceInfo(&DeviceExtensions::vk_ext_sampler_filter_minmax,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_gpu_shader_int16, DeviceInfo(&DeviceExtensions::vk_amd_gpu_shader_int16, {})},
#ifdef VK_ENABLE_BETA_EXTENSIONS
            {vvl::Extension::AMDX_shader_enqueue,
             DeviceInfo(&DeviceExtensions::vk_amdx_shader_enqueue,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_pipeline_library, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_spirv_1_4, VK_KHR_SPIRV_1_4_EXTENSION_NAME}}})},
#endif  // VK_ENABLE_BETA_EXTENSIONS
            {vvl::Extension::AMD_mixed_attachment_samples, DeviceInfo(&DeviceExtensions::vk_amd_mixed_attachment_samples, {})},
            {vvl::Extension::AMD_shader_fragment_mask, DeviceInfo(&DeviceExtensions::vk_amd_shader_fragment_mask, {})},
            {vvl::Extension::EXT_inline_uniform_block,
             DeviceInfo(&DeviceExtensions::vk_ext_inline_uniform_block,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_maintenance1, VK_KHR_MAINTENANCE_1_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_stencil_export, DeviceInfo(&DeviceExtensions::vk_ext_shader_stencil_export, {})},
            {vvl::Extension::EXT_sample_locations,
             DeviceInfo(&DeviceExtensions::vk_ext_sample_locations, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_blend_operation_advanced,
             DeviceInfo(&DeviceExtensions::vk_ext_blend_operation_advanced,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_fragment_coverage_to_color, DeviceInfo(&DeviceExtensions::vk_nv_fragment_coverage_to_color, {})},
            {vvl::Extension::NV_framebuffer_mixed_samples, DeviceInfo(&DeviceExtensions::vk_nv_framebuffer_mixed_samples, {})},
            {vvl::Extension::NV_fill_rectangle, DeviceInfo(&DeviceExtensions::vk_nv_fill_rectangle, {})},
            {vvl::Extension::NV_shader_sm_builtins, DeviceInfo(&DeviceExtensions::vk_nv_shader_sm_builtins,
                                                               {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::EXT_post_depth_coverage, DeviceInfo(&DeviceExtensions::vk_ext_post_depth_coverage, {})},
            {vvl::Extension::EXT_image_drm_format_modifier,
             DeviceInfo(&DeviceExtensions::vk_ext_image_drm_format_modifier,
                        {{{&DeviceExtensions::vk_khr_bind_memory2, VK_KHR_BIND_MEMORY_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_image_format_list, VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_validation_cache, DeviceInfo(&DeviceExtensions::vk_ext_validation_cache, {})},
            {vvl::Extension::EXT_descriptor_indexing,
             DeviceInfo(&DeviceExtensions::vk_ext_descriptor_indexing,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_maintenance3, VK_KHR_MAINTENANCE_3_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_viewport_index_layer,
             DeviceInfo(&DeviceExtensions::vk_ext_shader_viewport_index_layer, {})},
            {vvl::Extension::NV_shading_rate_image,
             DeviceInfo(&DeviceExtensions::vk_nv_shading_rate_image, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_ray_tracing,
             DeviceInfo(&DeviceExtensions::vk_nv_ray_tracing,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_memory_requirements2, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_representative_fragment_test,
             DeviceInfo(&DeviceExtensions::vk_nv_representative_fragment_test,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_filter_cubic, DeviceInfo(&DeviceExtensions::vk_ext_filter_cubic, {})},
            {vvl::Extension::QCOM_render_pass_shader_resolve,
             DeviceInfo(&DeviceExtensions::vk_qcom_render_pass_shader_resolve, {})},
            {vvl::Extension::EXT_global_priority, DeviceInfo(&DeviceExtensions::vk_ext_global_priority, {})},
            {vvl::Extension::EXT_external_memory_host,
             DeviceInfo(&DeviceExtensions::vk_ext_external_memory_host,
                        {{{&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_buffer_marker, DeviceInfo(&DeviceExtensions::vk_amd_buffer_marker, {})},
            {vvl::Extension::AMD_pipeline_compiler_control, DeviceInfo(&DeviceExtensions::vk_amd_pipeline_compiler_control, {})},
            {vvl::Extension::EXT_calibrated_timestamps, DeviceInfo(&DeviceExtensions::vk_ext_calibrated_timestamps,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_shader_core_properties, DeviceInfo(&DeviceExtensions::vk_amd_shader_core_properties,
                                                                    {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_memory_overallocation_behavior,
             DeviceInfo(&DeviceExtensions::vk_amd_memory_overallocation_behavior, {})},
            {vvl::Extension::EXT_vertex_attribute_divisor,
             DeviceInfo(&DeviceExtensions::vk_ext_vertex_attribute_divisor,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_GGP
            {vvl::Extension::GGP_frame_token, DeviceInfo(&DeviceExtensions::vk_ggp_frame_token,
                                                         {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                                                           {&DeviceExtensions::vk_ggp_stream_descriptor_surface,
                                                            VK_GGP_STREAM_DESCRIPTOR_SURFACE_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_GGP
            {vvl::Extension::EXT_pipeline_creation_feedback, DeviceInfo(&DeviceExtensions::vk_ext_pipeline_creation_feedback, {})},
            {vvl::Extension::NV_shader_subgroup_partitioned,
             DeviceInfo(&DeviceExtensions::vk_nv_shader_subgroup_partitioned,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::NV_compute_shader_derivatives,
             DeviceInfo(&DeviceExtensions::vk_nv_compute_shader_derivatives,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_mesh_shader,
             DeviceInfo(&DeviceExtensions::vk_nv_mesh_shader, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                 VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_fragment_shader_barycentric,
             DeviceInfo(&DeviceExtensions::vk_nv_fragment_shader_barycentric,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_shader_image_footprint, DeviceInfo(&DeviceExtensions::vk_nv_shader_image_footprint,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_scissor_exclusive,
             DeviceInfo(&DeviceExtensions::vk_nv_scissor_exclusive, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_device_diagnostic_checkpoints,
             DeviceInfo(&DeviceExtensions::vk_nv_device_diagnostic_checkpoints,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::INTEL_shader_integer_functions2,
             DeviceInfo(&DeviceExtensions::vk_intel_shader_integer_functions2,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::INTEL_performance_query, DeviceInfo(&DeviceExtensions::vk_intel_performance_query, {})},
            {vvl::Extension::EXT_pci_bus_info,
             DeviceInfo(&DeviceExtensions::vk_ext_pci_bus_info, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_display_native_hdr,
             DeviceInfo(&DeviceExtensions::vk_amd_display_native_hdr,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_surface_capabilities2, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_fragment_density_map, DeviceInfo(&DeviceExtensions::vk_ext_fragment_density_map,
                                                                  {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_scalar_block_layout, DeviceInfo(&DeviceExtensions::vk_ext_scalar_block_layout,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::GOOGLE_hlsl_functionality1, DeviceInfo(&DeviceExtensions::vk_google_hlsl_functionality1, {})},
            {vvl::Extension::GOOGLE_decorate_string, DeviceInfo(&DeviceExtensions::vk_google_decorate_string, {})},
            {vvl::Extension::EXT_subgroup_size_control,
             DeviceInfo(&DeviceExtensions::vk_ext_subgroup_size_control,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::AMD_shader_core_properties2,
             DeviceInfo(&DeviceExtensions::vk_amd_shader_core_properties2,
                        {{{&DeviceExtensions::vk_amd_shader_core_properties, VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_device_coherent_memory, DeviceInfo(&DeviceExtensions::vk_amd_device_coherent_memory,
                                                                    {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_image_atomic_int64,
             DeviceInfo(&DeviceExtensions::vk_ext_shader_image_atomic_int64,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_memory_budget,
             DeviceInfo(&DeviceExtensions::vk_ext_memory_budget, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_memory_priority,
             DeviceInfo(&DeviceExtensions::vk_ext_memory_priority, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_dedicated_allocation_image_aliasing,
             DeviceInfo(&DeviceExtensions::vk_nv_dedicated_allocation_image_aliasing,
                        {{{&DeviceExtensions::vk_khr_dedicated_allocation, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_buffer_device_address, DeviceInfo(&DeviceExtensions::vk_ext_buffer_device_address,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_tooling_info, DeviceInfo(&DeviceExtensions::vk_ext_tooling_info, {})},
            {vvl::Extension::EXT_separate_stencil_usage, DeviceInfo(&DeviceExtensions::vk_ext_separate_stencil_usage, {})},
            {vvl::Extension::NV_cooperative_matrix,
             DeviceInfo(&DeviceExtensions::vk_nv_cooperative_matrix, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_coverage_reduction_mode,
             DeviceInfo(&DeviceExtensions::vk_nv_coverage_reduction_mode,
                        {{{&DeviceExtensions::vk_nv_framebuffer_mixed_samples, VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_fragment_shader_interlock,
             DeviceInfo(&DeviceExtensions::vk_ext_fragment_shader_interlock,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_ycbcr_image_arrays,
             DeviceInfo(&DeviceExtensions::vk_ext_ycbcr_image_arrays,
                        {{{&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_provoking_vertex,
             DeviceInfo(&DeviceExtensions::vk_ext_provoking_vertex, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::EXT_full_screen_exclusive,
             DeviceInfo(&DeviceExtensions::vk_ext_full_screen_exclusive,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_surface_capabilities2, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::EXT_line_rasterization, DeviceInfo(&DeviceExtensions::vk_ext_line_rasterization,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_atomic_float, DeviceInfo(&DeviceExtensions::vk_ext_shader_atomic_float,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_host_query_reset,
             DeviceInfo(&DeviceExtensions::vk_ext_host_query_reset, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_index_type_uint8,
             DeviceInfo(&DeviceExtensions::vk_ext_index_type_uint8, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_extended_dynamic_state, DeviceInfo(&DeviceExtensions::vk_ext_extended_dynamic_state,
                                                                    {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_host_image_copy,
             DeviceInfo(&DeviceExtensions::vk_ext_host_image_copy,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_copy_commands2, VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_format_feature_flags2, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_atomic_float2,
             DeviceInfo(&DeviceExtensions::vk_ext_shader_atomic_float2,
                        {{{&DeviceExtensions::vk_ext_shader_atomic_float, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_swapchain_maintenance1,
             DeviceInfo(&DeviceExtensions::vk_ext_swapchain_maintenance1,
                        {{{&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_surface_maintenance1, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_demote_to_helper_invocation,
             DeviceInfo(&DeviceExtensions::vk_ext_shader_demote_to_helper_invocation,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_device_generated_commands,
             DeviceInfo(&DeviceExtensions::vk_nv_device_generated_commands,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                          {&DeviceExtensions::vk_khr_buffer_device_address, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME}}})},
            {vvl::Extension::NV_inherited_viewport_scissor,
             DeviceInfo(&DeviceExtensions::vk_nv_inherited_viewport_scissor,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_texel_buffer_alignment, DeviceInfo(&DeviceExtensions::vk_ext_texel_buffer_alignment,
                                                                    {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_render_pass_transform, DeviceInfo(&DeviceExtensions::vk_qcom_render_pass_transform, {})},
            {vvl::Extension::EXT_depth_bias_control, DeviceInfo(&DeviceExtensions::vk_ext_depth_bias_control,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_device_memory_report, DeviceInfo(&DeviceExtensions::vk_ext_device_memory_report,
                                                                  {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_robustness2,
             DeviceInfo(&DeviceExtensions::vk_ext_robustness2, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_custom_border_color, DeviceInfo(&DeviceExtensions::vk_ext_custom_border_color,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::GOOGLE_user_type, DeviceInfo(&DeviceExtensions::vk_google_user_type, {})},
            {vvl::Extension::NV_present_barrier,
             DeviceInfo(&DeviceExtensions::vk_nv_present_barrier,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_surface, VK_KHR_SURFACE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_surface_capabilities2, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_private_data,
             DeviceInfo(&DeviceExtensions::vk_ext_private_data, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_pipeline_creation_cache_control,
             DeviceInfo(&DeviceExtensions::vk_ext_pipeline_creation_cache_control,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_device_diagnostics_config,
             DeviceInfo(&DeviceExtensions::vk_nv_device_diagnostics_config,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_render_pass_store_ops, DeviceInfo(&DeviceExtensions::vk_qcom_render_pass_store_ops, {})},
            {vvl::Extension::NV_cuda_kernel_launch, DeviceInfo(&DeviceExtensions::vk_nv_cuda_kernel_launch, {})},
            {vvl::Extension::NV_low_latency, DeviceInfo(&DeviceExtensions::vk_nv_low_latency, {})},
#ifdef VK_USE_PLATFORM_METAL_EXT
            {vvl::Extension::EXT_metal_objects, DeviceInfo(&DeviceExtensions::vk_ext_metal_objects, {})},
#endif  // VK_USE_PLATFORM_METAL_EXT
            {vvl::Extension::EXT_descriptor_buffer,
             DeviceInfo(&DeviceExtensions::vk_ext_descriptor_buffer,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_buffer_device_address, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_descriptor_indexing, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_graphics_pipeline_library,
             DeviceInfo(&DeviceExtensions::vk_ext_graphics_pipeline_library,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_pipeline_library, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME}}})},
            {vvl::Extension::AMD_shader_early_and_late_fragment_tests,
             DeviceInfo(&DeviceExtensions::vk_amd_shader_early_and_late_fragment_tests,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_fragment_shading_rate_enums,
             DeviceInfo(&DeviceExtensions::vk_nv_fragment_shading_rate_enums,
                        {{{&DeviceExtensions::vk_khr_fragment_shading_rate, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME}}})},
            {vvl::Extension::NV_ray_tracing_motion_blur,
             DeviceInfo(&DeviceExtensions::vk_nv_ray_tracing_motion_blur,
                        {{{&DeviceExtensions::vk_khr_ray_tracing_pipeline, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_ycbcr_2plane_444_formats,
             DeviceInfo(&DeviceExtensions::vk_ext_ycbcr_2plane_444_formats,
                        {{{&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_fragment_density_map2,
             DeviceInfo(&DeviceExtensions::vk_ext_fragment_density_map2,
                        {{{&DeviceExtensions::vk_ext_fragment_density_map, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_rotated_copy_commands,
             DeviceInfo(&DeviceExtensions::vk_qcom_rotated_copy_commands,
                        {{{&DeviceExtensions::vk_khr_copy_commands2, VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_image_robustness,
             DeviceInfo(&DeviceExtensions::vk_ext_image_robustness, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_image_compression_control,
             DeviceInfo(&DeviceExtensions::vk_ext_image_compression_control,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_attachment_feedback_loop_layout,
             DeviceInfo(&DeviceExtensions::vk_ext_attachment_feedback_loop_layout,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_4444_formats,
             DeviceInfo(&DeviceExtensions::vk_ext_4444_formats, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_device_fault,
             DeviceInfo(&DeviceExtensions::vk_ext_device_fault, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::ARM_rasterization_order_attachment_access,
             DeviceInfo(&DeviceExtensions::vk_arm_rasterization_order_attachment_access,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_rgba10x6_formats,
             DeviceInfo(&DeviceExtensions::vk_ext_rgba10x6_formats,
                        {{{&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::NV_acquire_winrt_display,
             DeviceInfo(&DeviceExtensions::vk_nv_acquire_winrt_display,
                        {{{&DeviceExtensions::vk_ext_direct_mode_display, VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_WIN32_KHR
            {vvl::Extension::VALVE_mutable_descriptor_type,
             DeviceInfo(&DeviceExtensions::vk_valve_mutable_descriptor_type,
                        {{{&DeviceExtensions::vk_khr_maintenance3, VK_KHR_MAINTENANCE_3_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_vertex_input_dynamic_state,
             DeviceInfo(&DeviceExtensions::vk_ext_vertex_input_dynamic_state,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_physical_device_drm, DeviceInfo(&DeviceExtensions::vk_ext_physical_device_drm,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_device_address_binding_report,
             DeviceInfo(&DeviceExtensions::vk_ext_device_address_binding_report,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_debug_utils, VK_EXT_DEBUG_UTILS_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_depth_clip_control, DeviceInfo(&DeviceExtensions::vk_ext_depth_clip_control,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_primitive_topology_list_restart,
             DeviceInfo(&DeviceExtensions::vk_ext_primitive_topology_list_restart,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_FUCHSIA
            {vvl::Extension::FUCHSIA_external_memory,
             DeviceInfo(
                 &DeviceExtensions::vk_fuchsia_external_memory,
                 {{{&DeviceExtensions::vk_khr_external_memory_capabilities, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME},
                   {&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
            {vvl::Extension::FUCHSIA_external_semaphore,
             DeviceInfo(&DeviceExtensions::vk_fuchsia_external_semaphore,
                        {{{&DeviceExtensions::vk_khr_external_semaphore_capabilities,
                           VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_external_semaphore, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME}}})},
            {vvl::Extension::FUCHSIA_buffer_collection,
             DeviceInfo(&DeviceExtensions::vk_fuchsia_buffer_collection,
                        {{{&DeviceExtensions::vk_fuchsia_external_memory, VK_FUCHSIA_EXTERNAL_MEMORY_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_FUCHSIA
            {vvl::Extension::HUAWEI_subpass_shading,
             DeviceInfo(&DeviceExtensions::vk_huawei_subpass_shading,
                        {{{&DeviceExtensions::vk_khr_create_renderpass2, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::HUAWEI_invocation_mask,
             DeviceInfo(&DeviceExtensions::vk_huawei_invocation_mask,
                        {{{&DeviceExtensions::vk_khr_ray_tracing_pipeline, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_external_memory_rdma,
             DeviceInfo(&DeviceExtensions::vk_nv_external_memory_rdma,
                        {{{&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_pipeline_properties, DeviceInfo(&DeviceExtensions::vk_ext_pipeline_properties,
                                                                 {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_frame_boundary, DeviceInfo(&DeviceExtensions::vk_ext_frame_boundary, {})},
            {vvl::Extension::EXT_multisampled_render_to_single_sampled,
             DeviceInfo(&DeviceExtensions::vk_ext_multisampled_render_to_single_sampled,
                        {{{&DeviceExtensions::vk_khr_create_renderpass2, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_depth_stencil_resolve, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_extended_dynamic_state2, DeviceInfo(&DeviceExtensions::vk_ext_extended_dynamic_state2,
                                                                     {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_color_write_enable, DeviceInfo(&DeviceExtensions::vk_ext_color_write_enable,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_primitives_generated_query,
             DeviceInfo(&DeviceExtensions::vk_ext_primitives_generated_query,
                        {{{&DeviceExtensions::vk_ext_transform_feedback, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_global_priority_query,
             DeviceInfo(&DeviceExtensions::vk_ext_global_priority_query,
                        {{{&DeviceExtensions::vk_ext_global_priority, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_image_view_min_lod, DeviceInfo(&DeviceExtensions::vk_ext_image_view_min_lod,
                                                                {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_multi_draw,
             DeviceInfo(&DeviceExtensions::vk_ext_multi_draw, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                 VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_image_2d_view_of_3d,
             DeviceInfo(&DeviceExtensions::vk_ext_image_2d_view_of_3d,
                        {{{&DeviceExtensions::vk_khr_maintenance1, VK_KHR_MAINTENANCE_1_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_tile_image, DeviceInfo(&DeviceExtensions::vk_ext_shader_tile_image,
                                                               {{{&DeviceExtensions::vk_feature_version_1_3, "VK_VERSION_1_3"}}})},
            {vvl::Extension::EXT_opacity_micromap,
             DeviceInfo(&DeviceExtensions::vk_ext_opacity_micromap,
                        {{{&DeviceExtensions::vk_khr_acceleration_structure, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
#ifdef VK_ENABLE_BETA_EXTENSIONS
            {vvl::Extension::NV_displacement_micromap,
             DeviceInfo(&DeviceExtensions::vk_nv_displacement_micromap,
                        {{{&DeviceExtensions::vk_ext_opacity_micromap, VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME}}})},
#endif  // VK_ENABLE_BETA_EXTENSIONS
            {vvl::Extension::EXT_load_store_op_none, DeviceInfo(&DeviceExtensions::vk_ext_load_store_op_none, {})},
            {vvl::Extension::HUAWEI_cluster_culling_shader,
             DeviceInfo(&DeviceExtensions::vk_huawei_cluster_culling_shader,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_border_color_swizzle,
             DeviceInfo(&DeviceExtensions::vk_ext_border_color_swizzle,
                        {{{&DeviceExtensions::vk_ext_custom_border_color, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_pageable_device_local_memory,
             DeviceInfo(&DeviceExtensions::vk_ext_pageable_device_local_memory,
                        {{{&DeviceExtensions::vk_ext_memory_priority, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME}}})},
            {vvl::Extension::ARM_shader_core_properties,
             DeviceInfo(&DeviceExtensions::vk_arm_shader_core_properties,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::ARM_scheduling_controls,
             DeviceInfo(&DeviceExtensions::vk_arm_scheduling_controls,
                        {{{&DeviceExtensions::vk_arm_shader_core_builtins, VK_ARM_SHADER_CORE_BUILTINS_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_image_sliced_view_of_3d,
             DeviceInfo(&DeviceExtensions::vk_ext_image_sliced_view_of_3d,
                        {{{&DeviceExtensions::vk_khr_maintenance1, VK_KHR_MAINTENANCE_1_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::VALVE_descriptor_set_host_mapping,
             DeviceInfo(&DeviceExtensions::vk_valve_descriptor_set_host_mapping,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_depth_clamp_zero_one, DeviceInfo(&DeviceExtensions::vk_ext_depth_clamp_zero_one,
                                                                  {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_non_seamless_cube_map, DeviceInfo(&DeviceExtensions::vk_ext_non_seamless_cube_map,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::ARM_render_pass_striped,
             DeviceInfo(&DeviceExtensions::vk_arm_render_pass_striped,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_fragment_density_map_offset,
             DeviceInfo(&DeviceExtensions::vk_qcom_fragment_density_map_offset,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_fragment_density_map, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME}}})},
            {vvl::Extension::NV_copy_memory_indirect,
             DeviceInfo(&DeviceExtensions::vk_nv_copy_memory_indirect,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_buffer_device_address, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME}}})},
            {vvl::Extension::NV_memory_decompression,
             DeviceInfo(&DeviceExtensions::vk_nv_memory_decompression,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_buffer_device_address, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME}}})},
            {vvl::Extension::NV_device_generated_commands_compute,
             DeviceInfo(&DeviceExtensions::vk_nv_device_generated_commands_compute,
                        {{{&DeviceExtensions::vk_nv_device_generated_commands, VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME}}})},
            {vvl::Extension::NV_linear_color_attachment, DeviceInfo(&DeviceExtensions::vk_nv_linear_color_attachment,
                                                                    {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_image_compression_control_swapchain,
             DeviceInfo(
                 &DeviceExtensions::vk_ext_image_compression_control_swapchain,
                 {{{&DeviceExtensions::vk_ext_image_compression_control, VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_image_processing,
             DeviceInfo(&DeviceExtensions::vk_qcom_image_processing,
                        {{{&DeviceExtensions::vk_khr_format_feature_flags2, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_nested_command_buffer, DeviceInfo(&DeviceExtensions::vk_ext_nested_command_buffer,
                                                                   {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_external_memory_acquire_unmodified,
             DeviceInfo(&DeviceExtensions::vk_ext_external_memory_acquire_unmodified,
                        {{{&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_extended_dynamic_state3, DeviceInfo(&DeviceExtensions::vk_ext_extended_dynamic_state3,
                                                                     {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_subpass_merge_feedback, DeviceInfo(&DeviceExtensions::vk_ext_subpass_merge_feedback,
                                                                    {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_shader_module_identifier, DeviceInfo(&DeviceExtensions::vk_ext_shader_module_identifier,
                                                                      {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                         VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                                                                        {&DeviceExtensions::vk_ext_pipeline_creation_cache_control,
                                                                         VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_rasterization_order_attachment_access,
             DeviceInfo(&DeviceExtensions::vk_ext_rasterization_order_attachment_access,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_optical_flow,
             DeviceInfo(&DeviceExtensions::vk_nv_optical_flow,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_format_feature_flags2, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_synchronization2, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_legacy_dithering,
             DeviceInfo(&DeviceExtensions::vk_ext_legacy_dithering, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_pipeline_protected_access,
             DeviceInfo(&DeviceExtensions::vk_ext_pipeline_protected_access,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_ANDROID_KHR
            {vvl::Extension::ANDROID_external_format_resolve,
             DeviceInfo(&DeviceExtensions::vk_android_external_format_resolve,
                        {{{&DeviceExtensions::vk_android_external_memory_android_hardware_buffer,
                           VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_ANDROID_KHR
            {vvl::Extension::EXT_shader_object,
             DeviceInfo(&DeviceExtensions::vk_ext_shader_object,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_dynamic_rendering, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_tile_properties,
             DeviceInfo(&DeviceExtensions::vk_qcom_tile_properties, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::SEC_amigo_profiling,
             DeviceInfo(&DeviceExtensions::vk_sec_amigo_profiling, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_multiview_per_view_viewports,
             DeviceInfo(&DeviceExtensions::vk_qcom_multiview_per_view_viewports,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_ray_tracing_invocation_reorder,
             DeviceInfo(&DeviceExtensions::vk_nv_ray_tracing_invocation_reorder,
                        {{{&DeviceExtensions::vk_khr_ray_tracing_pipeline, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME}}})},
            {vvl::Extension::NV_extended_sparse_address_space,
             DeviceInfo(&DeviceExtensions::vk_nv_extended_sparse_address_space, {})},
            {vvl::Extension::EXT_mutable_descriptor_type,
             DeviceInfo(&DeviceExtensions::vk_ext_mutable_descriptor_type,
                        {{{&DeviceExtensions::vk_khr_maintenance3, VK_KHR_MAINTENANCE_3_EXTENSION_NAME}}})},
            {vvl::Extension::ARM_shader_core_builtins, DeviceInfo(&DeviceExtensions::vk_arm_shader_core_builtins,
                                                                  {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_pipeline_library_group_handles,
             DeviceInfo(&DeviceExtensions::vk_ext_pipeline_library_group_handles,
                        {{{&DeviceExtensions::vk_khr_ray_tracing_pipeline, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_pipeline_library, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_dynamic_rendering_unused_attachments,
             DeviceInfo(&DeviceExtensions::vk_ext_dynamic_rendering_unused_attachments,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_dynamic_rendering, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME}}})},
            {vvl::Extension::NV_low_latency2,
             DeviceInfo(&DeviceExtensions::vk_nv_low_latency2,
                        {{{&DeviceExtensions::vk_feature_version_1_2, "VK_VERSION_1_2"},
                          {&DeviceExtensions::vk_khr_timeline_semaphore, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_multiview_per_view_render_areas,
             DeviceInfo(&DeviceExtensions::vk_qcom_multiview_per_view_render_areas, {})},
            {vvl::Extension::NV_per_stage_descriptor_set,
             DeviceInfo(&DeviceExtensions::vk_nv_per_stage_descriptor_set,
                        {{{&DeviceExtensions::vk_khr_maintenance6, VK_KHR_MAINTENANCE_6_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_image_processing2,
             DeviceInfo(&DeviceExtensions::vk_qcom_image_processing2,
                        {{{&DeviceExtensions::vk_qcom_image_processing, VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_filter_cubic_weights,
             DeviceInfo(&DeviceExtensions::vk_qcom_filter_cubic_weights,
                        {{{&DeviceExtensions::vk_ext_filter_cubic, VK_EXT_FILTER_CUBIC_EXTENSION_NAME}}})},
            {vvl::Extension::QCOM_ycbcr_degamma, DeviceInfo(&DeviceExtensions::vk_qcom_ycbcr_degamma, {})},
            {vvl::Extension::QCOM_filter_cubic_clamp,
             DeviceInfo(&DeviceExtensions::vk_qcom_filter_cubic_clamp,
                        {{{&DeviceExtensions::vk_ext_filter_cubic, VK_EXT_FILTER_CUBIC_EXTENSION_NAME},
                          {&DeviceExtensions::vk_feature_version_1_2, "VK_VERSION_1_2"},
                          {&DeviceExtensions::vk_ext_sampler_filter_minmax, VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_attachment_feedback_loop_dynamic_state,
             DeviceInfo(&DeviceExtensions::vk_ext_attachment_feedback_loop_dynamic_state,
                        {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                           VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_attachment_feedback_loop_layout,
                           VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME}}})},
#ifdef VK_USE_PLATFORM_SCREEN_QNX
            {vvl::Extension::QNX_external_memory_screen_buffer,
             DeviceInfo(&DeviceExtensions::vk_qnx_external_memory_screen_buffer,
                        {{{&DeviceExtensions::vk_khr_sampler_ycbcr_conversion, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_external_memory, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_dedicated_allocation, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME},
                          {&DeviceExtensions::vk_ext_queue_family_foreign, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME}}})},
#endif  // VK_USE_PLATFORM_SCREEN_QNX
            {vvl::Extension::MSFT_layered_driver,
             DeviceInfo(&DeviceExtensions::vk_msft_layered_driver, {{{&DeviceExtensions::vk_khr_get_physical_device_properties2,
                                                                      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}}})},
            {vvl::Extension::NV_descriptor_pool_overallocation,
             DeviceInfo(&DeviceExtensions::vk_nv_descriptor_pool_overallocation,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"}}})},
            {vvl::Extension::KHR_acceleration_structure,
             DeviceInfo(&DeviceExtensions::vk_khr_acceleration_structure,
                        {{{&DeviceExtensions::vk_feature_version_1_1, "VK_VERSION_1_1"},
                          {&DeviceExtensions::vk_ext_descriptor_indexing, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_buffer_device_address, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_deferred_host_operations, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_ray_tracing_pipeline,
             DeviceInfo(&DeviceExtensions::vk_khr_ray_tracing_pipeline,
                        {{{&DeviceExtensions::vk_khr_spirv_1_4, VK_KHR_SPIRV_1_4_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_acceleration_structure, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME}}})},
            {vvl::Extension::KHR_ray_query,
             DeviceInfo(&DeviceExtensions::vk_khr_ray_query,
                        {{{&DeviceExtensions::vk_khr_spirv_1_4, VK_KHR_SPIRV_1_4_EXTENSION_NAME},
                          {&DeviceExtensions::vk_khr_acceleration_structure, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME}}})},
            {vvl::Extension::EXT_mesh_shader,
             DeviceInfo(&DeviceExtensions::vk_ext_mesh_shader,
                        {{{&DeviceExtensions::vk_khr_spirv_1_4, VK_KHR_SPIRV_1_4_EXTENSION_NAME}}})},

        };

        return info_map;
    }

    static const DeviceInfo &get_version_map(const char *version) {
        static const DeviceInfo empty_info{nullptr, DeviceReqVec()};
        static const vvl::unordered_map<std::string, DeviceInfo> version_map = {
            {"VK_VERSION_1_1", DeviceInfo(&DeviceExtensions::vk_feature_version_1_1, {})},
            {"VK_VERSION_1_2", DeviceInfo(&DeviceExtensions::vk_feature_version_1_2, {})},
            {"VK_VERSION_1_3", DeviceInfo(&DeviceExtensions::vk_feature_version_1_3, {})},
        };
        const auto info = version_map.find(version);
        return (info != version_map.cend()) ? info->second : empty_info;
    }

    static const DeviceInfo &get_info(vvl::Extension extension_name) {
        static const DeviceInfo empty_info{nullptr, DeviceReqVec()};
        const auto &ext_map = DeviceExtensions::get_info_map();
        const auto info = ext_map.find(extension_name);
        return (info != ext_map.cend()) ? info->second : empty_info;
    }

    DeviceExtensions() = default;
    DeviceExtensions(const InstanceExtensions &instance_ext) : InstanceExtensions(instance_ext) {}

    APIVersion InitFromDeviceCreateInfo(const InstanceExtensions *instance_extensions, APIVersion requested_api_version,
                                        const VkDeviceCreateInfo *pCreateInfo = nullptr);
};

constexpr bool IsInstanceExtension(vvl::Extension extension) {
    switch (extension) {
        case vvl::Extension::KHR_surface:
        case vvl::Extension::KHR_display:
        case vvl::Extension::KHR_xlib_surface:
        case vvl::Extension::KHR_xcb_surface:
        case vvl::Extension::KHR_wayland_surface:
        case vvl::Extension::KHR_android_surface:
        case vvl::Extension::KHR_win32_surface:
        case vvl::Extension::KHR_get_physical_device_properties2:
        case vvl::Extension::KHR_device_group_creation:
        case vvl::Extension::KHR_external_memory_capabilities:
        case vvl::Extension::KHR_external_semaphore_capabilities:
        case vvl::Extension::KHR_external_fence_capabilities:
        case vvl::Extension::KHR_get_surface_capabilities2:
        case vvl::Extension::KHR_get_display_properties2:
        case vvl::Extension::KHR_surface_protected_capabilities:
        case vvl::Extension::KHR_portability_enumeration:
        case vvl::Extension::EXT_debug_report:
        case vvl::Extension::GGP_stream_descriptor_surface:
        case vvl::Extension::NV_external_memory_capabilities:
        case vvl::Extension::EXT_validation_flags:
        case vvl::Extension::NN_vi_surface:
        case vvl::Extension::EXT_direct_mode_display:
        case vvl::Extension::EXT_acquire_xlib_display:
        case vvl::Extension::EXT_display_surface_counter:
        case vvl::Extension::EXT_swapchain_colorspace:
        case vvl::Extension::MVK_ios_surface:
        case vvl::Extension::MVK_macos_surface:
        case vvl::Extension::EXT_debug_utils:
        case vvl::Extension::FUCHSIA_imagepipe_surface:
        case vvl::Extension::EXT_metal_surface:
        case vvl::Extension::EXT_validation_features:
        case vvl::Extension::EXT_headless_surface:
        case vvl::Extension::EXT_surface_maintenance1:
        case vvl::Extension::EXT_acquire_drm_display:
        case vvl::Extension::EXT_directfb_surface:
        case vvl::Extension::QNX_screen_surface:
        case vvl::Extension::GOOGLE_surfaceless_query:
        case vvl::Extension::LUNARG_direct_driver_loading:
        case vvl::Extension::EXT_layer_settings:
            return true;
        default:
            return false;
    }
}

constexpr bool IsDeviceExtension(vvl::Extension extension) {
    switch (extension) {
        case vvl::Extension::KHR_swapchain:
        case vvl::Extension::KHR_display_swapchain:
        case vvl::Extension::KHR_sampler_mirror_clamp_to_edge:
        case vvl::Extension::KHR_video_queue:
        case vvl::Extension::KHR_video_decode_queue:
        case vvl::Extension::KHR_video_encode_h264:
        case vvl::Extension::KHR_video_encode_h265:
        case vvl::Extension::KHR_video_decode_h264:
        case vvl::Extension::KHR_dynamic_rendering:
        case vvl::Extension::KHR_multiview:
        case vvl::Extension::KHR_device_group:
        case vvl::Extension::KHR_shader_draw_parameters:
        case vvl::Extension::KHR_maintenance1:
        case vvl::Extension::KHR_external_memory:
        case vvl::Extension::KHR_external_memory_win32:
        case vvl::Extension::KHR_external_memory_fd:
        case vvl::Extension::KHR_win32_keyed_mutex:
        case vvl::Extension::KHR_external_semaphore:
        case vvl::Extension::KHR_external_semaphore_win32:
        case vvl::Extension::KHR_external_semaphore_fd:
        case vvl::Extension::KHR_push_descriptor:
        case vvl::Extension::KHR_shader_float16_int8:
        case vvl::Extension::KHR_16bit_storage:
        case vvl::Extension::KHR_incremental_present:
        case vvl::Extension::KHR_descriptor_update_template:
        case vvl::Extension::KHR_imageless_framebuffer:
        case vvl::Extension::KHR_create_renderpass2:
        case vvl::Extension::KHR_shared_presentable_image:
        case vvl::Extension::KHR_external_fence:
        case vvl::Extension::KHR_external_fence_win32:
        case vvl::Extension::KHR_external_fence_fd:
        case vvl::Extension::KHR_performance_query:
        case vvl::Extension::KHR_maintenance2:
        case vvl::Extension::KHR_variable_pointers:
        case vvl::Extension::KHR_dedicated_allocation:
        case vvl::Extension::KHR_storage_buffer_storage_class:
        case vvl::Extension::KHR_relaxed_block_layout:
        case vvl::Extension::KHR_get_memory_requirements2:
        case vvl::Extension::KHR_image_format_list:
        case vvl::Extension::KHR_sampler_ycbcr_conversion:
        case vvl::Extension::KHR_bind_memory2:
        case vvl::Extension::KHR_portability_subset:
        case vvl::Extension::KHR_maintenance3:
        case vvl::Extension::KHR_draw_indirect_count:
        case vvl::Extension::KHR_shader_subgroup_extended_types:
        case vvl::Extension::KHR_8bit_storage:
        case vvl::Extension::KHR_shader_atomic_int64:
        case vvl::Extension::KHR_shader_clock:
        case vvl::Extension::KHR_video_decode_h265:
        case vvl::Extension::KHR_global_priority:
        case vvl::Extension::KHR_driver_properties:
        case vvl::Extension::KHR_shader_float_controls:
        case vvl::Extension::KHR_depth_stencil_resolve:
        case vvl::Extension::KHR_swapchain_mutable_format:
        case vvl::Extension::KHR_timeline_semaphore:
        case vvl::Extension::KHR_vulkan_memory_model:
        case vvl::Extension::KHR_shader_terminate_invocation:
        case vvl::Extension::KHR_fragment_shading_rate:
        case vvl::Extension::KHR_dynamic_rendering_local_read:
        case vvl::Extension::KHR_shader_quad_control:
        case vvl::Extension::KHR_spirv_1_4:
        case vvl::Extension::KHR_separate_depth_stencil_layouts:
        case vvl::Extension::KHR_present_wait:
        case vvl::Extension::KHR_uniform_buffer_standard_layout:
        case vvl::Extension::KHR_buffer_device_address:
        case vvl::Extension::KHR_deferred_host_operations:
        case vvl::Extension::KHR_pipeline_executable_properties:
        case vvl::Extension::KHR_map_memory2:
        case vvl::Extension::KHR_shader_integer_dot_product:
        case vvl::Extension::KHR_pipeline_library:
        case vvl::Extension::KHR_shader_non_semantic_info:
        case vvl::Extension::KHR_present_id:
        case vvl::Extension::KHR_video_encode_queue:
        case vvl::Extension::KHR_synchronization2:
        case vvl::Extension::KHR_fragment_shader_barycentric:
        case vvl::Extension::KHR_shader_subgroup_uniform_control_flow:
        case vvl::Extension::KHR_zero_initialize_workgroup_memory:
        case vvl::Extension::KHR_workgroup_memory_explicit_layout:
        case vvl::Extension::KHR_copy_commands2:
        case vvl::Extension::KHR_format_feature_flags2:
        case vvl::Extension::KHR_ray_tracing_maintenance1:
        case vvl::Extension::KHR_maintenance4:
        case vvl::Extension::KHR_shader_subgroup_rotate:
        case vvl::Extension::KHR_shader_maximal_reconvergence:
        case vvl::Extension::KHR_maintenance5:
        case vvl::Extension::KHR_ray_tracing_position_fetch:
        case vvl::Extension::KHR_cooperative_matrix:
        case vvl::Extension::KHR_video_maintenance1:
        case vvl::Extension::KHR_vertex_attribute_divisor:
        case vvl::Extension::KHR_load_store_op_none:
        case vvl::Extension::KHR_shader_float_controls2:
        case vvl::Extension::KHR_index_type_uint8:
        case vvl::Extension::KHR_line_rasterization:
        case vvl::Extension::KHR_calibrated_timestamps:
        case vvl::Extension::KHR_shader_expect_assume:
        case vvl::Extension::KHR_maintenance6:
        case vvl::Extension::NV_glsl_shader:
        case vvl::Extension::EXT_depth_range_unrestricted:
        case vvl::Extension::IMG_filter_cubic:
        case vvl::Extension::AMD_rasterization_order:
        case vvl::Extension::AMD_shader_trinary_minmax:
        case vvl::Extension::AMD_shader_explicit_vertex_parameter:
        case vvl::Extension::EXT_debug_marker:
        case vvl::Extension::AMD_gcn_shader:
        case vvl::Extension::NV_dedicated_allocation:
        case vvl::Extension::EXT_transform_feedback:
        case vvl::Extension::NVX_binary_import:
        case vvl::Extension::NVX_image_view_handle:
        case vvl::Extension::AMD_draw_indirect_count:
        case vvl::Extension::AMD_negative_viewport_height:
        case vvl::Extension::AMD_gpu_shader_half_float:
        case vvl::Extension::AMD_shader_ballot:
        case vvl::Extension::AMD_texture_gather_bias_lod:
        case vvl::Extension::AMD_shader_info:
        case vvl::Extension::AMD_shader_image_load_store_lod:
        case vvl::Extension::NV_corner_sampled_image:
        case vvl::Extension::IMG_format_pvrtc:
        case vvl::Extension::NV_external_memory:
        case vvl::Extension::NV_external_memory_win32:
        case vvl::Extension::NV_win32_keyed_mutex:
        case vvl::Extension::EXT_shader_subgroup_ballot:
        case vvl::Extension::EXT_shader_subgroup_vote:
        case vvl::Extension::EXT_texture_compression_astc_hdr:
        case vvl::Extension::EXT_astc_decode_mode:
        case vvl::Extension::EXT_pipeline_robustness:
        case vvl::Extension::EXT_conditional_rendering:
        case vvl::Extension::NV_clip_space_w_scaling:
        case vvl::Extension::EXT_display_control:
        case vvl::Extension::GOOGLE_display_timing:
        case vvl::Extension::NV_sample_mask_override_coverage:
        case vvl::Extension::NV_geometry_shader_passthrough:
        case vvl::Extension::NV_viewport_array2:
        case vvl::Extension::NVX_multiview_per_view_attributes:
        case vvl::Extension::NV_viewport_swizzle:
        case vvl::Extension::EXT_discard_rectangles:
        case vvl::Extension::EXT_conservative_rasterization:
        case vvl::Extension::EXT_depth_clip_enable:
        case vvl::Extension::EXT_hdr_metadata:
        case vvl::Extension::IMG_relaxed_line_rasterization:
        case vvl::Extension::EXT_external_memory_dma_buf:
        case vvl::Extension::EXT_queue_family_foreign:
        case vvl::Extension::ANDROID_external_memory_android_hardware_buffer:
        case vvl::Extension::EXT_sampler_filter_minmax:
        case vvl::Extension::AMD_gpu_shader_int16:
        case vvl::Extension::AMDX_shader_enqueue:
        case vvl::Extension::AMD_mixed_attachment_samples:
        case vvl::Extension::AMD_shader_fragment_mask:
        case vvl::Extension::EXT_inline_uniform_block:
        case vvl::Extension::EXT_shader_stencil_export:
        case vvl::Extension::EXT_sample_locations:
        case vvl::Extension::EXT_blend_operation_advanced:
        case vvl::Extension::NV_fragment_coverage_to_color:
        case vvl::Extension::NV_framebuffer_mixed_samples:
        case vvl::Extension::NV_fill_rectangle:
        case vvl::Extension::NV_shader_sm_builtins:
        case vvl::Extension::EXT_post_depth_coverage:
        case vvl::Extension::EXT_image_drm_format_modifier:
        case vvl::Extension::EXT_validation_cache:
        case vvl::Extension::EXT_descriptor_indexing:
        case vvl::Extension::EXT_shader_viewport_index_layer:
        case vvl::Extension::NV_shading_rate_image:
        case vvl::Extension::NV_ray_tracing:
        case vvl::Extension::NV_representative_fragment_test:
        case vvl::Extension::EXT_filter_cubic:
        case vvl::Extension::QCOM_render_pass_shader_resolve:
        case vvl::Extension::EXT_global_priority:
        case vvl::Extension::EXT_external_memory_host:
        case vvl::Extension::AMD_buffer_marker:
        case vvl::Extension::AMD_pipeline_compiler_control:
        case vvl::Extension::EXT_calibrated_timestamps:
        case vvl::Extension::AMD_shader_core_properties:
        case vvl::Extension::AMD_memory_overallocation_behavior:
        case vvl::Extension::EXT_vertex_attribute_divisor:
        case vvl::Extension::GGP_frame_token:
        case vvl::Extension::EXT_pipeline_creation_feedback:
        case vvl::Extension::NV_shader_subgroup_partitioned:
        case vvl::Extension::NV_compute_shader_derivatives:
        case vvl::Extension::NV_mesh_shader:
        case vvl::Extension::NV_fragment_shader_barycentric:
        case vvl::Extension::NV_shader_image_footprint:
        case vvl::Extension::NV_scissor_exclusive:
        case vvl::Extension::NV_device_diagnostic_checkpoints:
        case vvl::Extension::INTEL_shader_integer_functions2:
        case vvl::Extension::INTEL_performance_query:
        case vvl::Extension::EXT_pci_bus_info:
        case vvl::Extension::AMD_display_native_hdr:
        case vvl::Extension::EXT_fragment_density_map:
        case vvl::Extension::EXT_scalar_block_layout:
        case vvl::Extension::GOOGLE_hlsl_functionality1:
        case vvl::Extension::GOOGLE_decorate_string:
        case vvl::Extension::EXT_subgroup_size_control:
        case vvl::Extension::AMD_shader_core_properties2:
        case vvl::Extension::AMD_device_coherent_memory:
        case vvl::Extension::EXT_shader_image_atomic_int64:
        case vvl::Extension::EXT_memory_budget:
        case vvl::Extension::EXT_memory_priority:
        case vvl::Extension::NV_dedicated_allocation_image_aliasing:
        case vvl::Extension::EXT_buffer_device_address:
        case vvl::Extension::EXT_tooling_info:
        case vvl::Extension::EXT_separate_stencil_usage:
        case vvl::Extension::NV_cooperative_matrix:
        case vvl::Extension::NV_coverage_reduction_mode:
        case vvl::Extension::EXT_fragment_shader_interlock:
        case vvl::Extension::EXT_ycbcr_image_arrays:
        case vvl::Extension::EXT_provoking_vertex:
        case vvl::Extension::EXT_full_screen_exclusive:
        case vvl::Extension::EXT_line_rasterization:
        case vvl::Extension::EXT_shader_atomic_float:
        case vvl::Extension::EXT_host_query_reset:
        case vvl::Extension::EXT_index_type_uint8:
        case vvl::Extension::EXT_extended_dynamic_state:
        case vvl::Extension::EXT_host_image_copy:
        case vvl::Extension::EXT_shader_atomic_float2:
        case vvl::Extension::EXT_swapchain_maintenance1:
        case vvl::Extension::EXT_shader_demote_to_helper_invocation:
        case vvl::Extension::NV_device_generated_commands:
        case vvl::Extension::NV_inherited_viewport_scissor:
        case vvl::Extension::EXT_texel_buffer_alignment:
        case vvl::Extension::QCOM_render_pass_transform:
        case vvl::Extension::EXT_depth_bias_control:
        case vvl::Extension::EXT_device_memory_report:
        case vvl::Extension::EXT_robustness2:
        case vvl::Extension::EXT_custom_border_color:
        case vvl::Extension::GOOGLE_user_type:
        case vvl::Extension::NV_present_barrier:
        case vvl::Extension::EXT_private_data:
        case vvl::Extension::EXT_pipeline_creation_cache_control:
        case vvl::Extension::NV_device_diagnostics_config:
        case vvl::Extension::QCOM_render_pass_store_ops:
        case vvl::Extension::NV_cuda_kernel_launch:
        case vvl::Extension::NV_low_latency:
        case vvl::Extension::EXT_metal_objects:
        case vvl::Extension::EXT_descriptor_buffer:
        case vvl::Extension::EXT_graphics_pipeline_library:
        case vvl::Extension::AMD_shader_early_and_late_fragment_tests:
        case vvl::Extension::NV_fragment_shading_rate_enums:
        case vvl::Extension::NV_ray_tracing_motion_blur:
        case vvl::Extension::EXT_ycbcr_2plane_444_formats:
        case vvl::Extension::EXT_fragment_density_map2:
        case vvl::Extension::QCOM_rotated_copy_commands:
        case vvl::Extension::EXT_image_robustness:
        case vvl::Extension::EXT_image_compression_control:
        case vvl::Extension::EXT_attachment_feedback_loop_layout:
        case vvl::Extension::EXT_4444_formats:
        case vvl::Extension::EXT_device_fault:
        case vvl::Extension::ARM_rasterization_order_attachment_access:
        case vvl::Extension::EXT_rgba10x6_formats:
        case vvl::Extension::NV_acquire_winrt_display:
        case vvl::Extension::VALVE_mutable_descriptor_type:
        case vvl::Extension::EXT_vertex_input_dynamic_state:
        case vvl::Extension::EXT_physical_device_drm:
        case vvl::Extension::EXT_device_address_binding_report:
        case vvl::Extension::EXT_depth_clip_control:
        case vvl::Extension::EXT_primitive_topology_list_restart:
        case vvl::Extension::FUCHSIA_external_memory:
        case vvl::Extension::FUCHSIA_external_semaphore:
        case vvl::Extension::FUCHSIA_buffer_collection:
        case vvl::Extension::HUAWEI_subpass_shading:
        case vvl::Extension::HUAWEI_invocation_mask:
        case vvl::Extension::NV_external_memory_rdma:
        case vvl::Extension::EXT_pipeline_properties:
        case vvl::Extension::EXT_frame_boundary:
        case vvl::Extension::EXT_multisampled_render_to_single_sampled:
        case vvl::Extension::EXT_extended_dynamic_state2:
        case vvl::Extension::EXT_color_write_enable:
        case vvl::Extension::EXT_primitives_generated_query:
        case vvl::Extension::EXT_global_priority_query:
        case vvl::Extension::EXT_image_view_min_lod:
        case vvl::Extension::EXT_multi_draw:
        case vvl::Extension::EXT_image_2d_view_of_3d:
        case vvl::Extension::EXT_shader_tile_image:
        case vvl::Extension::EXT_opacity_micromap:
        case vvl::Extension::NV_displacement_micromap:
        case vvl::Extension::EXT_load_store_op_none:
        case vvl::Extension::HUAWEI_cluster_culling_shader:
        case vvl::Extension::EXT_border_color_swizzle:
        case vvl::Extension::EXT_pageable_device_local_memory:
        case vvl::Extension::ARM_shader_core_properties:
        case vvl::Extension::ARM_scheduling_controls:
        case vvl::Extension::EXT_image_sliced_view_of_3d:
        case vvl::Extension::VALVE_descriptor_set_host_mapping:
        case vvl::Extension::EXT_depth_clamp_zero_one:
        case vvl::Extension::EXT_non_seamless_cube_map:
        case vvl::Extension::ARM_render_pass_striped:
        case vvl::Extension::QCOM_fragment_density_map_offset:
        case vvl::Extension::NV_copy_memory_indirect:
        case vvl::Extension::NV_memory_decompression:
        case vvl::Extension::NV_device_generated_commands_compute:
        case vvl::Extension::NV_linear_color_attachment:
        case vvl::Extension::EXT_image_compression_control_swapchain:
        case vvl::Extension::QCOM_image_processing:
        case vvl::Extension::EXT_nested_command_buffer:
        case vvl::Extension::EXT_external_memory_acquire_unmodified:
        case vvl::Extension::EXT_extended_dynamic_state3:
        case vvl::Extension::EXT_subpass_merge_feedback:
        case vvl::Extension::EXT_shader_module_identifier:
        case vvl::Extension::EXT_rasterization_order_attachment_access:
        case vvl::Extension::NV_optical_flow:
        case vvl::Extension::EXT_legacy_dithering:
        case vvl::Extension::EXT_pipeline_protected_access:
        case vvl::Extension::ANDROID_external_format_resolve:
        case vvl::Extension::EXT_shader_object:
        case vvl::Extension::QCOM_tile_properties:
        case vvl::Extension::SEC_amigo_profiling:
        case vvl::Extension::QCOM_multiview_per_view_viewports:
        case vvl::Extension::NV_ray_tracing_invocation_reorder:
        case vvl::Extension::NV_extended_sparse_address_space:
        case vvl::Extension::EXT_mutable_descriptor_type:
        case vvl::Extension::ARM_shader_core_builtins:
        case vvl::Extension::EXT_pipeline_library_group_handles:
        case vvl::Extension::EXT_dynamic_rendering_unused_attachments:
        case vvl::Extension::NV_low_latency2:
        case vvl::Extension::QCOM_multiview_per_view_render_areas:
        case vvl::Extension::NV_per_stage_descriptor_set:
        case vvl::Extension::QCOM_image_processing2:
        case vvl::Extension::QCOM_filter_cubic_weights:
        case vvl::Extension::QCOM_ycbcr_degamma:
        case vvl::Extension::QCOM_filter_cubic_clamp:
        case vvl::Extension::EXT_attachment_feedback_loop_dynamic_state:
        case vvl::Extension::QNX_external_memory_screen_buffer:
        case vvl::Extension::MSFT_layered_driver:
        case vvl::Extension::NV_descriptor_pool_overallocation:
        case vvl::Extension::KHR_acceleration_structure:
        case vvl::Extension::KHR_ray_tracing_pipeline:
        case vvl::Extension::KHR_ray_query:
        case vvl::Extension::EXT_mesh_shader:
            return true;
        default:
            return false;
    }
}

// NOLINTEND
