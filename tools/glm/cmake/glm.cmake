# tools/glm/cmake/glm.cmake - the GLM-5.3-Flash engine (GlmDense + the shared Ds4MoeTier).  include()d (OPTIONAL) at the
# end of the root CMakeLists.txt, after the tools/ds4 slices it builds on.
#   CUDA tree: glm_generate      (dense on the CUDA backend, the full expert tier)
#   CPU tree : glm_generate_cpu  (CPU backend + the CPU-only tier; mini fixtures)
if(NOT TARGET ggml-cpu)
  return()
endif()
if(STRATA_HIP_GFX906 AND NOT TARGET ggml-cuda)
  # gfx906 fork: the dense half needs the xolotl-patched ggml compiled as the HIP ggml-cuda backend
  # (tools/gfx906/ggml/hip-cmake.patch + -DSTRATA_GGML_CUDA=ON -DSTRATA_GGML_DIR=<patched clone>).  With that
  # target absent the engine stays out of the build, exactly as on a ggml-less machine.  Port log:
  # docs/gfx906/XOLOTL-PORT.md (ggml-CUDA-as-HIP built 2026-10-18).
  return()
endif()
add_library(glm_dense STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm/glm_dense.cpp)
# ggml-impl.h (graph uid): honour STRATA_GGML_DIR so a worktree without third_party/llama.cpp still builds here
set(_glm_ggml_src "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src")
if(STRATA_GGML_DIR)
  set(_glm_ggml_src "${STRATA_GGML_DIR}/ggml/src")
endif()
target_include_directories(glm_dense PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm
                           PRIVATE ${_glm_ggml_src})
target_link_libraries(glm_dense PRIVATE ggml ggml-cpu ggml-base)
if(TARGET ggml-cuda)
  target_link_libraries(glm_dense PRIVATE ggml-cuda)
endif()
# P5.2: tools/gfx906 holds device_split.hpp, the header-only --gpus/--layer-split stage math (header-only so
# tools/gfx906/test_device_split.cpp can compile it with plain g++, no CUDA/HIP).
set(_glm_inc ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4 ${CMAKE_CURRENT_SOURCE_DIR}/tools/gfx906)
if(TARGET ds4_moe_cuda)
  add_executable(glm_generate ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm/glm_generate.cpp)
  target_include_directories(glm_generate PRIVATE ${_glm_inc})
  target_link_libraries(glm_generate PRIVATE glm_dense ds4_moe_cuda ggml ggml-base)
  if(TARGET ggml-cuda)
    target_link_libraries(glm_generate PRIVATE ggml-cuda)
  endif()
elseif(TARGET ds4_moe_cpu)
  add_executable(glm_generate_cpu ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm/glm_generate.cpp)
  target_include_directories(glm_generate_cpu PRIVATE ${_glm_inc})
  target_link_libraries(glm_generate_cpu PRIVATE glm_dense ds4_moe_cpu ggml ggml-cpu ggml-base)
endif()
