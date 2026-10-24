# tools/ds4/cmake/ggml_cuda.cmake - fork: ggml's own CUDA backend for the DS4 dense half (one include() line in the
# root CMakeLists.txt, right after upstream sets GGML_CUDA OFF; docs/UPSTREAM_SYNC.md).
# ggml's own CUDA backend is off unless asked for: the DS4 dense half (tools/ds4/ds4_dense) runs its graphs on a
# ggml backend and needs ggml-cuda to run them on the GPU (-DSTRATA_GGML_CUDA=ON, build-ds4-gpu).
option(STRATA_GGML_CUDA "Build ggml's CUDA backend too (for the DS4 dense half on the GPU)" OFF)
if(STRATA_GGML_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  # ggml defaults CUDA graphs OFF when it is not llama.cpp's own build; without them the DS4 dense half's ~17k tiny
  # kernels per token run at launch-overhead speed (measured: 97 ms/token of attention, 2.1M cudaLaunchKernel)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
  if(STRATA_HIP_GFX906)
    # HIP: ggml-cuda/CMakeLists.txt needs tools/gfx906/ggml/hip-cmake.patch (skips find_package(CUDAToolkit) -
    # the CUDA::cudart/CUDA::cublas imported targets above already carry hip::host/roc::hipblas).
    # NO_VMM: the CUDA driver link and the cu* VMM path stay off - AND note common.cuh:263 honors only
    # GGML_HIP_NO_VMM under HIP (hip-cmake.patch adds it; witnessed MI50/ROCm 7.2: attribute says VMM yes,
    # hipMemSetAccess aborts at first pool growth). ALL_WARNINGS: the warning-probe block shells out
    # to nvcc, which does not exist here.
    set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)
    set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
    if(STRATA_GGML_DIR)
      # Two ways the checkout can say it carries the patch.  (1) The patch is applied and nothing else touched the
      # two files it touches, so a reverse-apply checks clean.  (2) The checkout carries the patch's EFFECT plus
      # further gfx906 edits of its own - which is the case on the box this fork was built on, where
      # ggml-cuda/CMakeLists.txt also sets GGML_HIP_NO_VMM (common.cuh:264 honours only that name under HIP).
      # A reverse-apply fails there and used to refuse the configure, so the marker line the patch itself inserts is
      # accepted as well.  The check still refuses a plain upstream checkout, which is what it is for.
      get_filename_component(_hip_ggml_parent ${STRATA_GGML_DIR} DIRECTORY)
      execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${_hip_ggml_parent}
                              git apply --reverse --check ${CMAKE_CURRENT_SOURCE_DIR}/tools/gfx906/ggml/hip-cmake.patch
                      WORKING_DIRECTORY ${STRATA_GGML_DIR}
                      RESULT_VARIABLE _hip_cmake_patched OUTPUT_QUIET ERROR_QUIET)
      if(NOT _hip_cmake_patched EQUAL 0)
        set(_hip_ggml_cuda_cmake ${STRATA_GGML_DIR}/ggml/src/ggml-cuda/CMakeLists.txt)
        if(EXISTS ${_hip_ggml_cuda_cmake})
          file(READ ${_hip_ggml_cuda_cmake} _hip_ggml_cuda_src)
          if(_hip_ggml_cuda_src MATCHES "gfx906 fork \\(tools/gfx906/ggml/hip-cmake\\.patch\\)")
            message(STATUS "ggml-cuda: hip-cmake.patch is not a clean reverse-apply (the checkout carries further "
                           "gfx906 edits); its marker line is present, so it is treated as applied")
            set(_hip_cmake_patched 0)
          endif()
        endif()
      endif()
      if(NOT _hip_cmake_patched EQUAL 0)
        message(FATAL_ERROR "STRATA_HIP_GFX906 + STRATA_GGML_CUDA: STRATA_GGML_DIR=${STRATA_GGML_DIR} does not carry "
                            "tools/gfx906/ggml/hip-cmake.patch - apply it to the llama.cpp checkout first "
                            "(git apply tools/gfx906/ggml/hip-cmake.patch); the GGML_HIP option in this llama.cpp "
                            "commit is dead code, the CMakeLists does not act on it")
      endif()
    endif()
  endif()
endif()
