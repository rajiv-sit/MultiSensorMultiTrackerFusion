# Initial setup
cmake_minimum_required(VERSION 3.28.1)
project(MultiSensorMultiTrackerFusion LANGUAGES CXX C VERSION 0.0.1)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
set(CMAKE_CXX_STANDARD 20)

# Default configuration values
set(DEFAULT_BUILD_TYPE "Release")
set(DEFAULT_ARCH "x86-64")

# Detect platform
string(COMPARE EQUAL "Linux"   ${CMAKE_SYSTEM_NAME} LINUX)
string(COMPARE EQUAL "Windows" ${CMAKE_SYSTEM_NAME} WINDOWS)
if(LINUX)
    set(OS_STRING "linux")
elseif(WINDOWS)
    set(OS_STRING "windows")
else()
    set(OS_STRING "Unknown")
endif()

# Detect compiler
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CLANG 1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CLANG 1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GCC 1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(INTEL 1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(MSVC 1)
endif()

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)

# Set architecture to default if not set
if(NOT ARCH)
    set(ARCH ${DEFAULT_ARCH})
endif()

# Set bitness
if(${ARCH} STREQUAL "x86-64")
    set(BITNESS "64")
    set(BITNESS_FLAG "-m${BITNESS}")
elseif(${ARCH} STREQUAL "x86")
    set(BITNESS "32")
    set(BITNESS_FLAG "-m${BITNESS}")
elseif(${ARCH} STREQUAL "armv8")
    set(BITNESS "64")
    set(BITNESS_FLAG "")
else()
    set(BITNESS "32")
    set(ARCH "x86")
    set(BITNESS_FLAG "-m${BITNESS}")
    message("Unknown architecture selected, defaulting to x86")
endif()

if(MSVC)
    # User cannot specify bitness with MSVC, so set it to whatever the generator is.
    string(TOLOWER ${CMAKE_GENERATOR} GENERATOR)
    if(GENERATOR MATCHES ".*win64.*")
        set(BITNESS "64")
    else()
        set(BITNESS "32")
    endif()
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ox")

elseif(GCC OR CLANG)
    # Treat warning return-type as error to avoid undefined behaviour
    # when a non-void function does not return a value.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BITNESS_FLAG} -std=c++11 -Werror=return-type")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-long-long")

elseif(INTEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BITNESS_FLAG} -std=c++11")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
endif()

if(WINDOWS) 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:100000000")
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
add_compile_definitions(_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
endif(WINDOWS) 

find_package(Eigen3 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glew REQUIRED)
find_package(imgui REQUIRED)
find_package(opengl REQUIRED)
find_package(GTest REQUIRED)

set(CPACK_NSIS_CONTACT "rajiv.sithiravel@gmail.com")

set(TRACKER_PROCESS_MODELS_SRC_LIST
    include/process_models/advanced_learned/gp.hpp
	include/process_models/advanced_learned/intent.hpp
	include/process_models/advanced_learned/mjls.hpp
	include/process_models/advanced_learned/neural.hpp
	include/process_models/advanced_learned/physics_neural.hpp
	include/process_models/advanced_learned/social.hpp
	include/process_models/advanced_learned/spline.hpp	
	include/process_models/aerial_space/ballistic.hpp
	include/process_models/aerial_space/cav.hpp
	include/process_models/aerial_space/const_force.hpp
	include/process_models/aerial_space/constant_specific_force.hpp
	include/process_models/aerial_space/hcw.hpp
	include/process_models/aerial_space/kepler.hpp
	include/process_models/aerial_space/perturbed_orbit.hpp
	include/process_models/aerial_space/reentry.hpp
	include/process_models/classical_baseline/brownian.hpp
	include/process_models/classical_baseline/ca.hpp
	include/process_models/classical_baseline/cj.hpp
	include/process_models/classical_baseline/ct.hpp
	include/process_models/classical_baseline/cta.hpp
	include/process_models/classical_baseline/ctra.hpp
	include/process_models/classical_baseline/cv.hpp
	include/process_models/classical_baseline/nca.hpp
	include/process_models/classical_baseline/ncv.hpp
	include/process_models/classical_baseline/singer.hpp
    include/process_models/classical_baseline/wna.hpp	
	include/process_models/domain_specific/animal.hpp
	include/process_models/domain_specific/auv.hpp
	include/process_models/domain_specific/drone_swarm.hpp
	include/process_models/domain_specific/finance.hpp
	include/process_models/domain_specific/human.hpp
	include/process_models/domain_specific/magnetic.hpp
	include/process_models/maritime/ccs.hpp
	include/process_models/maritime/maneuvering_ship.hpp
	include/process_models/maritime/wave_random.hpp	
	include/process_models/multi_domain/adaptive_cv.hpp
	include/process_models/multi_domain/cv_ca_imm.hpp
	include/process_models/multi_domain/cv_ct_imm.hpp
	include/process_models/multi_domain/hybrid_kinematic.hpp
	include/process_models/multi_domain/imm.hpp
	include/process_models/multi_domain/state_augmented.hpp
	include/process_models/nonlinear_extended/bearing_only.hpp
	include/process_models/nonlinear_extended/camera_plane.hpp
	include/process_models/nonlinear_extended/cv_altitude.hpp
	include/process_models/nonlinear_extended/cv_ca_3d.hpp
	include/process_models/nonlinear_extended/extended_cv.hpp
	include/process_models/nonlinear_extended/polar.hpp
	include/process_models/nonlinear_extended/range_only.hpp
	include/process_models/rfs_multi_target/lmb.hpp
	include/process_models/rfs_multi_target/phd.hpp
	include/process_models/rfs_multi_target/pmbm.hpp
	include/process_models/rfs_multi_target/spline_phd.hpp	
	include/process_models/vehicle/ackermann.hpp
	include/process_models/vehicle/bicycle.hpp
	include/process_models/vehicle/ctra.hpp
	include/process_models/vehicle/ctrv.hpp
	include/process_models/vehicle/dubins.hpp
	include/process_models/vehicle/lane_constrained.hpp
	include/process_models/vehicle/unicycle.hpp
)

set(TEST_SRC_LIST
    test/main.cpp
)

# add sources 
set(BINDINGS_SRC_LIST
	${PROJECT_SOURCE_DIR}/Bindings/imgui_impl_glfw.cpp
	${PROJECT_SOURCE_DIR}/Bindings/imgui_impl_opengl3.cpp	
)

add_executable(${PROJECT_NAME_LOWER} 
	${TRACKER_PROCESS_MODELS_SRC_LIST}
	${BINDINGS_SRC_LIST}
	${TEST_SRC_LIST}
)
	
if(WINDOWS) 				
target_compile_options(${PROJECT_NAME_LOWER} PRIVATE "/Od")   
endif(WINDOWS)

target_include_directories(${PROJECT_NAME_LOWER} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/process_models ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/Bindings ${PROJECT_SOURCE_DIR}/test) 
target_link_libraries(${PROJECT_NAME_LOWER} Eigen3::Eigen glfw GLEW::GLEW imgui::imgui opengl::opengl )

if(CMAKE_BUILD_TYPE STREQUAL DEBUG)
    message("Detected compiler and platform:")
	message("Clang:   ${CLANG}")
	message("GCC:     ${GCC}")
	message("Intel:   ${INTEL}")
	message("MSVC:    ${MSVC}")
	message("Linux:   ${LINUX}")
	message("Windows: ${WINDOWS}")
	message("OS X:    ${OS_X}")
endif()

message("Configuring ${PROJECT_NAME} version ${VERSION} in ${CMAKE_BUILD_TYPE} mode for ${ARCH} (${BITNESS} bit)")
message("Compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# --- Copy configuration file to build output directory ---
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/config)
configure_file(${CMAKE_SOURCE_DIR}/config/sim_config.ini ${CMAKE_BINARY_DIR}/config/sim_config.ini COPYONLY)

# -------------------------------
# Unit tests project
# -------------------------------
option(BUILD_TESTS "Build GoogleTest unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    add_executable(${PROJECT_NAME_LOWER}_tests
        test/test_cvmodel.cpp
    )

    target_link_libraries(${PROJECT_NAME_LOWER}_tests
        gtest::gtest
        Eigen3::Eigen
    )

    target_include_directories(${PROJECT_NAME_LOWER}_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/include/process_models
    )

    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME_LOWER}_tests)
endif()

