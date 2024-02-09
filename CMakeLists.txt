cmake_minimum_required (VERSION 3.0)

project (Experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(
    -Wall
    -Wextra
    # Other flags
    -fopenmp
)

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Add AIToolbox dependencies
find_package(Boost 1.53 COMPONENTS program_options REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 3.4 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(LpSolve REQUIRED)
include_directories(${LPSOLVE_INCLUDE_DIR})

find_package(OpenMP REQUIRED)

set(DEPENDENCIES "")
function (AddDependency name)
    set(DEPENDENCIES ${DEPENDENCIES} ${name} PARENT_SCOPE)
endfunction (AddDependency)

###################
#  DEPENDENCIES  ##
###################

# Add AI-Toolbox folder
AddDependency(AI-Toolbox)

##################
#  FINAL SETUP  ##
##################

# We first include all include directories so that the experiments can find the
# files.
foreach (DEPENDENCY ${DEPENDENCIES})
    include_directories(${PROJECT_SOURCE_DIR}/deps/${DEPENDENCY}/include)
endforeach(DEPENDENCY)

# Add project directories
# These will automatically set the parameters for building AI-Toolbox
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# Finally we add the dependencies' subdirectories to build them, knowing we
# have setup the parameters in the src directory.
foreach (DEPENDENCY ${DEPENDENCIES})
    add_subdirectory(${PROJECT_SOURCE_DIR}/deps/${DEPENDENCY})
endforeach(DEPENDENCY)
