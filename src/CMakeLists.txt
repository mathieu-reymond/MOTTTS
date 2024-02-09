cmake_minimum_required (VERSION 3.0)

#######################################
##   Experiment Creation Functions   ##
#######################################

# These functions can be used below to easily add experiments to build. The
# only obligatory parameter is the name of the experiment (usually in the format
# experimentname_methodname). Each function can be used depending on which
# features of AIToolbox you need.
#
# If you need to link the experiment file against some additional libraries,
# you can add them as optional arguments after the name.

function (AddExperimentMDP name)
    set(MAKE_MDP 1 CACHE INTERNAL "")
    set(exename ${name}_main)
    add_executable(${exename} ${exename}.cpp)
    target_link_libraries(${exename} AIToolboxMDP ${Boost_LIBRARIES} ${ARGN})
endfunction (AddExperimentMDP)

function (AddExperimentPOMDP name)
    set(MAKE_POMDP 1 CACHE INTERNAL "")
    set(exename ${name}_main)
    add_executable(${exename} ${exename}.cpp)
    target_link_libraries(${exename} AIToolboxMDP AIToolboxPOMDP ${Boost_LIBRARIES} ${ARGN})
endfunction (AddExperimentPOMDP)

function (AddExperimentFMDP name)
    set(MAKE_FMDP 1 CACHE INTERNAL "")
    set(exename ${name}_main)
    add_executable(${exename} ${exename}.cpp)
    target_link_libraries(${exename} AIToolboxMDP AIToolboxFMDP ${Boost_LIBRARIES} ${ARGN})
endfunction (AddExperimentFMDP)

# Here we simply add a definition in case somebody is asking for logging.
if (${AI_LOGGING_ENABLED})
    add_definitions(-DAI_LOGGING_ENABLED)
endif()

############################
##   Experiment Library   ##
############################

# If you need other cpp files to compile for the experiment, enable the
# following library. Remember to add "ExperimentLibrary" as an additional
# parameter to the "AddExperimentXXX" functions (they'll link the
# ExperimentLibrary then through the ARGN parameter).

# Simple example with library made of file1.cpp and file2.cpp:
#
add_library(ExperimentLibrary
    ./Utils.cpp
    ./newtest.cpp
    ./MONormalReward.cpp
    ./MOExperience.cpp
    ./MOThompsonSamplingPolicy.cpp
    ./MOTTTS.cpp
    ./UtilityFunction.cpp
    ./UtilityFunctionPosterior.cpp
    ./BayesianLogisticRegression.cpp
    ./MOBanditNormalPosterior.cpp
    ./Statistics.cpp
    ./IO.cpp
    ./UtilityFunctionParticlePosterior.cpp
    ./generate_mobandit.cpp
)
target_link_libraries(ExperimentLibrary AIToolboxMDP ${Boost_LIBRARIES} ${Python_LIBRARIES})


############################
## Experiment Declaration ##
############################

# Experiments should be in the format:
#
#     experimentname_methodname (e.g. mines_mauce)
#
# Thus you need to create an experiment file with name:
#
#     experimentname_methodname_main.cpp (e.g. mines_mauce_main.cpp)
#
# And add the name of the experiment here with the AddExperimentXXX methods
# (depending on which features of AIToolbox you need).

AddExperimentMDP(generate_mobandit ExperimentLibrary OpenMP::OpenMP_CXX)
AddExperimentMDP(its ExperimentLibrary OpenMP::OpenMP_CXX)

## For Python Experiments ##
#
# set(MAKE_PYTHON 1 CACHE INTERNAL "")
# find_package(PythonLibs 2.7 REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})
#
# Then you need to call AddExperimentXXX(name ${PYTHON_LIBRARIES})

