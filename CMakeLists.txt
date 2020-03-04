cmake_minimum_required(VERSION 3.10.0)
project(fast_ls)

find_package(Eigen3 REQUIRED)

set(CMAKE_BUILD_TYPE "Release")

###########
## Build ##
###########

add_executable(fast_ls
  src/fast_ls.cpp
)
target_include_directories(fast_ls PUBLIC
  include
  ${EIGEN3_INCLUDE_DIR}
)
