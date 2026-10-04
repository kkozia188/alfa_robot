cmake_minimum_required(VERSION 3.16)
project(v3_collision_benchmark)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(srdfdom REQUIRED)
find_package(urdf REQUIRED)

add_executable(v3_mesh_fcl_benchmark v3_mesh_fcl_benchmark.cpp)
target_compile_features(v3_mesh_fcl_benchmark PRIVATE cxx_std_17)
ament_target_dependencies(v3_mesh_fcl_benchmark
  moveit_core
  moveit_ros_planning
  srdfdom
  urdf
)
