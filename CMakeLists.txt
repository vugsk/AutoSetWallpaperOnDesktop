cmake_minimum_required(VERSION 3.30)
project(AutoSetWallpaperOnDesktop)

set(CMAKE_CXX_STANDARD 20)

# OpenCV v4.0
find_package( OpenCV 4 REQUIRED COMPONENTS core imgproc ximgproc )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# nlohmann
find_package(nlohmann_json 3.2.0 REQUIRED)


add_executable(AutoSetWallpaperOnDesktop src/main.cpp)
target_link_libraries( AutoSetWallpaperOnDesktop PUBLIC
                            ${OpenCV_LIBS} nlohmann_json::nlohmann_json)