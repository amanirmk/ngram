#!/bin/bash

# Exit on first error
set -e

# Set compiler and SDK environment variables
export CXX=clang++
export CC=clang
export CXXFLAGS+=" -std=c++11"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

# Clone vcpkg if it doesn't exist
if [ ! -d "vcpkg" ]; then
  git clone https://github.com/Microsoft/vcpkg.git
fi

# Navigate to the vcpkg directory
cd vcpkg

# Bootstrap vcpkg if it hasn't been bootstrapped
if [ ! -f "vcpkg" ]; then
  ./bootstrap-vcpkg.sh
fi

# Integrate vcpkg with user-wide integration
./vcpkg integrate install

# Set the CMAKE_TOOLCHAIN_FILE environment variable
export CMAKE_TOOLCHAIN_FILE=$(pwd)/scripts/buildsystems/vcpkg.cmake

# Install kenlm
./vcpkg install kenlm

# Navigate back to the original directory
cd ..