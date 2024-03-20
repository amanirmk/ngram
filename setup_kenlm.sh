set -e

export CXX=clang++
export CC=clang
export CXXFLAGS+=" -std=c++11"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

brew install cmake boost eigen
git clone https://github.com/kpu/kenlm.git
cd kenlm

mkdir build && cd build
cmake ..
make -j2

# try getting kenlm from surprisal