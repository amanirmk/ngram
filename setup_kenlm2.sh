conda install clangxx_osx-64 clang_osx-64 zlib bzip2

export ENV_LOC=$(conda info --envs | grep '*' | awk '{print $3}')
export CC=$ENV_LOC/bin/clang
export CXX=$ENV_LOC/bin/clang++
export LDFLAGS+=" -std=c++11"
export CXXFLAGS+=" -std=c++11"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
echo $SDKROOT
echo $(xcrun --sdk macosx --show-sdk-path)

# pip install pypi-kenlm
# brew install cmake boost eigen
# mkdir -p build
# cd build
# cmake ..
# make -j 4