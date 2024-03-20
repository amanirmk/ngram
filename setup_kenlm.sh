# Ensure dependencies
conda install clangxx_osx-64 clang_osx-64 zlib bzip2 cmake

# Set environment variables
export ENV_LOC=$(conda info --envs | grep '*' | awk '{print $3}')
export BOOST_ROOT=$ENV_LOC
export CC=gcc
export CXX=g++
export LDFLAGS+=" -std=c++11"
export CXXFLAGS+=" -std=c++11"
export CFLAGS="-march=armv8-a"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

# Install Boost
wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.bz2
tar --bzip2 -xf boost_1_82_0.tar.bz2
cd boost_1_82_0
./bootstrap.sh --with-libraries=all -prefix=$BOOST_ROOT --with-toolset=gcc
./b2 install --prefix=$BOOST_ROOT
cd ..
rm -rf boost_1_82_0
rm -rf boost_1_82_0.tar.bz2

# Install KenLM
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake .. -DCMAKE_CXX_FLAGS="-std=c++11" -DENABLE_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES=arm64
make -j2
cd ../../




# # Ensure dependencies
# conda install clangxx_osx-64 clang_osx-64 zlib bzip2 cmake

# # Set environment variables
# export ENV_LOC=$(conda info --envs | grep '*' | awk '{print $3}')
# export BOOST_ROOT=$ENV_LOC
# export CC=clang
# export CXX=clang++
# export LDFLAGS+=" -std=c++11"
# export CXXFLAGS+=" -std=c++11"
# export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

# # Install Boost
# wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.bz2
# tar --bzip2 -xf boost_1_82_0.tar.bz2
# cd boost_1_82_0
# ./bootstrap.sh --with-libraries=all --prefix=$BOOST_ROOT --with-toolset=clang
# ./b2 install --prefix=$BOOST_ROOT
# cd ..

# # Install KenLM
# wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
# mkdir kenlm/build
# cd kenlm/build
# cmake .. -DCMAKE_CXX_FLAGS="-std=c++11" -DENABLE_OPENMP=OFF
# make -j2 -v
# cd ../../