if $(conda info --envs | grep '*' | grep -q 'base'); then
  set -e

  export CXX=clang++
  export CC=clang
  export CXXFLAGS+=" -std=c++11"
  export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

  if [ -d "kenlm" ]; then
    echo "kenlm is already downloaded"
  else
    brew install cmake boost eigen
    git clone https://github.com/kpu/kenlm.git
  fi

  if [ -d "kenlm" ] && [ -d "kenlm/build" ] && [ -d "kenlm/build/bin" ] && $(ls -A "kenlm/build/bin" | grep -q .); then
    echo "kenlm is already built"
  else
    cd kenlm
    mkdir build && cd build
    cmake ..
    make -j2
  fi
else
  echo "Please deactivate your environment and return to base before running this script."
fi