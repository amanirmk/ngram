# Install dependencies
# conda install -c anaconda boost 
# conda install cmake clang clangxx_osx-64 clang_osx-64 zlib bzip2

# Export flags
export CXX=clang++
export CC=clang
export CXXFLAGS+=" -std=c++11"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

# Deactivate environment before building
# conda deactivate

# Clone the KenLM repository
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Build and install KenLM
mkdir build && cd build
cmake ..
make -j2

# Activate the conda environment again
# conda activate ngram

# Add KenLM to the conda environment
# echo "export KENLM_ROOT_DIR=$(pwd)" >> $CONDA_PREFIX/etc/conda/activate.d/kenlm.sh
# echo "export LD_LIBRARY_PATH=$(pwd)/lib:\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/kenlm.sh
# echo "unset KENLM_ROOT_DIR" >> $CONDA_PREFIX/etc/conda/deactivate.d/kenlm.sh
# echo "unset LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/kenlm.sh

# # Restart environment
# conda deactivate
# conda activate ngram

# Install the Python package
# cd ../..
# pip install ./kenlm