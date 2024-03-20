# Install dependencies
conda install -c anaconda boost 
conda install cmake clang clangxx_osx-64 clang_osx-64 zlib bzip2

export LDFLAGS+=" -std=c++11"
export CXXFLAGS+=" -std=c++11"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

cmake --version
clang --version

# Clone the KenLM repository
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Build and install KenLM
mkdir build && cd build
cmake ..
make -j2
sudo make install

# Add KenLM to the conda environment
echo "export KENLM_ROOT_DIR=$(pwd)" >> $CONDA_PREFIX/etc/conda/activate.d/kenlm.sh
echo "export LD_LIBRARY_PATH=$(pwd)/lib:\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/kenlm.sh
echo "unset KENLM_ROOT_DIR" >> $CONDA_PREFIX/etc/conda/deactivate.d/kenlm.sh
echo "unset LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/kenlm.sh

# Activate the conda environment again to apply the changes
cd ../..
conda deactivate
conda activate ngram
pip install ./kenlm