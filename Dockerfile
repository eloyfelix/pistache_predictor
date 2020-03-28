FROM debian:buster-slim

ARG RDKIT_VERSION=Release_2020_03_1b1

RUN apt-get update --fix-missing && \
    apt-get install -y g++ \
                       cmake \
                       curl \
                       unzip \
                       git \
                       nlohmann-json-dev \
                       libboost-dev \
                       libboost-iostreams-dev \
                       libboost-system-dev \
                       libboost-serialization-dev && \
    apt-get -qq -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Install RDKit
RUN curl -LO https://github.com/rdkit/rdkit/archive/${RDKIT_VERSION}.tar.gz && \
    tar -xzf ${RDKIT_VERSION}.tar.gz && \
    mv rdkit-${RDKIT_VERSION} rdkit && \
    mkdir rdkit/build && \
    cd /rdkit/build && \
    cmake -Wno-dev \
          -DCMAKE_BUILD_TYPE=Release \
          -DRDK_BUILD_INCHI_SUPPORT=ON \
          -DCMAKE_INSTALL_PREFIX=/usr \
          -DCMAKE_SYSTEM_PREFIX_PATH=/usr \
          -DRDK_BUILD_PYTHON_WRAPPERS=OFF \
          -DRDK_INSTALL_INTREE=OFF \
          -DRDK_BUILD_CPP_TESTS=OFF \
          .. && \
    make -j $(nproc) && \
    make install && \
    rm -rf /rdkit /${RDKIT_VERSION}.tar.gz

# Install pistache
RUN git clone https://github.com/oktal/pistache.git && \
    cd pistache && \
    git submodule update --init && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. && \
    make && \
    make install && \
    rm -rf /pistache*

# Install torchlib
RUN curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip

COPY src app/src
COPY CMakeLists.txt app/CMakeLists.txt

RUN mkdir app/build && \
    cd app/build && \
    cmake .. && \
    make && \
    cp PistachePredictor ../../PistachePredictor

CMD ["./PistachePredictor", "9999", "4"]
