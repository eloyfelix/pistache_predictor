FROM debian:bullseye-slim

ARG CHEMBL_VERSION=chembl_33
ARG RDKIT_VERSION=Release_2023_03_2
ARG ONNX_VERSION=1.15.1
ARG PISTACHE_COMMIT=a68ad0902d2cfc23f69fc16e26747ac77bc2f123

RUN apt-get update --fix-missing && \
    apt-get install -y g++ \
                       cmake \
                       pkg-config \
                       meson \
                       python3-setuptools \
                       libssl-dev \
                       curl \
                       unzip \
                       git \
                       zlib1g-dev \
                       nlohmann-json3-dev \
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
          -DRDK_BUILD_FREETYPE_SUPPORT=OFF \
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
RUN git clone https://github.com/pistacheio/pistache.git && \
    cd pistache && \
    git checkout ${PISTACHE_COMMIT} && \
    meson setup build \
      --buildtype=release \
      -DPISTACHE_USE_SSL=true \
      -DPISTACHE_BUILD_EXAMPLES=false \
      -DPISTACHE_BUILD_TESTS=false \
      -DPISTACHE_BUILD_DOCS=false \
      --prefix=/usr && \
    meson compile -C build && \
    meson install -C build

# Install MS ONNX runtime
ENV ONNXRUNTIME_ROOTDIR=/onnxruntime-linux-x64-${ONNX_VERSION}
RUN curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz && \
    rm onnxruntime-linux-x64-${ONNX_VERSION}.tgz

COPY src app/src
COPY CMakeLists.txt app/CMakeLists.txt

# Download the onnx model file
RUN curl -LJ https://github.com/chembl/chembl_multitask_model/raw/main/trained_models/${CHEMBL_VERSION}_model/${CHEMBL_VERSION}_multitask_q8.onnx -o app/src/chembl_multitask.onnx

RUN mkdir app/build && \
    cd app/build && \
    cmake .. && \
    make && \
    cp PistachePredictor ../../PistachePredictor

CMD ["./PistachePredictor", "9080", "4"]
