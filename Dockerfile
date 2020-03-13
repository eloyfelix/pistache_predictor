FROM ubuntu:disco

ARG RDKIT_VERSION=Release_2019_09_3

RUN apt-get update && \
    apt-get install -y gnupg2 && \
    apt-get -qq -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN echo deb http://ppa.launchpad.net/pistache+team/unstable/ubuntu disco main >> /etc/apt/sources.list
RUN echo deb-src http://ppa.launchpad.net/pistache+team/unstable/ubuntu disco main  >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 2EEA295DCBF66B6DE281E0A193E2268577BD194B 

# install required ubuntu packages
RUN apt-get update --fix-missing && \
    apt-get install -y nlohmann-json-dev \
                       libpistache-dev \
                       cmake \
                       curl \
                       unzip \
                       libboost-dev \
                       libboost-iostreams-dev \
                       libboost-system-dev \
                       libboost-serialization-dev && \
    apt-get -qq -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# download and instal rdkit
# https://github.com/rdkit/rdkit/blob/master/CMakeLists.txt
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
    make install

# download Torch
RUN curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip

# assertion failure if ran in / directory
COPY src app/src
COPY cmake app/cmake
COPY CMakeLists.txt app/CMakeLists.txt

WORKDIR /app

RUN mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cp PistachePredictor ../PistachePredictor

CMD ["./PistachePredictor"]
