# =============================================================================
# ONNX Runtime C++ GPU Development Container
# Base: CUDA 12.8 + cuDNN on Ubuntu 24.04
# ONNX Runtime: 1.24.3 (pre-built C++ GPU tarball)
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- Core C++ dev tools + Python (for model generation) --------------------
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl \
    libssl-dev pkg-config zlib1g-dev \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ---- ONNX Runtime C++ (GPU) -----------------------------------------------
ARG ONNX_VERSION=1.24.3
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz \
    && tar xzf onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz \
    && mv onnxruntime-linux-x64-gpu-${ONNX_VERSION} /opt/onnxruntime \
    && rm onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# ---- Python ONNX (for model generation only) --------------------------------
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir onnx numpy
ENV PATH="/opt/venv/bin:${PATH}"

ARG CXX_STANDARD=20

# Install OS dependencies
RUN apt-get update &&\
    apt-get install -y libfmt-dev libboost-filesystem-dev libboost-all-dev libspdlog-dev libboost-test-dev libxmu-dev libexpat-dev libtbb-dev &&\
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

### ---------------------------------------------------------------------- ###
###          fmt build                                                     ###
### ---------------------------------------------------------------------- ###
ENV FMT_VERSION=12.1.0
ENV FMT_SRC_PATH=${APP_ROOT}/fmt-src \
    FMT_BLD_PATH=${APP_ROOT}/fmt-build \
    FMT_INS_PATH=${APP_ROOT}/fmt

RUN set -ex && \
    git clone --branch ${FMT_VERSION} --depth 1 https://github.com/fmtlib/fmt.git "${FMT_SRC_PATH}" && \
    mkdir -p "${FMT_BLD_PATH}" && cd "${FMT_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${FMT_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
      "${FMT_SRC_PATH}" && \
    cmake --build . --target install -j"${BUILD_THREADS}" && \
    rm -rf "${FMT_BLD_PATH}" "${FMT_SRC_PATH}"

ENV LD_LIBRARY_PATH="${FMT_INS_PATH}/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="${FMT_INS_PATH}/lib/cmake/fmt:${CMAKE_PREFIX_PATH}"

### ---------------------------------------------------------------------- ###
###          Catch2 build                                                  ###
### ---------------------------------------------------------------------- ###
ENV CATCH2_VERSION=v3.13.0
ENV CATCH2_SRC_PATH=${APP_ROOT}/catch2-src \
    CATCH2_BLD_PATH=${APP_ROOT}/catch2-build \
    CATCH2_INS_PATH=${APP_ROOT}/catch2

RUN set -ex && \
    git clone --branch ${CATCH2_VERSION} --depth 1 https://github.com/catchorg/Catch2.git "${CATCH2_SRC_PATH}" && \
    mkdir -p "${CATCH2_BLD_PATH}" && cd "${CATCH2_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${CATCH2_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
      "${CATCH2_SRC_PATH}" && \
    cmake --build . --target install -j"${BUILD_THREADS}" && \
    rm -rf "${CATCH2_BLD_PATH}" "${CATCH2_SRC_PATH}"

ENV LD_LIBRARY_PATH="${CATCH2_INS_PATH}/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="${CATCH2_INS_PATH}/lib/cmake/Catch2:${CMAKE_PREFIX_PATH}"

### ---------------------------------------------------------------------- ###
###          Eigen3 build                                                  ###
### ---------------------------------------------------------------------- ###
ENV EIGEN3_VERSION=5.0.0
ENV EIGEN3_SRC_PATH=${APP_ROOT}/eigen3-src \
    EIGEN3_BLD_PATH=${APP_ROOT}/eigen3-build \
    EIGEN3_INS_PATH=${APP_ROOT}/eigen3

RUN set -ex && \
    git clone --branch ${EIGEN3_VERSION} --depth 1 https://gitlab.com/libeigen/eigen.git "${EIGEN3_SRC_PATH}" && \
    mkdir -p "${EIGEN3_BLD_PATH}" && cd "${EIGEN3_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${EIGEN3_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      "${EIGEN3_SRC_PATH}" && \
    cmake --build . --target install -j"${BUILD_THREADS}" && \
    rm -rf "${EIGEN3_BLD_PATH}" "${EIGEN3_SRC_PATH}"

ENV CMAKE_PREFIX_PATH="${EIGEN3_INS_PATH}/share/eigen3/cmake:${CMAKE_PREFIX_PATH}"

### ---------------------------------------------------------------------- ###
###          HepMC3 build                                                  ###
### ---------------------------------------------------------------------- ###
ENV HEPMC3_VERSION=3.3.1
ENV HEPMC3_SRC_PATH=${APP_ROOT}/hepmc3-src \
    HEPMC3_BLD_PATH=${APP_ROOT}/hepmc3-build \
    HEPMC3_INS_PATH=${APP_ROOT}/hepmc3

RUN set -ex && \
    git clone --branch ${HEPMC3_VERSION} --depth 1 https://gitlab.cern.ch/hepmc/HepMC3.git "${HEPMC3_SRC_PATH}" && \
    mkdir -p "${HEPMC3_BLD_PATH}" && cd "${HEPMC3_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${HEPMC3_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
      -DHEPMC3_ENABLE_PYTHON:BOOL=OFF \
      -DPython_LIBRARIES=/usr/lib/python3.12/config-3.12-x86_64-linux-gnu \
      -DPython_EXECUTABLE=/usr/bin/python3 \
      -DPython_INCLUDE_DIRS=/usr/include/python3.12 \
      "${HEPMC3_SRC_PATH}" && \
    cmake --build . --target install -j"${BUILD_THREADS}" && \
    rm -rf "${HEPMC3_BLD_PATH}" "${HEPMC3_SRC_PATH}"

ENV HEPMC3_DIR=${HEPMC3_INS_PATH} \
    PATH="${HEPMC3_INS_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="${HEPMC3_INS_PATH}/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="${HEPMC3_INS_PATH}/share/HepMC3/cmake:${CMAKE_PREFIX_PATH}"

### ---------------------------------------------------------------------- ###
###          CERN ROOT build                                               ###
### ---------------------------------------------------------------------- ###
RUN apt-get update && apt-get install -y \
    libx11-dev libxext-dev libxft-dev libxpm-dev \
    libpcre2-dev libgsl-dev libfftw3-dev \
    libsqlite3-dev liblzma-dev \
    libpng-dev libjpeg-dev python3-dev \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENV ROOT_VERSION=v6-36-08
ENV ROOT_SRC_PATH=${APP_ROOT}/root-src \
    ROOT_BLD_PATH=${APP_ROOT}/root-build \
    ROOT_INS_PATH=${APP_ROOT}/root

RUN set -ex && \
    git clone --branch ${ROOT_VERSION} --depth 1 https://github.com/root-project/root.git "${ROOT_SRC_PATH}" && \
    mkdir -p "${ROOT_BLD_PATH}" && cd "${ROOT_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${ROOT_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
      -Dminimal=ON \
      -Dpyroot=OFF \
      "${ROOT_SRC_PATH}" && \
    cmake --build . --target install -j"${BUILD_THREADS}" && \
    rm -rf "${ROOT_BLD_PATH}" "${ROOT_SRC_PATH}"

ENV ROOTSYS=${ROOT_INS_PATH} \
    PATH="${ROOT_INS_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="${ROOT_INS_PATH}/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="${ROOT_INS_PATH}/cmake:${CMAKE_PREFIX_PATH}"

### ---------------------------------------------------------------------- ###
###          JANA2 build                                                   ###
### ---------------------------------------------------------------------- ###
ARG VERSION_JANA2=master
ENV JANA2_SRC_PATH=${APP_ROOT}/jana2-src \
    JANA2_BLD_PATH=${APP_ROOT}/jana2-build \
    JANA2_INS_PATH=${APP_ROOT}/jana2

RUN set -ex && \
    git clone  --branch ${VERSION_JANA2} --depth 1 https://github.com/JeffersonLab/JANA2.git "${JANA2_SRC_PATH}" && \
    mkdir -p "${JANA2_BLD_PATH}" && cd "${JANA2_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${JANA2_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
      -DUSE_PODIO=Off \
      -DUSE_ROOT=On \
      "${JANA2_SRC_PATH}" && \
    cmake --build . --target install -j"${BUILD_THREADS}" && \
    rm -rf "${JANA2_BLD_PATH}" "${JANA2_SRC_PATH}"

ENV JANA_HOME=${JANA2_INS_PATH} \
    JANA_PLUGIN_PATH="${JANA2_INS_PATH}/plugins" \
    PATH="${JANA2_INS_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="${JANA2_INS_PATH}/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="${JANA2_INS_PATH}/lib/cmake/JANA:${CMAKE_PREFIX_PATH}"

SHELL ["/bin/bash", "-c"]
