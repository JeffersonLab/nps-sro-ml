# =============================================================================
# Build:
#     docker build -t eicdev/nps-onnx-jana2:latest --build-arg BUILD_THREADS=24 .
#  using buildx for clean cmake error report (but verbose):
#     docker buildx build --tag eicdev/nps-onnx-jana2:latest --progress=plain --push .
#     docker buildx build --build-arg BUILD_THREADS=20 --tag eicdev/nps-onnx-jana2:latest --progress=plain --push .
#  using stop and enter an image on failure
#     BUILDX_EXPERIMENTAL=1 docker buildx debug --invoke bash build --progress=plain .
#  in PShell instead of BUILDX_EXPERIMENTAL=1 use:
#    $env:BUILDX_EXPERIMENTAL="1"

# ONNX Runtime C++ GPU Development Container
# Base: CUDA 12.8 + cuDNN on Ubuntu 24.04
# ONNX Runtime: 1.24.3 (pre-built C++ GPU tarball)
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# OCI Image Labels - most automatically populated by buildx via BUILDX_GIT_LABELS
LABEL org.opencontainers.image.authors="Dmitry Romanov <romanov@jlab.org>" \
      org.opencontainers.image.url="https://github.com/JeffersonLab/nps-sro-ml" \
      org.opencontainers.image.documentation="https://github.com/JeffersonLab/nps-sro-ml/blob/main/README.md" \
      org.opencontainers.image.vendor="Dmitry Romanov" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="EIC Ubuntu Development Environment with ROOT" \
      org.opencontainers.image.description="Ubuntu-based development container with CUDA, ONNX, CERN ROOT, scientific computing libraries, and development tools for EIC software development"

# Legacy labels for compatibility
LABEL maintainer.name="Dmitry Romanov" \
      maintainer.email="romanov@jlab.org"

ARG DEBIAN_FRONTEND=noninteractive
ENV APP_ROOT=/app

ENV DEBIAN_FRONTEND=noninteractive

# ---- Core C++ dev tools + Python (for model generation) --------------------
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl \
    libssl-dev pkg-config zlib1g-dev \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p ${APP_ROOT}

# ---- ONNX Runtime C++ (GPU) -----------------------------------------------
ARG ONNX_VERSION=1.24.3
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz \
    && tar xzf onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz \
    && mv onnxruntime-linux-x64-gpu-${ONNX_VERSION} ${APP_ROOT}/onnxruntime \
    && rm onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz

ENV ONNXRUNTIME_ROOT=${APP_ROOT}/onnxruntime
ENV LD_LIBRARY_PATH=${APP_ROOT}/onnxruntime/lib:${LD_LIBRARY_PATH}

# ---- Python ONNX (for model generation only) --------------------------------
RUN python3 -m venv ${APP_ROOT}/venv \
    && ${APP_ROOT}/venv/bin/pip install --no-cache-dir onnx numpy
ENV PATH="${APP_ROOT}/venv/bin:${PATH}"

ARG CXX_STANDARD=20
ARG BUILD_THREADS=4

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
    CMAKE_PREFIX_PATH="${FMT_INS_PATH}/lib/cmake/fmt"

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
###          CERN ROOT build (C++20, builtin LLVM)                       ###
### ---------------------------------------------------------------------- ###
RUN apt-get update && apt-get install -y \
    libx11-dev libxext-dev libxft-dev libxpm-dev \
    libxmu-dev libgsl-dev libfftw3-dev \
    libsqlite3-dev liblzma-dev \
    libpng-dev libjpeg-dev python3-dev \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENV ROOT_VERSION=6.36.08
ENV ROOT_SRC_PATH=${APP_ROOT}/root-src \
    ROOT_BLD_PATH=${APP_ROOT}/root-build \
    ROOT_INS_PATH=${APP_ROOT}/root

RUN set -ex && \
    ROOT_TAG="v$(echo ${ROOT_VERSION} | tr '.' '-')" && \
    git clone --branch ${ROOT_TAG} --depth 1 https://github.com/root-project/root.git "${ROOT_SRC_PATH}" && \
    mkdir -p "${ROOT_BLD_PATH}" && cd "${ROOT_BLD_PATH}" && \
    cmake \
      -DCMAKE_INSTALL_PREFIX="${ROOT_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
      -Dxrootd=ON \
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
