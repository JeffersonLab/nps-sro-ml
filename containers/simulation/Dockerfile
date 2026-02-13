# NPS2023 Simulation Docker Image
# ================================
# Builds the full NPS simulation stack on top of eicdev/eic-evio-pre
# which provides ROOT, PODIO, EDM4hep, JANA2, and other EIC dependencies.
#
# Build:
#     docker build -t eicdev/nps-sim:latest --build-arg BUILD_THREADS=24 .
#  using buildx for clean cmake error report (but verbose):
#     docker buildx build --tag eicdev/nps-sim:latest --progress=plain --push .
#     docker buildx build --build-arg BUILD_THREADS=20 --tag eicdev/nps-sim:latest --progress=plain --push .
#  using stop and enter an image on failure
#     BUILDX_EXPERIMENTAL=1 docker buildx debug --invoke bash build --progress=plain .
#  in PShell instead of BUILDX_EXPERIMENTAL=1 use:
#    $env:BUILDX_EXPERIMENTAL="1"
#
# Run:
#   docker run --rm -it --init eicdev/nps-sim:latest
#
# ======== APPTAINERS / SINGULARITY =============
# To pull image from dockerhub
#    apptainer pull nps-sim.sif docker://eicdev/nps-sim:latest
#
# To run interactively
#    apptainer shell nps-sim.sif
#

FROM eicdev/eic-evio-pre:latest

# ---------------------------------------------------------------------------#
#  Build-time arguments & environment                                        #
# ---------------------------------------------------------------------------#
ARG BUILD_THREADS=8

ENV APP_ROOT=/opt
WORKDIR ${APP_ROOT}

# Additional OS dependencies for Geant4 and NPS packages
RUN apt-get update && \
    apt-get install -y \
        libxerces-c3-dev \
        libexpat-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libx11-dev \
        libxt-dev \
        libxmu-dev \
        libxrender-dev \
        libxpm-dev \
        libxft-dev \
        libxi-dev \
        qt6-base-dev \
        libqt6opengl6-dev \
        tcsh \
        gfortran \
        default-libmysqlclient-dev \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*


### ---------------------------------------------------------------------- ###
###          Geant4 build                                                  ###
### ---------------------------------------------------------------------- ###
ARG VERSION_GEANT4=v11.4.0
ENV GEANT4_SRC_PATH=${APP_ROOT}/geant4-src \
    GEANT4_BLD_PATH=${APP_ROOT}/geant4-build \
    GEANT4_INS_PATH=${APP_ROOT}/geant4

RUN set -ex && \
    git clone --branch ${VERSION_GEANT4} --depth 1 https://github.com/Geant4/geant4.git "${GEANT4_SRC_PATH}" && \
    cmake \
      -S "${GEANT4_SRC_PATH}" \
      -B "${GEANT4_BLD_PATH}" \
      -Wno-dev \
      -DCMAKE_INSTALL_PREFIX="${GEANT4_INS_PATH}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_USE_GDML=ON \
      -DGEANT4_USE_OPENGL_X11=ON \
      -DGEANT4_USE_RAYTRACER_X11=ON \
      -DGEANT4_USE_QT=ON \
      -DGEANT4_USE_SYSTEM_EXPAT=ON \
      -DGEANT4_BUILD_MULTITHREADED=ON \
      -DGEANT4_BUILD_TLS_MODEL=global-dynamic && \
    cmake --build "${GEANT4_BLD_PATH}" --target install -j"${BUILD_THREADS}" && \
    rm -rf "${GEANT4_BLD_PATH}" "${GEANT4_SRC_PATH}"

ENV GEANT4_DIR=${GEANT4_INS_PATH} \
    PATH="${GEANT4_INS_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="${GEANT4_INS_PATH}/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="${GEANT4_INS_PATH}/lib/cmake/Geant4:${CMAKE_PREFIX_PATH}"

# Source the Geant4 data environment (sets G4...DATA variables)
# We bake these into the image by extracting them from geant4.sh
RUN bash -c 'source ${GEANT4_INS_PATH}/bin/geant4.sh && env' | \
    grep '^G4' | sed 's/^/ENV /' > /tmp/g4env.sh || true
# Alternative: use a login snippet
RUN echo "source ${GEANT4_INS_PATH}/bin/geant4.sh" >> /etc/bash.bashrc


### ---------------------------------------------------------------------- ###
###          DVCS Event Generator (dvcs_gen)                               ###
### ---------------------------------------------------------------------- ###
ENV DVCS_GEN_DIR=${APP_ROOT}/dvcs_gen

RUN set -ex && \
    git clone --depth 1 https://github.com/hhuang-hep/dvcs_gen.git "${DVCS_GEN_DIR}" && \
    cd "${DVCS_GEN_DIR}" && \
    make lib

ENV LD_LIBRARY_PATH="${DVCS_GEN_DIR}:${LD_LIBRARY_PATH}"


### ---------------------------------------------------------------------- ###
###          NPS Reconstruction Software (NPS_SOFT)                        ###
### ---------------------------------------------------------------------- ###
ENV NPS_SOFT_DIR=${APP_ROOT}/NPS_SOFT \
    DVCS_SOFT=${APP_ROOT}/NPS_SOFT

RUN set -ex && \
    git clone --depth 1 https://github.com/hhuang-hep/NPS_SOFT.git "${NPS_SOFT_DIR}" && \
    # Hall A Analyzer headers are needed for compilation (THaEvent.h, THaRun.h, etc.)
    git clone --depth 1 --branch Release-1522 \
        https://github.com/JeffersonLab/analyzer.git "${NPS_SOFT_DIR}/v0.65" && \
    cd "${NPS_SOFT_DIR}" && \
    # Fix hardcoded absolute path to THaEvent.h (now found via -I include path)
    sed -i 's|#include "/work/hallc/nps/hosan/analyzer-1.5.22/src/THaEvent.h"|#include "THaEvent.h"|' TDVCSGlobal.h && \
    # Symlink analyzer headers into working dir so rootcint can also find them
    ln -sf v0.65/src/THaEvent.h . && \
    ln -sf v0.65/src/THaRun.h . && \
    # Add missing TMath.h include in THCalo.cxx
    sed -i '1i #include <TMath.h>' THCalo.cxx && \
    make biglib

ENV LD_LIBRARY_PATH="${NPS_SOFT_DIR}:${LD_LIBRARY_PATH}"


### ---------------------------------------------------------------------- ###
###          NPS-Geant4 Package (HallC_NPS)                               ###
### ---------------------------------------------------------------------- ###
ENV HALLC_NPS_DIR=${APP_ROOT}/HallC_NPS \
    HALLC_NPS_BLD=${APP_ROOT}/HallC_NPS/DVCS_evt_gen/build

COPY RFunction.C /tmp/RFunction.C

RUN set -ex && \
    git clone --depth 1 https://github.com/hhuang-hep/HallC_NPS.git "${HALLC_NPS_DIR}" && \
    # Install HMS acceptance R-function (from JLab ifarm)
    mkdir -p "${HALLC_NPS_DIR}/RFunction" && \
    cp /tmp/RFunction.C "${HALLC_NPS_DIR}/RFunction/RFunction.C" && \
    # Fix hardcoded JLab path in CMakeLists.txt to use local copy
    sed -i 's|/work/hallc/nps/hosan/RFunction|'"${HALLC_NPS_DIR}"'/RFunction|' \
        "${HALLC_NPS_DIR}/DVCS_evt_gen/DVCS/CMakeLists.txt" && \
    # Add transitive deps of libDVCS.so (ROOT graphics/histogram + mysql)
    echo 'target_link_libraries(DVCS Hist Graf Gpad mysqlclient)' \
        >> "${HALLC_NPS_DIR}/DVCS_evt_gen/DVCS/CMakeLists.txt" && \
    mkdir -p "${HALLC_NPS_BLD}" && \
    bash -c 'source ${GEANT4_INS_PATH}/bin/geant4.sh && \
    cmake \
      -S "${HALLC_NPS_DIR}/DVCS_evt_gen/DVCS" \
      -B "${HALLC_NPS_BLD}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
    cmake --build "${HALLC_NPS_BLD}" -j"${BUILD_THREADS}"'


### ---------------------------------------------------------------------- ###
###          HMS MC-single-arm Package                                     ###
### ---------------------------------------------------------------------- ###
ENV HMS_DIR=${APP_ROOT}/mc-single-arm

# ROOTSYS should already be set by the parent image;
# if not, the line below provides a fallback
ENV ROOTSYS=${ROOTSYS:-/opt/root}

RUN set -ex && \
    git clone --branch NPS --depth 1 https://github.com/YaopengZhang/mc-single-arm.git "${HMS_DIR}" && \
    cd "${HMS_DIR}/src" && \
    make ROOTDIR="${ROOTSYS}" && \
    # Skip util/ntuple (requires CERNLIB, legacy ntuple format — not needed)
    cd "${HMS_DIR}/util/root_tree/" && \
    make ROOTDIR="${ROOTSYS}"


### ---------------------------------------------------------------------- ###
###          NPS2023 Simulation Scripts                                    ###
### ---------------------------------------------------------------------- ###
ENV NPS_SIM_DIR=${APP_ROOT}/NPS2023_Simulation

RUN set -ex && \
    git clone --depth 1 https://github.com/hhuang-hep/NPS2023_Simulation.git "${NPS_SIM_DIR}"

# Generate a source_simu.sh that points to our installed paths
RUN cat > "${NPS_SIM_DIR}/source_simu.sh" <<'SIMEOF'
#!/bin/bash
export folder_g4=/opt/HallC_NPS/DVCS_evt_gen/build
export folder_hms=/opt/mc-single-arm
export folder_NPS_SOFT=/opt/NPS_SOFT
export folder_DVCS_EVENT_GEN=/opt/dvcs_gen
SIMEOF

# Run the setup script to create directories and symlinks
RUN cd "${NPS_SIM_DIR}" && \
    chmod +x simu_setup.sh NPS_HMS_simu.sh MakeRunMacro.sh && \
    bash -c 'source source_simu.sh && ./simu_setup.sh' || \
    echo "WARN: simu_setup.sh may need manual adjustments for Docker paths"


### ---------------------------------------------------------------------- ###
###          Final environment & entrypoint                                ###
### ---------------------------------------------------------------------- ###
WORKDIR ${NPS_SIM_DIR}

# Ensure Geant4 data vars are available in every shell
SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash", "-l"]