# Cross-build and emulation image for the Cortex-M firmware under firmware/.
#
# Everything is pinned, so a cycle count produced in this image on one machine
# is the cycle count produced in it on any other:
#
#   Renode               1.16.1 (the base image)
#   Arm GNU Toolchain    14.2.Rel1, checked against Arm's published SHA-256
#   TensorFlow Lite Micro  commit f59d987 (28 August 2026), CMSIS-NN kernels
#
# Build:  docker build -f docker/firmware.Dockerfile -t semg-firmware .
# Run:    docker run --rm -v "$PWD:/work" -w /work semg-firmware bash firmware/scripts/build_and_run.sh

FROM antmicro/renode:1.16.1

ARG ARM_GNU_VERSION=14.2.rel1
ARG ARM_GNU_SHA256=62a63b981fe391a9cbad7ef51b17e49aeaa3e7b0d029b36ca1e9c3b2a9b78823
ARG TFLM_COMMIT=f59d98795086b522d7d53f683a0df3ee704afc18

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates cmake curl git make ninja-build patch python3 \
        python3-numpy python3-pil unzip wget xxd xz-utils \
 && rm -rf /var/lib/apt/lists/*

# The compiler a Cortex-M team would ship with, straight from Arm.
RUN curl -fsSL -o /tmp/arm-gnu.tar.xz \
        "https://developer.arm.com/-/media/Files/downloads/gnu/${ARM_GNU_VERSION}/binrel/arm-gnu-toolchain-${ARM_GNU_VERSION}-x86_64-arm-none-eabi.tar.xz" \
 && echo "${ARM_GNU_SHA256}  /tmp/arm-gnu.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/arm-gnu \
 && tar -xJf /tmp/arm-gnu.tar.xz -C /opt/arm-gnu --strip-components=1 \
 && rm /tmp/arm-gnu.tar.xz
ENV PATH=/opt/arm-gnu/bin:$PATH

# TensorFlow Lite Micro, exported as a plain source tree with the CMSIS-NN
# optimised kernels selected for a Cortex-M4 with FPU. The exporter is TFLM's
# own; CMake then compiles the tree like any other library. The toolchain root
# is passed so the exporter does not fetch a second copy of GCC.
RUN git init -q /tmp/tflm \
 && git -C /tmp/tflm remote add origin https://github.com/tensorflow/tflite-micro.git \
 && git -C /tmp/tflm fetch -q --depth 1 origin "${TFLM_COMMIT}" \
 && git -C /tmp/tflm checkout -q FETCH_HEAD \
 && cd /tmp/tflm \
 && python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
        --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_TOOLCHAIN_ROOT=/opt/arm-gnu/bin/" \
        /opt/tflm \
 && echo "${TFLM_COMMIT}" > /opt/tflm/TFLM_COMMIT \
 && rm -rf /tmp/tflm

ENV TFLM_TREE=/opt/tflm
