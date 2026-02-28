# CI Base Image

This directory contains a CI-ready Docker image definition for local and CI usage.

## Contents

The image is based on `ubuntu:20.04` and preinstalls the dependencies used by the pipeline:

- Build tooling: `build-essential`, `pkg-config`, `ccache`, `git`, `curl`, `zip`, `unzip`, `tar`
- Python tooling: `python3-pip`, `cmake` (pip), `pre-commit` (pip)
- Docs tooling: `doxygen`, `graphviz`
- LLVM toolchain: LLVM/Clang 20 (`clang-20`, `lld-20`, `llvm-20`, `llvm-20-tools`, `clang-format-20`, `clang-tidy-20`)
- Rust toolchain: `rustup` + Rust `1.93.0` with `rustfmt` and `clippy`

`PATH` is set so `cargo`, `llvm-profdata`, and `llvm-cov` are available directly.

## Build

```bash
docker build -t basalt-headers-ci:ubuntu20 -f .docker/Dockerfile .
```

## Build And Push Multi-Arch

The helper script builds and pushes `linux/amd64` and `linux/arm64` images to:

- `vladyslavusenko/basalt-ci:ubuntu20`

Before running it, log in to Docker Hub:

```bash
docker login
```

Then run:

```bash
.docker/build_and_push.sh
```

Optional overrides:

```bash
IMAGE_NAME=vladyslavusenko/basalt-ci IMAGE_TAG=ubuntu20 PLATFORMS=linux/amd64,linux/arm64 .docker/build_and_push.sh
```

## Quick Check

```bash
docker run --rm basalt-headers-ci:ubuntu20 bash -lc 'clang-20 --version && llvm-profdata --version && rustc --version && cargo --version'
```
