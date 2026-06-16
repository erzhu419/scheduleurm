# Salus Source-Build Feasibility Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `source_build_ready` | false |
| `valid_fullstack_substitute_ready` | false |
| `direct_fullstack_salus_superiority_ready` | false |

## Tools

| Tool | Available | Version | Path |
|---|---:|---|---|
| `cmake` | true | `cmake version 3.22.1` | `/usr/bin/cmake` |
| `docker` | false | `` | `` |
| `g++` | true | `g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0` | `/usr/bin/g++` |
| `gcc` | true | `gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0` | `/usr/bin/gcc` |
| `kubectl` | false | `` | `` |
| `nvidia-smi` | true | `NVIDIA-SMI version  : 610.43.02` | `/usr/lib/wsl/lib/nvidia-smi` |
| `pkg-config` | true | `0.29.2` | `/usr/bin/pkg-config` |
| `protoc` | false | `` | `` |

## TensorFlow-Salus Contract

| Field | Value |
|---|---|
| `TensorFlow_DIR` | `` |
| `has_bazel_tensorflow` | `false` |
| `has_libtensorflow_framework` | `false` |
| `has_libtensorflow_kernels` | `false` |
| `local_tensorflow_salus_repo.exists` | `false` |

## Blockers

| Blocker |
|---|
| protoc is not available in the current local environment |
| docker is not available in the current local environment |
| kubectl is not available in the current local environment |
| TensorFlow-Salus source repository is not present under reference/repos/tensorflow-salus |
| TensorFlow_DIR does not point to a TensorFlow-Salus bazel source tree |
| TensorFlow-Salus libtensorflow_kernels.so artifact is absent |
| Salus requires Protobuf 3.4.0 exactly; the current environment has no verified matching protoc |
| Salus requires Boost 1.66 exactly; no verified matching local Boost tree is present |
| upstream production image is pinned to CUDA 9.1/cuDNN7, so source-build substitution still needs old-runtime compatibility evidence |

## Scope

Strict feasibility certificate for replacing the official Salus image with a local source build.  A source build is not a valid full-stack Salus baseline unless it includes TensorFlow-Salus and the pinned native dependency contract required by upstream Salus.
