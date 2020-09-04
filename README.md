# pose-diff

## Installation

Following steps are only tested on [Google Colab](https://colab.research.google.com/). If you prefer to install on other platform, please follow the [installation instruction of Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md).

### Step 1
clone the repo and install [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
```
! git clone https://github.com/WangShaoRu/pose-diff.git
! cd pose-diff && git submodule update --init --recursive --remote
# cmake
! wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
! tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local
# dependency
! cd pose-diff/openpose && chmod -R a+x ./scripts && ./scripts/ubuntu/install_deps.sh
# make
! cd pose-diff/openpose && mkdir build && cd build && cmake -DBUILD_PYTHON=ON -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
# build
! cd pose-diff/openpose/build/ && make -j`nproc`
```

### Step 2
Run pose diff. Be sure you are in `pose-diff` directory.
```
python pose_diff.py [--ref ${REF_PATH} [--test ${TEST_PATH} [--show]]]
```
Optional arguments:
- `REF_PATH`: Path to the reference image.
- `TEST_PATH`: Path to the test image.
- `show`: If specified, results will be displayed in a new window.

### A notebook demo run on [google colab](https://colab.research.google.com/) can be found in demo/pose_diff.ipynb