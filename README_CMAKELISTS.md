# Instructions


## Prerequisites

```bash
sudo apt update --yes
sudo apt install --yes python3-matplotlib
```

## Build

```bash
cd soillib
rm -rf build/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) -- VERBOSE=1
cmake --install build --prefix ~/.local

```

## Run

```bash
cd soillib/example
python3 erosion_basic.py
```