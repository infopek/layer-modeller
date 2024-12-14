# Layer Modeller
University thesis project. This layer modeller is capable of processing a TIFF and a JSON of boreholes to produce a 3D rendered image of the lithostratigraphy. It's also able to generate export files of the layers for external programs to use.

## Prerequisites
- CMake (Version >= 3.16)
- C++ Compiler supporting C++20 standard

### Installing CMake
If you don't have CMake installed yet, follow these steps to install it:

#### Windows
1. Visit the [CMake download page](https://cmake.org/download/) and download the latest version of CMake for Windows.
2. Run the downloaded installer and follow the installation instructions.
3. Once installed, you can verify the installation by opening a command prompt and typing `cmake --version`.

## Getting Started
1. Clone this repository to your local machine:
```bash
git clone --recursive https://github.com/infopek/layer-modeller.git
```

2. Navigate to the cloned directory:
```bash
cd ./layer-modeller/
```

3. Generate build files:
```bash
cmake --preset=x64-<debug/release>
```

4. Build the project:

### Debug
```bash
cd ./build-x64-debug/
cmake --build . -j8
```

### Release
```bash
cd ./build-x64-release/
cmake --build . --config Release -j8
```

This project is licensed under the MIT License - see the LICENSE file for details.
