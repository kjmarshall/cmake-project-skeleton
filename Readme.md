# CMake Project Skeleton
This repo presents a generic and extensible CMake project skeleton to help a user organize more complicated `C++` projects and speed up productivity.  These `C++` projects may depend on multiple internally written libraries, external submodules, or pre-compiled libraries from other GitHub projects.  This CMake skeleton defaults to using or "linking" with header only library implementations (though static/dynamic libraries can be easily added).  Executable projects are stored/organized in the `tests/` directory are linked to these libraries using the CMake build system.  In addition this repo shows an example of how a CUDA project may be complied and linked against `C++` libraries.  It is suggested that this build system be combined with Emacs using the addon packages [Helm](https://github.com/emacs-helm/helm), [Projectile](https://github.com/bbatsov/projectile), and [Malinka](https://github.com/LefterisJP/malinka).  This combination nearly automates project compliation, error detection, and installation.

## Assumptions
1. Compiler must support `C++11` or `C++0x`
2. CMake version &#8805; 2.8 defaults to 3.10, controllable in the top-level directory's CMakeLists.txt file
3. Optional:  Installed modern version of Nvidia's `nvcc` compiler and CUDA API, Nvidia graphics card
## Usage
Change or modify directory and file names to fit one's project.  Run the following commands to create an out of source build,

1. `mkdir build` to create a build directory
2. `cd build` to switch to the build directory
3. `cmake ..` to configure and run cmake
4. `make -j4` to build projects in `tests/`
5. `make install` to copy and install executables in `bin/`

## Directory Structure
Out of source builds are recommended by utilizing a `build/` directory.  The directory structure takes the following form,
```
 bin/
 build/
 cmake-modules/
 external/
 lib/
 source/
   project1/
     CMakeLists.txt
   project2/
     CMakeLists.txt
   project3-CUDA/
     CMakeLists.txt
     cuda-helpers.hpp
     project3-CUDA.cu
   CMakeLists.txt
 tests/
   test-project1/
     CMakeLists.txt
     test-project1.cpp
   test-project2/
     CMakeLists.txt
     test-project2.cpp
   test-project3-CUDA/
     CMakeLists.txt
     test-project3-CUDA.cu
   CMakeLists.txt
 CMakeLists.txt
 Readme.md
```

### Directory Structure Notes
1. `bin/` is where `make install` copies compiled executables
2. `build/` contains all CMake files and build files associated with an out of source build
3. `cmake-modules/` contains custom modules and settings.  If one tries to load a CMake package using standard methods e.g. `find_package( <package> [REQUIRED] )` it is convenient to put project dependent look up scripts in this module directory
4. `external/` is a directory which should be used to hold any external libraries that the project may depend on.  Using Git submodules one may add libraries using `git submodule add -b <external-repo-branch> <external-repo-address> external/<external-repo-name>`
5. `lib/` is where internal libraries are moved after they compiled.  By default, these libraries are assumed to be header only implementations and take advantage of the `INTERFACE` CMake build option.  This can be changed by modifying the CMakeLists.txt files in each `source/project<1,2>` directory and requiring a `STATIC` library build with assumed function definitions/implementations given in subdirectory source `.cpp` files.
6. `tests/` is where we create executable code and link against both internal libraries (compiled in lib/) and external libraries

## Including External Libraries in Projects
It is fairly easy to include external libraries submodules into the current CMake skeleton.  For example, to include [Eigen 3.3](https://github.com/eigenteam/eigen-git-mirror/tree/branches/3.3) one would add the Git submodule,

`git submodule add -b brances/3.3 https://github.com/eigenteam/eigen-git-mirror.git external/eigen-3.3`.

The top-level CMakeLists.txt file would then be modified by adding,
```
## ----- ##
## Eigen ##
## ----- ##
set( EIGEN_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3")
set( EIGEN_INCLUDE_DIRS ${EIGEN_DIR} )
list( APPEND INCLUDES "${EIGEN_INCLUDE_DIRS}" )
```
This approach works for the inclusion of any header only library.  For example, one may also include VCGLib using the same techniques (see the top-level CMakeLists.txt file).  Note that external projects may also be added in similar ways using CMake's `ExternalProject_Add( <name> ... )` build command.
