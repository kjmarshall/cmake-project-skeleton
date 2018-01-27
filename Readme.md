# CMake Project Skeleton
This project presents a generic and extensible project skeleton that allows one to use CMake to build more complex `C++` projects.  These projects may depend on multiple internally written libraries and/or external submodules or pre-compiled libraries from other GitHub projects.  This CMake skeleton defaults to creating header only library implementations that are appropriately linked to executable projects in the `tests/` directory.  Executable projects are created in the `tests/` directory by linking with appropriate internal and external libraries.  A user may choose to rename directories and executable files.  Renaming of files and/or directories also requires that one change both directory and file names in each directory's CMakeLists.txt file.

# Assumptions
Compiler must support C++11 or C++0x
CMake version >= 2.8 defaults to 3.10, controllable in the top directory's CMakeLists.txt file

# Usage
Change or modify directory and file names to fit one's project.  Run the following commands to create an out of source build,
1. `mkdir build` to create a build directory
2. `cd build` to switch to the build directory
3. `cmake ..` to configure and run cmake
4. `make -j4` to build projects in `tests/`
5. `make install` to copy and install executables in `bin/`

# Directory Structure
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
1. `bin/` is where `make install` copies compiled executables
2. `build/` contains all CMake files and build files associated with an out of source build
3. `cmake-modules/` contains custom modules and settings.  If one tries to load a CMake package using standard methods e.g. `find_package( <package> [REQUIRED] )` it is convenient to put project dependent look up scripts in this module directory
4. `external/` is a directory which should be used to hold any external libraries that the project may depend on.  Using Git submodules one may add libraries using `git submodule add -b <external-repo-branch> <external-repo-address> external/<external-repo-name>`
5. `lib/` is where internal libraries are moved after they compiled.  By default, these libraries are assumed to be header only implementations and take advantage of the `INTERFACE` CMake build option.  This can be changed by modifying the CMakeLists.txt files in each `source/project<1,2>` directory and requiring a `STATIC` library build with assumed function definitions/implementations given in subdirectory source `.cpp` files.
6. `tests/` is where we create executable code and link against both internal libraries (compiled in lib/) and external libraries

# Including External Libraries in Projects
It is fairly easy to include external libraries submodules into the current CMake skeleton.  For example, to include Eigen 3.3 one would add the Git submodule,
`git submodule add -b brances/3.3 https://github.com/eigenteam/eigen-git-mirror.git external/eigen-3.3`

The top-level CMakeLists.txt file would then be modified by adding,
```
## ----- ##
## Eigen ##
## ----- ##
set( EIGEN_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3")
set( EIGEN_INCLUDE_DIRS ${EIGEN_DIR} )
list( APPEND INCLUDES "${EIGEN_INCLUDE_DIRS}" )
```
Such an approach works for any header only library.  One may also include VCGLib using the same techniques (see the top-level CMakeLists.txt file).
