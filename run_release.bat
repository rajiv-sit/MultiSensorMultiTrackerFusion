@ECHO ON
REM ==============================
REM MultiTracker Build Script - Release Mode
REM ==============================

REM Save current directory
SET BASEDIR=%~dp0
PUSHD %BASEDIR%

REM ------------------------------
REM Remove previous build folder
REM ------------------------------
IF EXIST build (
    RMDIR /Q /S build
)

REM ------------------------------
REM Install Conan dependencies
REM ------------------------------
conan install . ^
    -c tools.system.package_manager:mode=install ^
    -c tools.system.package_manager:sudo=True ^
    --output-folder=build ^
    --build=missing ^
    --settings=build_type=Release

REM ------------------------------
REM Generate Visual Studio project
REM ------------------------------
cd build
cmake .. ^
    -G "Visual Studio 17 2022" ^
    -DCMAKE_TOOLCHAIN_FILE=build/generators/conan_toolchain.cmake ^
    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW ^
    -DCMAKE_BUILD_TYPE=Release

REM ------------------------------
REM Build project
REM ------------------------------
cmake --build . --config Release

REM ------------------------------
REM Run unit tests
REM ------------------------------
ctest --output-on-failure

REM ------------------------------
REM Run executable
REM ------------------------------
Release\multi_tracker.exe

REM Return to original directory
POPD
