import os
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake, cmake_layout
from conan.tools.build import check_min_cppstd
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy

class MultiSensorMultiTrackerFusion(ConanFile):
    name = "msmfusion"
    version = "1.0"
    author = "Rajiv Sithiravel"

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "optimized": [1, 2, 3]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "optimized": 1
    }

    exports_sources = "CMakeLists.txt", "source/*", "include/*"

    # ---------------------------
    # Configurations
    # ---------------------------
    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    # ---------------------------
    # Validation
    # ---------------------------
    def validate(self):
        # Supported OS
        if self.settings.os not in ["Windows", "Linux"]:
            raise ConanInvalidConfiguration(
                "This package is only compatible with Windows or Linux"
            )

        # Compiler-specific C++ standard checks
        if self.settings.compiler == "msvc":
            check_min_cppstd(self, "17")
        elif self.settings.compiler == "gcc":
            check_min_cppstd(self, "gnu23")
        elif self.settings.compiler == "intel-cc":
            check_min_cppstd(self, "gnu23")
            self.settings.compiler.mode = "dpcpp"

    # ---------------------------
    # Dependencies
    # ---------------------------
    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("glfw/3.4")
        self.requires("glew/2.2.0")
        self.requires("imgui/cci.20230105+1.89.2.docking")
        self.requires("opengl/system")
        self.requires("gtest/1.15.0")

    def build_requirements(self):
        self.tool_requires("cmake/3.28.1")

    # ---------------------------
    # Generate
    # ---------------------------
    def generate(self):
        # Copy ImGui bindings
        copy(
            self,
            "*glfw*",
            src=os.path.join(self.dependencies["imgui"].package_folder, "res", "Bindings"),
            dst=os.path.join(self.source_folder, "Bindings")
        )
        copy(
            self,
            "*opengl3*",
            src=os.path.join(self.dependencies["imgui"].package_folder, "res", "Bindings"),
            dst=os.path.join(self.source_folder, "Bindings")
        )

        # Generate CMake toolchain and dependencies
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    # ---------------------------
    # Layout
    # ---------------------------
    def layout(self):
        cmake_layout(self)

    # ---------------------------
    # Build
    # ---------------------------
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    # ---------------------------
    # Package
    # ---------------------------
    def package(self):
        cmake = CMake(self)
        cmake.install()






        
    