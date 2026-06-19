"""
Modernized build for ROS 2 Jazzy (Python 3.12, Cython 3, numpy 2, setuptools).
  CPU :  python setup.py build_ext --inplace
  GPU :  WITH_CUDA=ON [CUDAHOME=/path/to/cuda] python setup.py build_ext --inplace
"""
import os
import platform
from os.path import join as pjoin
import numpy
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


def check_for_flag(flag_str, truemsg=None, falsemsg=None):
    enabled = os.environ.get(flag_str, "").lower() == "on"
    if enabled and truemsg:
        print(truemsg)
    elif not enabled and falsemsg:
        print(falsemsg + "\n   $ " + flag_str + "=ON python setup.py build_ext --inplace")
    return enabled


def locate_cuda():
    home = os.environ.get("CUDAHOME") or os.environ.get("CUDA_HOME")
    if not home:
        for c in ("/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-11"):
            if os.path.isdir(c):
                home = c; break
    if not home:
        nvcc = next((pjoin(p, "nvcc") for p in os.environ.get("PATH", "").split(os.pathsep)
                     if os.path.exists(pjoin(p, "nvcc"))), None)
        if nvcc:
            home = os.path.dirname(os.path.dirname(nvcc))
    if not home:
        raise EnvironmentError("CUDA not found; set CUDAHOME")
    lib = pjoin(home, "lib64") if os.path.isdir(pjoin(home, "lib64")) else pjoin(home, "lib")
    cfg = {"home": home, "nvcc": pjoin(home, "bin", "nvcc"),
           "include": pjoin(home, "include"), "lib64": lib}
    for k, v in cfg.items():
        if not os.path.exists(v):
            raise EnvironmentError("CUDA %s not found at %s" % (k, v))
    return cfg


use_cuda = check_for_flag("WITH_CUDA", "Compiling WITH CUDA support",
                          "Compiling WITHOUT CUDA support.")
trace = check_for_flag("TRACE", "Compiling with Bresenham trace", None)

if platform.system().lower() == "darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", platform.mac_ver()[0])
    os.environ["CC"] = "c++"

gcc_flags = ["-w", "-std=c++17", "-O3", "-ffast-math", "-fno-math-errno", "-march=native"]
include_dirs = ["../", numpy.get_include()]
sources = ["RangeLibc.pyx", "../vendor/lodepng/lodepng.cpp"]
if trace:
    gcc_flags.append("-D_MAKE_TRACE_MAP=1")

# ---- CPU build ----------------------------------------------------------
if not use_cuda:
    ext = Extension("range_libc", sources, include_dirs=include_dirs,
                    extra_compile_args=gcc_flags, extra_link_args=["-std=c++17"],
                    language="c++")
    setup(name="range_libc", version="0.2", author="Corey Walsh",
          ext_modules=cythonize([ext], language_level=3,
                                compiler_directives={"boundscheck": False, "wraparound": False}))

# ---- CUDA build (rmgpu) -------------------------------------------------
else:
    CUDA = locate_cuda()
    defs = ["-DUSE_CUDA=1", "-DCHUNK_SIZE=262144", "-DNUM_THREADS=256"]
    nvcc_flags = ["-O3", "--use_fast_math", "-Xcompiler", "-fPIC", "-std=c++17",
                  "--expt-relaxed-constexpr"] + defs
    sources.append("kernels.cu")
    include_dirs.append(CUDA["include"])

    def customize_for_nvcc(self):
        self.src_extensions.append(".cu")
        default_so = self.compiler_so
        sup = self._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == ".cu":
                self.set_executable("compiler_so", CUDA["nvcc"])
                postargs = extra_postargs["nvcc"]
            else:
                postargs = extra_postargs["gcc"]
            sup(obj, src, ext, cc_args, postargs, pp_opts)
            self.compiler_so = default_so
        self._compile = _compile

    class cuda_build_ext(build_ext):
        def build_extensions(self):
            customize_for_nvcc(self.compiler)
            build_ext.build_extensions(self)

    ext = Extension("range_libc", sources, include_dirs=include_dirs,
                    extra_compile_args={"gcc": gcc_flags + defs, "nvcc": nvcc_flags},
                    extra_link_args=["-std=c++17"], library_dirs=[CUDA["lib64"]],
                    libraries=["cudart"], runtime_library_dirs=[CUDA["lib64"]], language="c++")
    setup(name="range_libc", version="0.2", author="Corey Walsh",
          ext_modules=cythonize([ext], language_level=3,
                                compiler_directives={"boundscheck": False, "wraparound": False}),
          cmdclass={"build_ext": cuda_build_ext})
