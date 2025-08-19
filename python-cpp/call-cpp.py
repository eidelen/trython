
# ctypes_test.py
import ctypes
import pathlib

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "cmake-build-debug/libpython_cpp.so"
    print(libname)
    c_lib = ctypes.CDLL(libname)
    # Set function prototype for correct conversions
    c_lib.cmult.argtypes = [ctypes.c_int, ctypes.c_float]
    c_lib.cmult.restype = ctypes.c_float

    x, y = 6, 2.3
    answer = c_lib.cmult(x, y)
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
