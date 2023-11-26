import os
import subprocess


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys
        sys.exit(1)
