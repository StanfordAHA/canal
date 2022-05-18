import shutil
import os
import gemstone
import glob


def copy_sv_files(tempdir):
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    path = os.path.abspath(gemstone.__file__)
    for i in range(2):
        path = os.path.dirname(path)
    vector_dir = os.path.join(path, "tests", "common", "rtl", "*.sv")
    sv_files = glob.glob(vector_dir)
    for f in sv_files:
        shutil.copy(f, tempdir)
