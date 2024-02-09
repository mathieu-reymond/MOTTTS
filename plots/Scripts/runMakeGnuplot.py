#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import sys
import os
import errno
import shutil
import subprocess
import re

def mkdirMinusP(dirName):
    """ This function creates the folder specified, even if nested. """
    try:
        os.makedirs(dirName)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dirName):
            pass
        else:
            raise

plotTmpDir = "/tmp/plotDir"
mkdirMinusP(plotTmpDir);

dir_path = os.path.dirname(os.path.realpath(__file__))
execall = [dir_path + "/makeGnuplotScript.sh"]

# Drop this script's name
del sys.argv[0]

# Massage directories
cwd = os.getcwd()
print(cwd)
for v in sys.argv:
    if v.startswith("./"):
        v = os.path.join(cwd, v[2:])
    execall.append(v)

# Create plot script
with open(os.path.join(plotTmpDir, "plot"), "w+") as plotFile:
    subprocess.call(execall, stdout=plotFile)

with open(os.path.join(plotTmpDir, "plot"), "r") as plotFile:
    print(plotFile.read())
# Plot
subprocess.call(["gnuplot","-p","plot"], cwd=plotTmpDir)

shutil.rmtree(plotTmpDir)
