import configparser
import os
import subprocess
import sys
from nmrutil import *

if not len(sys.argv)!=1:
    print("You need to give a project name as parameter!")
    sys.exit(0)

project=sys.argv[1]
cp = readprops(project)
datapath=cp.get('datadir')

checkprojectdir(datapath, project, cp)

print("Simulating spectra for your compounds...")

