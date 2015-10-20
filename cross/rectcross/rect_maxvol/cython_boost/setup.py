# -*- coding: utf-8 -*-
#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
from os.path import exists, getmtime
import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('cython_boost', parent_package, top_path)
    olddir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    if not exists('maxvol.pyx') or getmtime('gen_maxvol.py') > getmtime('maxvol.pyx'):
        execfile('gen_maxvol.py')
    os.chdir(olddir)
    config.add_extension('maxvol', sources='maxvol.pyx', extra_compile_args=['-undefined,dynamic_lookup'])
    return config

if __name__ == '__main__':
    print 'This is the wrong setup.py to run'
