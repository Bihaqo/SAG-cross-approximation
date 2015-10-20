#This script will build the main subpackages  
from distutils.util import get_platform 
from numpy.distutils.misc_util import Configuration, get_info
import sys
import os
from os.path import join as pjoin, dirname
import zipfile
import warnings
import shutil
from distutils.cmd import Command
from distutils.command.clean import clean
from distutils.version import LooseVersion
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError
from numpy.distutils.misc_util import appendpath
from numpy.distutils import log
from numpy.distutils.misc_util import is_string

def have_good_cython():
    try:
        from Cython.Compiler.Version import version
    except ImportError:
        return False
    return LooseVersion(version) >= LooseVersion('0.16')

def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method

    Uses Cython instead of Pyrex.
    '''
    good_cython = have_good_cython()
    if self.inplace or not good_cython:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    sources_changed = newer_group(depends, target_file, 'newer') 
    if self.force or sources_changed:
        if good_cython:
            # add distribution (package-wide) include directories, in order
            # to pick up needed .pxd files for cython compilation
            incl_dirs = extension.include_dirs[:]
            dist_incl_dirs = self.distribution.include_dirs
            if not dist_incl_dirs is None:
                incl_dirs += dist_incl_dirs
            import Cython.Compiler.Main
            log.info("cythonc:> %s" % (target_file))
            self.mkpath(target_dir)
            options = Cython.Compiler.Main.CompilationOptions(
                defaults=Cython.Compiler.Main.default_options,
                include_path=incl_dirs,
                output_file=target_file)
            cython_result = Cython.Compiler.Main.compile(source,
                                                       options=options)
            if cython_result.num_errors != 0:
                raise DistutilsError("%d errors while compiling "
                                     "%r with Cython"
                                     % (cython_result.num_errors, source))
        elif sources_changed and os.path.isfile(target_file):
            raise DistutilsError("Cython >=%s required for compiling %r"
                                 " because sources (%s) have changed" %
                                 (CYTHON_MIN_VERSION, source, ','.join(depends)))
        else:
            raise DistutilsError("Cython >=%s required for compiling %r"
                                 " but not available" %
                                 (CYTHON_MIN_VERSION, source))
    return target_file
from numpy.distutils.command.build_src import build_src
build_src.generate_a_pyrex_source = generate_a_pyrex_source

def configuration(parent_package='',top_path=None):
    #sys.argv.extend ( ['config_fc'] )
    config = Configuration('rect_maxvol', parent_package, top_path) 
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False,     
    )
    
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    inc_dir = ['build/temp%s' % plat_specifier]
    
    config.add_include_dirs(inc_dir)
    
    config.add_subpackage('cython_boost')
    return config
    


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
