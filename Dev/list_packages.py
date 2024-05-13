# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:38:37 2024

@author: Richard
"""

import pkg_resources
import prettytable
import pandas as pd


def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'

def print_packages_and_licenses():
    t = pd.DataFrame(columns=['Package', 'License'])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        ti=pd.DataFrame({'Package':pkg,'License':get_pkg_license(pkg)},index=[0])
        
        t = t._append(ti, ignore_index = True)
    return t
    # print(t)



df_licenses=print_packages_and_licenses()
    
#%%
import os.path, pkgutil
import testpkg

folder=r'C:\Users\Richard\OneDrive - Vysoké učení technické v Brně\Dokumenty\Github\CrackPy\src\crackest'
pkgpath = os.path.dirname(folder.__file__)
print([name for _, name, _ in pkgutil.iter_modules([pkgpath])])
#%%

import imp
import os
MODULE_EXTENSIONS = ('.py', '.pyc', '.pyo')

def package_contents(package_name):
    file, pathname, description = imp.find_module(package_name)
    if file:
        raise ImportError('Not a package: %r', package_name)
    # Use a set because some may be both source and compiled.
    return set([os.path.splitext(module)[0]
        for module in os.listdir(pathname)
        if module.endswith(MODULE_EXTENSIONS)])

package_contents('numpy')
