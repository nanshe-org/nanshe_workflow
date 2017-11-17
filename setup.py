from __future__ import print_function


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 18, 2015 12:09:31 EDT$"


import os
import shutil
import sys

from setuptools import setup, find_packages

import versioneer


build_requires = []
install_requires = []
tests_requires = []
if len(sys.argv) == 1:
    pass
elif ("--help" in sys.argv) or ("-h" in sys.argv):
    pass
elif sys.argv[1] == "bdist_conda":
    import yaml

    recipe = {}
    with open("nanshe_workflow.recipe/meta.yaml", "r") as recipe_file:
        lines = recipe_file.readlines()
        lines = [_ for _ in lines if "version:" not in _]
        lines = "\n".join(lines)
        recipe = yaml.load(lines)

    recipe_reqs = recipe.get("requirements", {})
    recipe_reqs["build"] = recipe_reqs.get("build", [])
    recipe_reqs["run"] = recipe_reqs.get("run", [])
    recipe_reqs["test"] = recipe.get("test", {}).get("requires", [])

    build_requires = recipe_reqs["build"]
    install_requires = recipe_reqs["run"]
    tests_requires = recipe_reqs["test"]
elif sys.argv[1] == "clean":
    if os.path.exists(".eggs"):
        print("removing '.eggs'")
        shutil.rmtree(".eggs")
    else:
        print("'.eggs' does not exist -- can't clean it")
elif sys.argv[1] == "develop":
    if (len(sys.argv) > 2) and (sys.argv[2] in ["-u", "--uninstall"]):
        if os.path.exists("nanshe_workflow.egg-info"):
            print("removing 'nanshe_workflow.egg-info'")
            shutil.rmtree("nanshe_workflow.egg-info")
        else:
            print("'nanshe_workflow.egg-info' does not exist -- can't clean it")

setup(
    name="nanshe_workflow",
    version=versioneer.get_version(),
    description="Provides an iPython workflow for running nanshe.",
    url="https://github.com/nanshe-org/nanshe_workflow",
    license="Apache 2.0",
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    py_modules=["versioneer"],
    packages=find_packages(exclude=["tests*"]),
    package_data={"nanshe_workflow": ["dask_config.yaml"]},
    cmdclass=versioneer.get_cmdclass(),
    build_requires=build_requires,
    install_requires=install_requires,
    tests_require=tests_requires,
    test_suite="test_nanshe_workflow",
    zip_safe=True
)
