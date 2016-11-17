#!/usr/bin/env python


import contextlib
import os
import sys
import unittest

import nbconvert
import nbconvert.nbconvertapp


@contextlib.contextmanager
def sys_args(argv):
    sys.argv, argv = argv, sys.argv
    yield
    sys.argv, argv = argv, sys.argv

class TestNansheWorkflow(unittest.TestCase):
    def testRunNansheIPython(self):
        sdir = os.path.abspath(os.path.curdir)
        nb_filenames = [
            "nanshe_ipython.ipynb",
        ]
        for each_nb_filename in nb_filenames:
            argv = (
                "jupyter",
                "nbconvert",
                "--ExecutePreprocessor.kernel_name=python%i" % sys.version_info[0],
                "--to",
                "notebook",
                "--stdout",
                "--execute",
                "%s/nanshe_ipython.ipynb" % sdir
            )
            with sys_args(argv):
                os.environ["TEST_NOTEBOOKS"] = "true"
                nbconvert.nbconvertapp.main()

if __name__ == '__main__':
    unittest.main()
