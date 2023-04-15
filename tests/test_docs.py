#!/usr/bin/env python

import doctest
import .cosine_warmup
import unittest


def tests():
    suite = unittest.TestSuite()
    modules = [
        cosine_warmup.scheduler
    ]

    for m in modules:
        suite.addTests(doctest.DocTestSuite(m))

    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    tests()
