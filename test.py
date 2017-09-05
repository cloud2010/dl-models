# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Min'

import sys
import snn.model as snn_m


def test():
    args = sys.argv
    if len(args) == 1:
        print('Hello, world!')
        snn_m.hello_model()
    elif len(args) == 2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')


if __name__ == '__main__':
    test()
