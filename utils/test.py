import sys
import inspect
import numpy as np

def test1():
    print('test1')
    
def test2():
    print('test2')
    
def test3():
    print('test3')

# and name.startwith('test') and name != 'testall'
testfunctions = [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isfunction(obj)]
testfunctions[0]()