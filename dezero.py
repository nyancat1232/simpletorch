from dataclasses import dataclass,field
from typing import Any,List,Self
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

def init_ndarray(data)->np.ndarray:
    if not isinstance(data,np.ndarray):
        return np.array(data)
    else:
        return data

class Variable:

    def __init__(self,data,creator:Self=None):
        assert not isinstance(data,Variable)
        assert not isinstance(data,(list,tuple))

        self.data = init_ndarray(data)
        assert isinstance(self.data,np.ndarray)

        if creator is not None:
            self.creator = creator

    def backward(self)->List:
        qu = [self.creator]
        last_vars = []
        try:
            while previous_func := qu.pop():
                previous_input = previous_func.generate_input_grad()
                try:
                    qu.append(previous_input.creator)
                except:
                    last_vars.append(previous_input)
        except:
            return last_vars
    
    def __repr__(self):
        ret_str = f'data:{self.data}\t'
        try:
            ret_str = ret_str + f'grad:{self.grad}\t'
        except:
            pass
        try:
            ret_str = ret_str + f'previous function:{self.creator}\t'
        except:
            pass
        return ret_str

def init_variable(data,creator:Variable=None)->Variable:
    if not isinstance(data,Variable):
        return Variable(data,creator)
    else:
        return data
    

class Function:
    def __call__(self,*inputs:Variable)->Variable:
        if len(inputs)>1:
            self.inputs = [init_variable(input) for input in inputs]
        else:
            self.input = init_variable(inputs[0])

        return self.generate_output()
    
    def generate_output(self)->Variable:
        assert not hasattr(self,'output')

        self.output = Variable(self.forward(),self)
        return self.output
    
    def generate_input_grad(self)->Variable:
        self.input.grad = self.backward()
        return self.input

    def forward(self):
        raise NotImplementedError('You must implement forward')
    def backward(self):
        raise NotImplementedError('You must implement forward')
    
class Square(Function):
    def forward(self):
        return self.input.data ** 2
    def backward(self):
        return (2 * self.input.data ) * self.output.grad
def square(input:Variable)->Variable:
    return Square()(input)
    
class Exp(Function):
    def forward(self):
        return np.exp(self.input.data)
    def backward(self):
        return np.exp(self.input.data ) * self.output.grad
def exp(input:Variable)->Variable:
    return Exp()(input)
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)