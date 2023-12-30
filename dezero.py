from dataclasses import dataclass,field
from typing import Any,List,Self
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

def init_as_ndarray(data:Any)->np.ndarray:
    if not isinstance(data,np.ndarray):
        print(f'not ndarray {type(data)}')
        return np.array(data)
    else:
        return data

class Variable:

    def __init__(self,data:Any,creator:Self=None):
        self.data = init_as_ndarray(data)
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
                    print('last variable')
                    last_vars.append(previous_input)
        except:
            print('end')
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
    

class Function:
    def __call__(self,input:Variable):
        self.input = input
        return self.generate_output()
    
    def generate_output(self)->Variable:
        try:
            self.output.data = self.forward()
        except:
            print('generate new Variable')
            self.output = Variable(self.forward(),self)
        return self.output
    
    def generate_input_grad(self)->Variable:
        self.input.grad = self.backward()
        return self.input

    def forward(self)->Any:
        raise NotImplementedError('You must implement forward')
    def backward(self) -> Any:
        raise NotImplementedError('You must implement forward')
    
class Square(Function):
    def forward(self)->Any:
        return self.input.data ** 2
    def backward(self) -> Any:
        return (2 * self.input.data ) * self.output.grad
def square(input:Variable)->Variable:
    return Square()(input)
    
class Exp(Function):
    def forward(self)->Any:
        return np.exp(self.input.data)
    def backward(self) -> Any:
        return np.exp(self.input.data ) * self.output.grad
def exp(input:Variable)->Variable:
    return Exp()(input)
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)