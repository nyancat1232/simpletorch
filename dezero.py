from dataclasses import dataclass,field
from typing import Any,Self
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

class Variable:

    def __init__(self,data,creator=None):
        self.data = data
        self.creator = creator

    def backward(self):
        previous_func = self.creator
        if previous_func is not None:
            previous_input = previous_func.generate_input_grad()
            return previous_input.backward()
        else:
            return self
    
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
    

@dataclass
class Function:
    input : Variable
    output : Variable = field(init=False,repr=False)
    
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
    
class Exp(Function):
    def forward(self)->Any:
        return np.exp(self.input.data)
    def backward(self) -> Any:
        return np.exp(self.input.data ) * self.output.grad
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)