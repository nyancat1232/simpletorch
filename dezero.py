from dataclasses import dataclass,field
from typing import Any,Self
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

@dataclass
class Variable:
    data : Any
    creator : Any = None
    grad : Any = field(init=False)

    def __post_init__(self):
        self.grad = None

@dataclass
class Function:
    memory_input : Any = None

    def __call__(self:Self,input:Variable)->Variable:
        self.input_data = input.data
        self.output_data = self.forward(self.input_data)
        output = Variable(self.output_data,self)

        return output
    
    def forward(self,input_data:Any)->Any:
        raise NotImplementedError('You must implement forward')
    def backward(self, output_grad: Any) -> Any:
        raise NotImplementedError('You must implement forward')
    
class Square(Function):
    def forward(self,input_data:Any)->Any:
        return input_data ** 2
    def backward(self, output_grad: Any) -> Any:
        return (2 * self.input_data ) * output_grad
    
class Exp(Function):
    def forward(self, input_data: Any) -> Any:
        return np.exp(input_data)
    def backward(self, output_grad: Any) -> Any:
        return np.exp(self.input_data ) * output_grad
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)