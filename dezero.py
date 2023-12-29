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

    def backward(self):
        previous_func = self.creator
        if previous_func is not None:
            output = previous_func.output #if optimize, then set to 'self'
            input = previous_func.input
            
            input.grad = previous_func.backward(output.grad) 
            input.backward()

@dataclass
class Function:

    def __call__(self:Self,input:Variable)->Variable:
        self.input = input
        output_data = self.forward(self.input.data)
        self.output = Variable(output_data,self)

        return self.output
    
    def forward(self,input_data:Any)->Any:
        raise NotImplementedError('You must implement forward')
    def backward(self, output_grad: Any) -> Any:
        raise NotImplementedError('You must implement forward')
    
class Square(Function):
    def forward(self,input_data:Any)->Any:
        return input_data ** 2
    def backward(self, output_grad: Any) -> Any:
        return (2 * self.input.data ) * output_grad
    
class Exp(Function):
    def forward(self, input_data: Any) -> Any:
        return np.exp(input_data)
    def backward(self, output_grad: Any) -> Any:
        return np.exp(self.input.data ) * output_grad
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)