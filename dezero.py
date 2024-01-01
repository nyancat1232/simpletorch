from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,List,Self,Callable,Union,Tuple,TypeVar
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

apply_each_T : TypeVar('apply_each_T')

def apply_each(value,
               func_apply):
    '''
    apply each elements with func_apply
    
    Examples
    --------
    v = 3
    vr = apply_each(v,lambda v:v**2)
    print(vr)
    >>>> (9, False)

    v = [2]
    vr = apply_each(v,lambda v:v**2)
    print(vr)
    >>>> (4, False)

    v = [4,5,6]
    vr = apply_each(v,lambda v:v**2)
    print(vr)
    >>>> ([16, 25, 36], True)
    '''
    try:
        if len(value)>1:
            return [func_apply(v) for v in value], True
        else:
            return func_apply(value[0]), False
    except TypeError:
        return func_apply(value),False
    
def init_ndarray(data)->np.ndarray:
    if not isinstance(data,np.ndarray):
        return np.array(data)
    else:
        return data

def init_variable(data,creator:Variable=None)->Variable:
    if not isinstance(data,Variable):
        return Variable(data,creator)
    else:
        return data
    
class Variable:
    data : np.ndarray
    grad : np.ndarray
    creator : Function
    generation : int

    def __init__(self,data,creator:Function=None):
        self.data = init_ndarray(data)
        if creator is not None:
            self.creator = creator
            self.generation = creator.generation + 1
        else:
            self.generation = 0

    def backward(self)->List:
        pass
    
    def __repr__(self):
        ret_str = f'data:{self.data}\t'
        ret_str = ret_str + f'generation:{self.generation}\t'
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
    inputs : List[Variable]
    outputs : List[Variable]
    generation : int

    def __call__(self,*inputs:Any):
        pass
    
    def generate_output(self):
        pass
    
    def calculate_input_grad(self):
        pass

    def forward(self):
        raise NotImplementedError('You must implement forward')
    def backward(self):
        raise NotImplementedError('You must implement forward')
    
    def __repr__(self):
        ret_str = f'inputs:{self.inputs}\t'
        ret_str = ret_str + f'generation:{self.generation}\t'
        try:
            ret_str = ret_str + f'outputs:{self.outputs}\t'
        except:
            ret_str = ret_str + f'no outputs\t'
        return ret_str

    
class Square(Function):
    def forward(self):
        pass
    def backward(self):
        pass
def square(input)->Variable:
    return Square()(input)
    
class Exp(Function):
    def forward(self):
        pass
    def backward(self):
        pass
def exp(input)->Variable:
    return Exp()(input)

class Add(Function):
    def forward(self):
        pass
    def backward(self):
        pass
def add(*inputs)->Variable:
    return Add()(*inputs)
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)