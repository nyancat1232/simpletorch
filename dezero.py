from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,List,Self,Callable,Union
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

def apply_each(value:Any,func_apply:Callable[[Union[Any,List[Any]]],Union[Any,List[Any]]]):
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

class Variable:
    data : np.ndarray
    grad : np.ndarray
    creator : Function
    generation : int

    def __init__(self,data,creator:Function=None):
        assert not isinstance(data,Variable)
        assert not isinstance(data,(list,tuple))

        self.data = init_ndarray(data)
        assert isinstance(self.data,np.ndarray)
        self.grad = np.ones_like(self.data)
        
        if creator is not None:
            self.creator = creator

    def backward(self)->List:
        qu = [self.creator]
        last_vars = []
        try:
            while previous_func := qu.pop():
                previous_inputs = previous_func.calculate_input_grad()
                def apply_queue(previous_inputs:Union[Variable,List[Variable]]):
                    try:
                        qu.append(previous_inputs.creator)
                    except:
                        last_vars.append(previous_inputs)
                apply_each(previous_inputs,apply_queue)
        except IndexError as ie:
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
    input : Variable
    inputs : List[Variable]
    output : Variable
    outputs : List[Variable]
    generation : int

    def __call__(self,*inputs:Any):
        appl_val, is_multiple = apply_each(inputs,init_variable)

        if is_multiple:
            self.inputs = appl_val
        else:
            self.input = appl_val

        return self.generate_output()
    
    def generate_output(self):
        assert not hasattr(self,'output')
        assert not hasattr(self,'outputs')

        forward_result = self.forward()
        ff = lambda fr:init_variable(fr,self)
        appl_val, is_multiple = apply_each(forward_result,ff)

        if is_multiple:
            self.outputs = appl_val
        else:
            self.output = appl_val
        return appl_val
    
    def calculate_input_grad(self)->Variable:
        backward_result=self.backward()
        if isinstance(backward_result,list):
            for input,grad in zip(self.inputs,backward_result):
                input.grad = grad
        else:
            self.input.grad = backward_result
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
def square(input)->Variable:
    return Square()(input)
    
class Exp(Function):
    def forward(self):
        return np.exp(self.input.data)
    def backward(self):
        return np.exp(self.input.data ) * self.output.grad
def exp(input)->Variable:
    return Exp()(input)

class Add(Function):
    def forward(self):
        return sum([input.data for input in self.inputs])
    def backward(self):
        return [self.output.grad for input in self.inputs]
def add(*inputs)->Variable:
    return Add()(*inputs)
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)