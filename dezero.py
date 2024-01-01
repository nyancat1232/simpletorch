from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,List,Self,Callable,Union,Tuple,TypeVar
import numpy as np
#This source references 'ゼロから作る Deep Learning' by 斎藤 康毅

apply_each_T : TypeVar('apply_each_T')

def apply_each(value,
               func_apply,return_multiple=False):
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
    def ret(data,b):
        if return_multiple:
            return data, b
        else:
            return data

    try:
        if len(value)>1:
            return ret([func_apply(v) for v in value], True)
        else:
            return ret(func_apply(value[0]), False)
    except TypeError:
        return ret(func_apply(value),False)
    
def listify(check_value):
    if not isinstance(check_value,list):
        return [check_value]
    else:
        return check_value

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

        return ret_str


class Function:
    inputs : List[Variable]
    outputs : List[Variable]
    generation : int

    def __call__(self,*inputs:Any)->List[Variable]:

        self.inputs = apply_each(inputs,lambda data:init_variable(data))
        def check(c,type):
            assert isinstance(c,type)
        apply_each(self.inputs,lambda c:check(c,Variable))

        self.inputs = listify(self.inputs)

        self.generation = max([input.generation for input in self.inputs])
        output_result = self.forward([input.data for input in self.inputs])
        assert isinstance(output_result,list)
        self.outputs = apply_each(output_result,lambda c:init_variable(c,self))
        self.outputs = listify(self.outputs)
        return self.outputs
    
    def calculate_input_grad(self):
        pass

    def forward(self,input_datas:List[Any])->List[Any]:
        raise NotImplementedError('You must implement forward')
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        raise NotImplementedError('You must implement backward')
    
    def __repr__(self):
        ret_str = f'inputs:{self.inputs}\t'
        ret_str = ret_str + f'generation:{self.generation}\t'
        try:
            ret_str = ret_str + f'outputs:{self.outputs}\t'
        except:
            ret_str = ret_str + f'no outputs\t'
        return ret_str

    
class Square(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [input_datas[0] ** 2]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return [2 * input_datas[0] * output_grads[0]]
def square(*input)->Variable:
    return Square()(*input)
    
class Exp(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [np.exp(input_datas[0]) ]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return [np.exp(input_datas[0]) * output_grads[0]]
def exp(*input)->Variable:
    return Exp()(*input)

class Add(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [sum(input_datas)]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return apply_each(input_datas,lambda i:output_grads[0])
def add(*inputs)->Variable:
    return Add()(*inputs)
    
def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)