from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,List,Self,Callable,Union,Tuple,TypeVar
import numpy as np
import contextlib
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
        return Variable(data,creator=creator)
    else:
        return data
    
class Variable:
    data : np.ndarray
    grad : np.ndarray
    creator : Function
    name : str
    generation : int

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

    def __init__(self,data,name:str = None,creator:Function=None):
        self.data = init_ndarray(data)
        self.name = name
        if creator is not None:
            self.creator = creator
            self.generation = creator.generation + 1
        else:
            self.generation = 0

    def backward_from_end(self):
        self.grad = 1.0

        if hasattr(self,'creator'):
            all_order_func = self.creator.get_all_parent_variable()
            for func in all_order_func:
                func.calculate_input_grad()
    
    def __repr__(self):
        ret_str = f'data:{self.data}\t'
        ret_str = ret_str + f'generation:{self.generation}\t'
        try:
            ret_str = ret_str + f'grad:{self.grad}\t'
        except:
            pass

        return ret_str

    def graphviz_from_end(self):
        resstr = "digraph {\n"
        if hasattr(self,'creator'):
            all_order_func = self.creator.get_all_parent_variable()
            for func in all_order_func:
                resstr= resstr+func.to_graphviz_line()
        resstr = resstr + "}"
        return resstr


single_out = lambda cl: lambda *inputs:cl()(*inputs)[0]

class Config:
    enable_backprop = True

@contextlib.contextmanager
def config_test(name,value):
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

class Function:
    inputs : List[Variable]
    outputs : List[Variable]
    generation : int

    def __call__(self,*inputs:Any)->List[Variable]:

        self.inputs = apply_each(inputs,lambda data:init_variable(data))
        def check(c,type):
            assert isinstance(c,type) , "each input must be a Variable"
        apply_each(self.inputs,lambda c:check(c,Variable))

        self.inputs = listify(self.inputs)

        output_result = self.forward([input.data for input in self.inputs])
        assert isinstance(output_result,list), "output must be a list"
        
        if Config.enable_backprop:
            self.generation = max([input.generation for input in self.inputs])
            self.outputs = apply_each(output_result,lambda c:init_variable(c,self))
        else:
            self.outputs = apply_each(output_result,lambda c:init_variable(c))
        self.outputs = listify(self.outputs)

        return self.outputs
    #x,y,z
    #*
    #xyz
    #
    #Lyz,Lxz,Lxy
    #x,x,x
    #*
    #x^3
    #
    #3x^2L
    
    #x,y,z
    #+
    #x+y+z
    #
    #L,L,L
    #x,x,x
    #+
    #3x
    #
    #3L
    def calculate_input_grad(self):
        input_datas = [input.data for input in self.inputs]
        output_grads = [output.grad for output in self.outputs]

        input_grad_result = self.backward(input_datas,output_grads)
        assert isinstance(input_grad_result,list), "backward's return type must be a list"
        for input_var, res_grad in zip(self.inputs,input_grad_result):
            try:
                input_var.grad = input_var.grad + res_grad
            except:
                input_var.grad = res_grad

        #We don't need output's grad anymore. 
        def remove_output_grad(output:Variable):
            del output.grad
        apply_each(self.outputs,remove_output_grad)

    def get_all_parent_variable(self):
        l = [self]
        for input in self.inputs:
            try:
                rr = input.creator.get_all_parent_variable()
                l.extend(rr)
            except:
                pass
        l.sort(key=lambda elem:elem.generation,reverse=True)
        return l

    def forward(self,input_datas:List[Any])->List[Any]:
        raise NotImplementedError('You must implement forward')
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        raise NotImplementedError('You must implement backward')

    def to_graphviz_line(self):
        func= f'{id(self)} [label="{self.__class__.__name__}", color=lightblue, style=filled, shape=box] \n'
        for input in self.inputs:
            func = func + f'{id(input)} [label="{input.name}", color=orange, style=filled] \n'
            func = func + f'{id(input)} -> {id(self)} \n'
        for output in self.outputs:
            func = func + f'{id(output)} [label="{output.name}", color=orange, style=filled] \n'
            func = func + f'{id(self)} -> {id(output)} \n'
        
        return func
    
    def __repr__(self):
        ret_str = f'generation:{self.generation}\t'
        return ret_str
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
square = single_out(Square)
class Exp(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [np.exp(input_datas[0]) ]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return [np.exp(input_datas[0]) * output_grads[0]]
exp = single_out(Exp)


class Neg(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [-input_datas[0]]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return apply_each(input_datas,lambda i:output_grads[0])
neg = single_out(Neg)



class Add(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [input_datas[0]+input_datas[1]]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return [output_grads[0],output_grads[0]]
add = single_out(Add)

class Sub(Function):
    def forward(self, input_datas: List[Any]) -> List[Any]:
        return [input_datas[0]-input_datas[1]]
    def backward(self, input_datas: List[Any], output_grads: List[Any]) -> List[Any]:
        return [output_grads[0], -output_grads[0]]
sub = lambda a,b:Sub()(a,b)[0]

class Mul(Function):
    def forward(self,input_datas:List[Any])->List[Any]:
        return [ input_datas[0]*input_datas[1] ]
    def backward(self,input_datas:List[Any],output_grads:List[Any])->List[Any]:
        return [output_grads[0]*input_datas[1],output_grads[0]*input_datas[0]]
mul = single_out(Mul)

class Div(Function):
    def forward(self, input_datas: List[Any]) -> List[Any]:
        return [ input_datas[0]/input_datas[1] ]
    def backward(self, input_datas: List[Any], output_grads: List[Any]) -> List[Any]:
        gy = output_grads[0]
        return [gy/input_datas[1],gy*(-input_datas[0]/(input_datas[1]**2))]
div = lambda a,b:Div()(a,b)[0]

class Pow(Function):
    def forward(self, input_datas: List[Any]) -> List[Any]:
        return [ input_datas[0] ** input_datas[1] ]
    def backward(self, input_datas: List[Any], output_grads: List[Any]) -> List[Any]:
        gy = output_grads[0]
        return [ gy* input_datas[1] * (input_datas[0] ** (input_datas[1] - 1)) ,
                gy * (input_datas[0] ** input_datas[1]) * np.log(input_datas[0]) ]
pow = lambda a,b:Pow()(a,b)[0]

def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__sub__ = sub
    Variable.__rsub__ = lambda self,other:Sub()(other,self)[0]
    Variable.__truediv__ = div
    Variable.__rtruediv__ = lambda self,other:Div()(other,self)[0]
    Variable.__pow__ = pow

def numerical_diff(f:Function,x:Variable,eps:float=1e-4):
    x0=x.data-eps
    x1=x.data+eps
    fx0=f(Variable(x0))
    fx1=f(Variable(x1))
    return (fx1.data-fx0.data)/(2.*eps)