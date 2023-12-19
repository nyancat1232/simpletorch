import torch
import torch.nn as nn
from dataclasses import dataclass,field
from typing import Any,Dict,Callable,Self,Tuple,Union,List

from enum import Enum

class TTPType(Enum):
    DEFAULT = 0
    INPUT = 1
    PARAMETER = 2


class ProcessMode(Enum):
    ASSIGN = 1
    PROCESS = 2

@dataclass
class TorchTensorPlusInternal():
    '''
    (description)
    ## Parameters:
    ttype : TTPType
    tensor type.
    axis_sequence : int
    if -1, then this is not a sequence.
    else, then this is a sequence of axis_sequence
    ttype : TTPType
    tensor type.
    '''
    ttype : TTPType
    axis_sequence : int = -1
    _tensor : torch.Tensor = field(repr=False,init=False)
    @property
    def tensor(self):
        return self._tensor
    @tensor.setter
    def tensor(self,tor_tensor : torch.Tensor):
        self._tensor = tor_tensor
        if self.ttype==TTPType.PARAMETER:
            self._tensor.requires_grad = True
        return self._tensor
        
    def __getitem__(self,key):
        if self.axis_sequence == 0:
            return self._tensor[key]
        elif self.axis_sequence <0 :
            return self._tensor
        else:
            raise "error"
    

#train mode
#input,output,... => sequence, parameter => nonsequence
#prediction mode
#input => sequence, parameter=> nonsequence, default=> not used 

@dataclass
class TensorsSquence:
    _tensor_name : List[str] = field(repr=False,init=False)
    _tensors : List[TorchTensorPlusInternal] = field(repr=False,init=False)

    def __post_init__(self):
        self._tensor_name = []
        self._tensors = []

    def __getitem__(self,sequence_ind):
        return {self._tensor_name[index] : self._tensors[index][sequence_ind] for index,_ in enumerate(self._tensor_name)}
    
    def new_tensor(self,name,tensorplus:TorchTensorPlusInternal,tensor:torch.Tensor):
        self._tensor_name.append(name)
        current_ttp = tensorplus
        current_ttp.tensor = tensor
        self._tensors.append(current_ttp) 

    def change_tensor(self,name,tensor:torch.Tensor):
        for current_name_index,current_name in enumerate(self._tensor_name):
            if current_name == name:
                self._tensors[current_name_index].tensor = tensor
    
    def get_all_params(self):
        return {name:tensor.tensor for name,tensor in zip(self._tensor_name,self._tensors) if tensor.ttype == TTPType.PARAMETER}


def unsqueeze_to(tensor:torch.Tensor,dim):
    new_tensor = tensor
    current_dim = new_tensor.dim()
    for _ in range(dim-current_dim):
        new_tensor = new_tensor.unsqueeze(0)
    return new_tensor

def unsqueeze_tensors(tensors:Dict[str,torch.Tensor],max_dim=None):
    if max_dim is None:
        max_dim = max([tensors[key].dim() for key in tensors])

    return {key : unsqueeze_to(tensors[key],max_dim) for key in tensors},max_dim

        