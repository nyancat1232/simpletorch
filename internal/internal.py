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
class TensorInternalSequencedUnsqeezed:
    name : str
    ttype : TTPType
    _tensor : torch.Tensor = field(repr=False,init=False)
    @property
    def tensor(self):
        return self._tensor
    @tensor.setter
    def tensor(self,tor_tensor : torch.Tensor) ->torch.Tensor:
        self._tensor = tor_tensor
        if self.ttype==TTPType.PARAMETER:
            self._tensor.requires_grad = True
        return self._tensor

@dataclass
class TensorInternalSequenced(TensorInternalSequencedUnsqeezed):
    def unsqueeze_to(self,dim):
        ret = TensorInternalSequencedUnsqeezed(name=self.name,ttype=self.ttype)
        current_dim = self.tensor.dim()
        for _ in range(dim-current_dim):
            ret.tensor = self.tensor.unsqueeze(0)
        try:
            ret.tensor
        except:
            ret.tensor = self.tensor
        return ret

@dataclass
class TensorInternal(TensorInternalSequenced):
    axis_sequence : int = -1
        
    def __getitem__(self,key) ->TensorInternalSequenced:
        if self.axis_sequence == 0:
            ret = TensorInternalSequenced(self.name,self.ttype)
            ret.tensor = self.tensor[key]
            return ret
        elif self.axis_sequence <0 :
            ret = TensorInternalSequenced(self.name,self.ttype)
            ret.tensor = self.tensor
            return ret
        else:
            raise NotImplemented("error")
    

#train mode
#input,output,... => sequence, parameter => nonsequence
#prediction mode
#input => sequence, parameter=> nonsequence, default=> not used 

@dataclass
class TensorsManager:
    _tensors : List[TensorInternal] = field(repr=False,init=False)

    def __post_init__(self):
        self._tensors = []

    def __getitem__(self,sequence_ind) ->Dict[str,TensorInternalSequenced]:
        return [tensor[sequence_ind] for tensor in self._tensors]
    
    def new_tensor(self,name:str,ttype:TTPType,axis_sequence:int,tensor:torch.Tensor):
        current_ttp = TensorInternal(name=name,ttype=ttype,axis_sequence=axis_sequence)
        current_ttp.tensor = tensor
        self._tensors.append(current_ttp) 

    def change_tensor(self,name,tensor:torch.Tensor):
        for current_tensor in self._tensors:
            if current_tensor.name == name:
                current_tensor.tensor = tensor
    
    def get_all_params(self):
        return {tensor.name :tensor.tensor for tensor in self._tensors if tensor.ttype == TTPType.PARAMETER}


def unsqueeze_tensors(tensors:Dict[str,TensorInternalSequenced],max_dim=None):
    if max_dim is None:
        max_dim = max([tensors[key].tensor.dim() for key,_ in enumerate(tensors)])

    return {tensors[key].name : tensors[key].unsqueeze_to(max_dim) for key,_ in enumerate(tensors)},max_dim

        