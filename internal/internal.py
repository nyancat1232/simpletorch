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
class TorchTensorPlusInternalSequencedUnsqeezed:
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
class TorchTensorPlusInternalSequenced(TorchTensorPlusInternalSequencedUnsqeezed):
    def unsqueeze_to(self,dim):
        ret = TorchTensorPlusInternalSequencedUnsqeezed(name=self.name,ttype=self.ttype)
        current_dim = self.tensor.dim()
        for _ in range(dim-current_dim):
            ret.tensor = self.tensor.unsqueeze(0)
        try:
            ret.tensor
        except:
            ret.tensor = self.tensor
        return ret

@dataclass
class TorchTensorPlusInternal(TorchTensorPlusInternalSequenced):
    axis_sequence : int = -1
        
    def __getitem__(self,key) ->TorchTensorPlusInternalSequenced:
        if self.axis_sequence == 0:
            ret = TorchTensorPlusInternalSequenced(self.name,self.ttype)
            ret.tensor = self.tensor[key]
            return ret
        elif self.axis_sequence <0 :
            ret = TorchTensorPlusInternalSequenced(self.name,self.ttype)
            ret.tensor = self.tensor
            return ret
        else:
            raise NotImplemented("error")
    

#train mode
#input,output,... => sequence, parameter => nonsequence
#prediction mode
#input => sequence, parameter=> nonsequence, default=> not used 

@dataclass
class TensorsSquence:
    _tensors : List[TorchTensorPlusInternal] = field(repr=False,init=False)

    def __post_init__(self):
        self._tensor_name = []
        self._tensors = []

    def __getitem__(self,sequence_ind) ->Dict[str,TorchTensorPlusInternalSequenced]:
        return {self._tensor_name[index] : self._tensors[index][sequence_ind] for index,_ in enumerate(self._tensor_name)}
    
    def new_tensor(self,tensorplus:TorchTensorPlusInternal,tensor:torch.Tensor):
        current_ttp = tensorplus
        current_ttp.tensor = tensor
        self._tensors.append(current_ttp) 
        self._tensor_name.append(current_ttp.name)

    def change_tensor(self,name,tensor:torch.Tensor):
        for current_name_index,current_name in enumerate(self._tensor_name):
            if current_name == name:
                self._tensors[current_name_index].tensor = tensor
    
    def get_all_params(self):
        return {name:tensor.tensor for name,tensor in zip(self._tensor_name,self._tensors) if tensor.ttype == TTPType.PARAMETER}


def unsqueeze_tensors(tensors:Dict[str,TorchTensorPlusInternalSequenced],max_dim=None):
    if max_dim is None:
        max_dim = max([tensors[key].tensor.dim() for key in tensors])

    return {key : tensors[key].unsqueeze_to(max_dim) for key in tensors},max_dim

        