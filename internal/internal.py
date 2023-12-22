import torch
import torch.nn as nn
from dataclasses import dataclass,field
from typing import Any,Dict,Callable,Self,Tuple,Union,List

from enum import Enum

class MetaTensorType(Enum):
    DEFAULT = 0
    INPUT = 1
    PARAMETER = 2

class TensorDataType(Enum):
    NUMERICAL = 1
    CATEGORICAL = 2

class ProcessMode(Enum):
    ASSIGN = 1
    PROCESS = 2

@dataclass
class TensorInternalSequencedUnsqeezed:
    name : str
    ttype : MetaTensorType = MetaTensorType.DEFAULT
    _tensor : torch.Tensor = field(repr=False,init=False)
    @property
    def tensor(self):
        return self._tensor
    @tensor.setter
    def tensor(self,tor_tensor : torch.Tensor) ->torch.Tensor:
        self._tensor = tor_tensor
        if self.ttype==MetaTensorType.PARAMETER:
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
class TensorsManagerSequenced:
    tensors : List[TensorInternal] = field(default_factory=list)

    def new_tensor(self,tensor:torch.Tensor,axis_sequence:int,ttype:MetaTensorType=MetaTensorType.DEFAULT,name:str=None):
        current_ttp = TensorInternal(name=name,ttype=ttype,axis_sequence=axis_sequence)
        current_ttp.tensor = tensor
        self.tensors.append(current_ttp) 

    def change_tensor(self,name,tensor:torch.Tensor):
        for current_tensor in self.tensors:
            if current_tensor.name == name:
                current_tensor.tensor = tensor

    def get_tensor(self,name) -> TensorInternal:
        for current_tensor in self.tensors:
            if current_tensor.name == name:
                return current_tensor

    def get_all_params(self):
        return {tensor.name :tensor.tensor for tensor in self.tensors if tensor.ttype == MetaTensorType.PARAMETER}

    def get_max_dim(self):
        return max([tensor.tensor.dim() for tensor in self.tensors])
    def unsqueeze_tensors(self,max_dim=None):
        if max_dim is None:
            max_dim = self.get_max_dim()
        return TensorsManagerSequenced([tensor.unsqueeze_to(max_dim) for tensor in self.tensors]),max_dim

    def get_min_sequence_length(self,type):
        return min([len(tensor.tensor) for tensor in self.tensors if tensor.ttype == type])
        


@dataclass
class TensorsManager(TensorsManagerSequenced):

    def __getitem__(self,sequence_ind) ->TensorsManagerSequenced:
        ret = TensorsManagerSequenced([tensor[sequence_ind] for tensor in self.tensors])
        return ret
    
    def is_empty(self):
        if len(self.tensors) > 0:
            return False
        else:
            return True
    
    
