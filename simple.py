import torch
import torch.nn as nn
from dataclasses import dataclass,field
from typing import Any,Dict,Callable,Self,Tuple,Union

from enum import Enum


class TTPType(Enum):
    DEFAULT = 0
    INPUT = 1
    PARAMETER = 2

class ModeType(Enum):
    TRAIN = 0
    PREDICT = 1

@dataclass
class TorchTensorPlus():
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
class SequenceTensorManager:
    tensors : Dict[str,TorchTensorPlus] = field(default_factory=dict)

    def __getitem__(self,pos:Union[Tuple[int,str],int]):
        try:
            sequence_index , tensor_name = pos
            return self.tensors[tensor_name][sequence_index]
        except:
            return {key : self.tensors[key][pos] for key in self.tensors}

    def get_all_params(self):
        return {key:self.tensors[key].tensor for key in self.tensors if self.tensors[key].ttype == TTPType.PARAMETER}
    
    def get_length(self,mode:ModeType):
        #set as sequence length of input if prediction.
        if mode == ModeType.TRAIN:
            comparison = [len(self.tensors[tensor_name].tensor) for tensor_name in self.tensors if self.tensors[tensor_name].axis_sequence >= 0]
            return min(comparison)
        elif mode == ModeType.PREDICT:
            comparison = [len(self.tensors[tensor_name].tensor) for tensor_name in self.tensors if self.tensors[tensor_name].ttype == TTPType.INPUT]
            return comparison[0]
        
    def get_current_tensors(self:Self,current_sequence:int,mode:ModeType):
        _current_tensors = {tensor_name: self.tensors[tensor_name][current_sequence] for tensor_name in self.tensors}
        if mode == ModeType.PREDICT:
            for tensor_name in self.tensors:
                if self.tensors[tensor_name].ttype == TTPType.DEFAULT:
                    _current_tensors[tensor_name] = TorchTensorPlus()
            

        return _current_tensors
    
    def get_current_tensors_unsqueezed(self:Self,current_sequence:int,mode:ModeType):
        def to_dim(tensor:torch.Tensor,dim:int):
            dim_diff = dim-tensor.dim()
            new_tensor = tensor
            for _ in range(dim_diff):
                new_tensor=new_tensor.unsqueeze(0)
            return new_tensor
        
        current_tensors = self.get_current_tensors(current_sequence,mode)
        max_dim = max([current_tensors[tensor_name].dim() for tensor_name in current_tensors])
        return {tensor_name:to_dim(current_tensors[tensor_name],max_dim) for tensor_name in current_tensors}




        

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=lambda:{'lr':0.015})
    meta_optimizer_epoch : int = 2000
    meta_error_measurement : Any = torch.nn.MSELoss
    meta_activator : Any = nn.ReLU
    
    _all_leaf_tensors : SequenceTensorManager = field(init=False,default_factory=SequenceTensorManager)
    def __getitem__(self,key):
        return self._all_leaf_tensors.tensors[key]

    def __setitem__(self,key,value):
        self._all_leaf_tensors.tensors[key] = value
    
    assign_process_process : Callable = None

    def train_one_step_by_equation(self,label,prediction_quation):
        loss = self.meta_error_measurement()(label,  prediction_quation)
        optim = self.meta_optimizer(self._all_leaf_tensors.get_all_params().values(),**self.meta_optimizer_params)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def train(self):
        #filter current sequence => unify dimensions => cals
        current_mode = ModeType.TRAIN
        #all terminals
        self._current_activator = self.meta_activator()

        for _ in range(self.meta_optimizer_epoch):
            for sequence_ind in range(self._all_leaf_tensors.get_length(current_mode)):
                _label,_pred = self.assign_process_process(self._all_leaf_tensors.get_current_tensors_unsqueezed(sequence_ind,current_mode),self._current_activator)
                self.train_one_step_by_equation(_label,_pred)

        return self._all_leaf_tensors.get_all_params()
    
    def predict(self,**kwarg):
        current_mode = ModeType.PREDICT
        for key in kwarg:
            self[key].tensor = kwarg[key]
        
        
        ret = []
        for sequence_ind in range(self._all_leaf_tensors.get_length(current_mode)):
            _,_pred = self.assign_process_process(self._all_leaf_tensors.get_current_tensors_unsqueezed(sequence_ind,current_mode),self._current_activator)
            ret.append(_pred)
        
        return ret