import torch
from dataclasses import dataclass,field
from typing import Any,Dict,Callable,Self

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
            return self._tensor.unsqueeze(0)
        else:
            raise "error"
    
    def __iter__(self):
        if self.axis_sequence == 0:
            return self._tensor.__iter__()
        elif self.axis_sequence <0 :
            return self._tensor.unsqueeze(0).__iter__()
        else:
            raise "error"
    

#train mode
#input,output,... => sequence, parameter => nonsequence
#prediction mode
#input => sequence, parameter=> nonsequence, default=> not used 

@dataclass
class TensorManager:
    tensors : Dict[str,TorchTensorPlus] = field(default_factory=dict)

    def get_all_tensors(self:Self,current_sequence:int,mode:ModeType):
        if mode == ModeType.TRAIN:
            return {tensor_name: self.tensors[tensor_name][current_sequence]  for tensor_name in self.tensors}
        elif mode == ModeType.PREDICT:
            ret={tensor_name: self.tensors[tensor_name][current_sequence]  for tensor_name in self.tensors if self.tensors[tensor_name].ttype != TTPType.DEFAULT}
            nonret={tensor_name: None for tensor_name in self.tensors if self.tensors[tensor_name].ttype == TTPType.DEFAULT}
            ret.update(nonret.items())
            return ret

    def get_length(self,mode:ModeType):
        #set as sequence length of input if prediction.
        if mode == ModeType.TRAIN:
            comparison = [len(self.tensors[tensor_name].tensor) for tensor_name in self.tensors]
            return min(comparison)
        elif mode == ModeType.PREDICT:
            comparison = [len(self.tensors[tensor_name].tensor) for tensor_name in self.tensors if self.tensors[tensor_name].ttype == TTPType.INPUT]
            return comparison[0]
        

    def get_all_params(self):
        return [self.tensors[key].tensor for key in self.tensors if self.tensors[key].ttype == TTPType.PARAMETER]
    


        

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=dict)
    meta_optimizer_epoch : int = 2000
    meta_error_measurement : Any = torch.nn.MSELoss
    meta_activator : Any = None
    
    _all_leaf_tensors : TensorManager = field(init=False,default_factory=TensorManager)
    def __getitem__(self,key):
        return self._all_leaf_tensors.tensors[key]

    def __setitem__(self,key,value):
        self._all_leaf_tensors.tensors[key] = value
    
    assign_process_process : Callable = None

    def train_one_step_by_equation(self,label,prediction_quation):
        loss = self.meta_error_measurement()(label,  prediction_quation)
        optim = self.meta_optimizer(self._all_leaf_tensors.get_all_params(),**self.meta_optimizer_params)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def train(self):
        current_mode = ModeType.TRAIN
        #all terminals
        self._current_activator = self.meta_activator()

        for _ in range(self.meta_optimizer_epoch):
            for sequence_ind in range(self._all_leaf_tensors.get_length(current_mode)):
                _label,_pred = self.assign_process_process(self._all_leaf_tensors.get_all_tensors(sequence_ind,current_mode))
                self.train_one_step_by_equation(_label,_pred)

        return self._all_leaf_tensors.get_all_params()
    
    def predict(self,**kwarg):
        current_mode = ModeType.PREDICT
        for key in kwarg:
            self[key].tensor = kwarg[key]
        
        
        ret = []
        for sequence_ind in range(self._all_leaf_tensors.get_length(current_mode)):
            print(sequence_ind)
            _,_pred = self.assign_process_process(self._all_leaf_tensors.get_all_tensors(sequence_ind,current_mode))
            ret.append(_pred)
        
        return ret