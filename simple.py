import torch
from dataclasses import dataclass,field
from typing import Any,Dict,Callable

from enum import Enum


class TTPType(Enum):
    DEFAULT = 0
    INPUT = 1
    PARAMETER = 2

@dataclass
class TorchTensorPlus():
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
    

@dataclass
class AllTensorsToSequence:
    tensors : Dict[str,TorchTensorPlus] = field(default_factory=dict)

    def get_all_tensors(self,key):
        return {tensor_name: self.tensors[tensor_name][key]  for tensor_name in self.tensors}

    def get_length(self):
        for tensor_name in self.tensors:
            return self.tensors[tensor_name].tensor.shape[0]
        #return [len(self.tensors[tensor_name].tensor) for tensor_name in self.tensors]

    def get_all_params(self):
        return [self.tensors[key].tensor for key in self.tensors if self.tensors[key].ttype == TTPType.PARAMETER]
    


        

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_learning_rate : float = 0.015
    meta_optimizer_epoch : int = 2000
    meta_error_measurement : Any = torch.nn.MSELoss
    meta_activator : Any = None
    
    all_leaf_tensors : AllTensorsToSequence = field(default_factory=AllTensorsToSequence)

    assign_leaf_tensors : Callable = None
    assign_process_process : Callable = None

    def train_one_step_by_equation(self,label,prediction_quation):
        loss = self.meta_error_measurement()(label,  prediction_quation)
        optim = self.meta_optimizer(self.all_leaf_tensors.get_all_params(),lr=self.meta_optimizer_learning_rate)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def train(self):
        #all terminals
        self.assign_leaf_tensors(self)
        self._current_activator = self.meta_activator()

        for _ in range(self.meta_optimizer_epoch):
            for sequence_ind in range(self.all_leaf_tensors.get_length()):
                _label,_pred = self.assign_process_process(self.all_leaf_tensors.get_all_tensors(sequence_ind))
                self.train_one_step_by_equation(_label,_pred)

        return self.all_leaf_tensors.get_all_params()
    
    def predict(self,**kwarg):
        for key in kwarg:
            self.all_leaf_tensors.tensors[key].tensor = kwarg[key]
        
        _pred=None
        for sequence_ind in range(self.all_leaf_tensors.get_length()):
            _pred = self.assign_process_process(self.all_leaf_tensors.get_all_tensors(sequence_ind))
        
        return _pred