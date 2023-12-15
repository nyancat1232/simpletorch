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
    sequence_axis : int = -1
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
    
    
    
    
        

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_learning_rate : float = 0.015
    meta_optimizer_epoch : int = 2000
    meta_error_measurement : Any = torch.nn.MSELoss
    meta_activator : Any = None
    
    all_leaf_tensors : Dict[str,TorchTensorPlus] = field(default_factory=dict)

    assign_leaf_tensors : Callable = None
    assign_process_process : Callable = None

    def get_all_params(self):
        return [self.all_leaf_tensors[key].tensor for key in self.all_leaf_tensors if self.all_leaf_tensors[key].ttype == TTPType.PARAMETER]
    
    def train_one_step_by_equation(self,label,prediction_quation):
        loss = self.meta_error_measurement()(label,  prediction_quation)
        optim = self.meta_optimizer(self.get_all_params(),lr=self.meta_optimizer_learning_rate)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    def train(self):
        #all terminals
        self.assign_leaf_tensors(self)
        
        for _ in range(self.meta_optimizer_epoch):
            #process
            self.assign_process_process(self)

            #train
            self.train_one_step_by_equation(self._label,self._pred)
        return self.get_all_params()
    
    def predict(self,**kwarg):
        for key in kwarg:
            self.all_leaf_tensors[key].tensor = kwarg[key]
        self.assign_process_process(self)
        return self._pred