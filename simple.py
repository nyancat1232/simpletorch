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

#default tensor's axis_sequence => 0 if train, -1 if predict
#default tensor's tensor => ? if train, ? if predict

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
    tensors_prediction : Dict[str,TorchTensorPlus] = field(default_factory=dict)
    tensors_label : TorchTensorPlus = None

    def __getitem__(self,pos:Union[Tuple[int,str],int]):
        try:
            sequence_index , tensor_name = pos
            return self.tensors_prediction[tensor_name][sequence_index]
        except:
            return {key : self.tensors_prediction[key][pos] for key in self.tensors_prediction}
    

    def get_length(self):
        #set as sequence length of input if prediction.
        for tensor_name in self.tensors_prediction:
            if self.tensors_prediction[tensor_name].ttype == TTPType.INPUT:
                return len(self.tensors_prediction[tensor_name].tensor)
    
    def get_all_params(self):
        return {key:self.tensors_prediction[key].tensor for key in self.tensors_prediction if self.tensors_prediction[key].ttype == TTPType.PARAMETER}



        

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
        return self._all_leaf_tensors.tensors_prediction[key]

    def __setitem__(self,key,value):
        self._all_leaf_tensors.tensors_prediction[key] = value
    
    assign_process_prediction : Callable = None

    def train_one_step_by_equation(self,label,prediction_quation):
        loss = self.meta_error_measurement()(label,  prediction_quation)
        optim = self.meta_optimizer(self._all_leaf_tensors.get_all_params().values(),**self.meta_optimizer_params)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def train(self):
        #filter current sequence => unify dimensions => cals


        for _ in range(self.meta_optimizer_epoch):
            for pred_tensors,lab_tensors in zip(self._all_leaf_tensors,self._all_leaf_tensors.tensors_label):
                pred = self.assign_process_prediction(pred_tensors,self.meta_activator)
                self.train_one_step_by_equation(lab_tensors,pred)

        return self._all_leaf_tensors.get_all_params()
    
    def predict(self,**kwarg):

        for key in kwarg:
            self[key].tensor = kwarg[key]
        
        
        ret = []
        for pred_tensors in self._all_leaf_tensors:
            pred = self.assign_process_prediction(pred_tensors,self.meta_activator)
            ret.append(pred)
        
        return ret