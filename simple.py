import torch
import torch.nn as nn
from dataclasses import dataclass,field
from typing import Any,Dict,Callable,Self,Tuple,Union,List

from enum import Enum


class TTPType(Enum):
    DEFAULT = 0
    INPUT = 1
    PARAMETER = 2


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

    def __getitem__(self,pos:Union[Tuple[int,str],int])->Union[torch.Tensor,Dict[str,torch.Tensor]]:
        try:
            sequence_index , tensor_name = pos
            return self.tensors_prediction[tensor_name][sequence_index]
        except:
            return {key : self.tensors_prediction[key][pos] for key in self.tensors_prediction}
    
    
    def get_all_params(self):
        return {key:self.tensors_prediction[key].tensor for key in self.tensors_prediction if self.tensors_prediction[key].ttype == TTPType.PARAMETER}

def unsqueeze_to(tensor:torch.Tensor,dim):
    new_tensor = tensor
    current_dim = new_tensor.dim()
    for _ in range(dim-current_dim):
        new_tensor = new_tensor.unsqueeze(0)
    return new_tensor

def unsqueeze_tensors(tensors:Dict[str,torch.Tensor],max_dim=None):

    max_dim = max([tensors[key].dim() for key in tensors])

    return {key : unsqueeze_to(tensors[key],max_dim) for key in tensors},max_dim

        

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=lambda:{'lr':1e-5})
    meta_optimizer_epoch : int = 2000
    meta_optimizer_data_per_iteration : int = 1
    meta_error_measurement : Any = torch.nn.MSELoss()
    meta_activator : Callable = torch.relu
    
    _all_leaf_tensors : SequenceTensorManager = field(init=False,default_factory=SequenceTensorManager)
    def __getitem__(self,key):
        return self._all_leaf_tensors.tensors_prediction[key]

    def __setitem__(self,key,value):
        self._all_leaf_tensors.tensors_prediction[key] = value

    @property
    def label_tensor(self):
        return self._all_leaf_tensors.tensors_label
    @label_tensor.setter
    def label_tensor(self,tor_tensor : torch.Tensor):
        self._all_leaf_tensors.tensors_label = tor_tensor
        return self._all_leaf_tensors.tensors_label

        
    
    assign_process_prediction : Callable = None

    def train_one_step_by_equation(self,label,prediction_quation):
        optim = self.meta_optimizer(self._all_leaf_tensors.get_all_params().values(),**self.meta_optimizer_params)
        optim.zero_grad()
        
        loss = self.meta_error_measurement(label,  prediction_quation)
        print(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

        return loss

    def train(self):
        #filter current sequence => unify dimensions => cals

        for _ in range(self.meta_optimizer_epoch):
            for pred_tensors,lab_tensors in zip(self._all_leaf_tensors,self._all_leaf_tensors.tensors_label):
                pred_unsqueezed,max_dim = unsqueeze_tensors(pred_tensors)
                lab_unsqueezed = unsqueeze_to(lab_tensors,max_dim)
                pred = self.assign_process_prediction(pred_unsqueezed,self.meta_activator)
                loss = self.train_one_step_by_equation(lab_unsqueezed,pred)
                
        return self._all_leaf_tensors.get_all_params()
    
    def predict(self,**kwarg):

        for key in kwarg:
            self[key].tensor = kwarg[key]
        
        
        ret = []
        for pred_tensors in self._all_leaf_tensors:
            pred_unsqueezed,_ = unsqueeze_tensors(pred_tensors)
            pred = self.assign_process_prediction(pred_unsqueezed,self.meta_activator)
            ret.append(pred)
        
        return ret