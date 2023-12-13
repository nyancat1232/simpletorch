import torch
from dataclasses import dataclass,field

from typing import Any,List,Dict
#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = None
    meta_optimizer_learning_rate : float = 0.01
    meta_optimizer_epoch : int = 2000
    meta_error_measurement : Any = None
    meta_activator : Any = None
    
    all_tensors : Dict = field(default_factory=dict)

    def get_all_params(self):
        return [self.all_tensors[key] for key in self.all_tensors if self.all_tensors[key].requires_grad]
    def train_one_step_by_equation(self,label,prediction_quation):
        loss = self.meta_error_measurement()(label,  prediction_quation)
        optim = self.meta_optimizer(self.get_all_params(),lr=self.meta_optimizer_learning_rate)
        optim.zero_grad()
        loss.backward()
        optim.step()