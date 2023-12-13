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