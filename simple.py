import torch
from dataclasses import dataclass,field

from typing import Any,List

@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = None
    meta_error_measurement : Any = None
    meta_activator : Any = None
    model : torch.nn.Module = None
    model_input_datas : Any = None
    model_input_labels : Any = None

    def init_metas(self):
        loss_fn = self.error_measurement()
        mo_instance = self.model()
        optimizer = self.optimizer(mo_instance.parameters(), lr=1e-3)

        
        return TorchPlusInstance(loss=loss_fn,model=mo_instance,optimizer=optimizer)
    
@dataclass
class TorchPlusInstance:
    loss : Any
    model: Any
    optimizer: Any
