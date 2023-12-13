import torch
from dataclasses import dataclass,field

from typing import Any,List
#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = None
    meta_optimizer_learning_rate : float = 0.01
    meta_optimizer_epoch : int = 2000
    meta_error_measurement : Any = None
    meta_activator : Any = None
    constatns : Any
    parameters : List[Any] = None
    reflect_predictions : Any
    reflect_labels : Any

    def calc(self):
        loss_fn = self.meta_error_measurement()
        optimizer = self.meta_optimizer(self.model_parameters, lr=self.meta_optimizer_learning_rate)
        for t in range(self.meta_optimizer_epoch):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = loss_fn(y_pred, y)
            if t % 100 == 99:
                print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
