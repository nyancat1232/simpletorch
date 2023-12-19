from pyplus.pytorch.internal.internal import *



#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=lambda:{'lr':1e-4})
    meta_epoch : int = 3000
    meta_data_per_iteration : int = 1
    meta_error_measurement : Any = torch.nn.MSELoss()
    meta_activator : Callable = field(default_factory=nn.LeakyReLU)
    
    all_predict_tensors : TensorsSquence = field(init=False,default_factory=TensorsSquence)
    all_label_tensors : TensorsSquence = field(init=False,default_factory=TensorsSquence)

    assign_process_prediction : Callable = None

    def train_one_step_by_equation(self,label,prediction_quation):
        optim = self.meta_optimizer(self.all_predict_tensors.get_all_params().values(),**self.meta_optimizer_params)
        optim.zero_grad()
        
        loss = self.meta_error_measurement(label,  prediction_quation)
        loss.backward()
        optim.step()
        optim.zero_grad()

        return loss

    def train(self,show_progress=True):
        #filter current sequence => unify dimensions => cals
        self._current_mode = ProcessMode.ASSIGN
        self.assign_process_prediction(self.meta_activator)
        self._current_mode = ProcessMode.PROCESS

        for _ in range(self.meta_epoch):
            for pred_tensors,lab_tensors in zip(self.all_predict_tensors,self.all_label_tensors):
                if show_progress:
                    print(pred_tensors)
                self._pred_unsqueezed,max_dim = unsqueeze_tensors(pred_tensors)
                self._lab_unsqueezed,_ = unsqueeze_tensors(lab_tensors,max_dim)

                pred = self.assign_process_prediction(self.meta_activator)
                loss = self.train_one_step_by_equation([value for value in self._lab_unsqueezed.values()][0],pred)
                
        return lambda **kwarg: self.predict(**kwarg)
    
    def predict(self,**kwarg):
        self._current_mode = ProcessMode.PROCESS

        for key in kwarg:
            self.all_predict_tensors.change_tensor(key,kwarg[key])
        
        
        ret = []
        for pred_tensors in self.all_predict_tensors:
            self._pred_unsqueezed,_ = unsqueeze_tensors(pred_tensors)

            self._current_mode = ProcessMode.PROCESS
            pred = self.assign_process_prediction(self.meta_activator)
            ret.append(pred)
        
        return ret
    
    def input(self:Self,name:str,tensor:torch.Tensor):
        if self._current_mode == ProcessMode.ASSIGN:
            self.all_predict_tensors.new_tensor(name,TorchTensorPlusInternal(ttype=TTPType.DEFAULT,axis_sequence=0),tensor)
            return tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._pred_unsqueezed[name] 

    def parameter(self:Self,name:str,tensor:torch.Tensor):
        if self._current_mode == ProcessMode.ASSIGN:
            self.all_predict_tensors.new_tensor(name,TorchTensorPlusInternal(ttype=TTPType.PARAMETER,axis_sequence=-1),tensor)
            return tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._pred_unsqueezed[name] 

    def label(self:Self,name:str,tensor:torch.Tensor):
        if self._current_mode == ProcessMode.ASSIGN:
            self.all_label_tensors.new_tensor(name,TorchTensorPlusInternal(ttype=TTPType.DEFAULT,axis_sequence=0),tensor)
            return tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._lab_unsqueezed[name] 


