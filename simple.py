from pyplus.pytorch.internal.internal import *
from typing import Dict


#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=lambda:{'lr':1e-4})
    meta_epoch : int = 3000
    meta_data_per_iteration : int = 1
    meta_error_measurement : Any = torch.nn.MSELoss
    #meta_activator : Callable = field(default_factory=nn.LeakyReLU)
    
    all_predict_tensors : TensorsManager = field(init=False,default_factory=TensorsManager)
    all_label_tensors : TensorsManager = field(init=False,default_factory=TensorsManager)

    _assign_process_prediction : Callable = None
    
    @property
    def process(self):
        return self._assign_process_prediction

    @process.setter
    def process(self,function:Callable):
        self._assign_process_prediction = function


    def _train_one_step_by_equation(self,label:torch.Tensor,prediction_quation:torch.Tensor):
        optim = self.meta_optimizer(self.all_predict_tensors.get_all_params().values(),**self.meta_optimizer_params)
        optim.zero_grad()
        
        loss = self.meta_error_measurement()(label,  prediction_quation)
        loss.backward()
        optim.step()
        optim.zero_grad()

        return loss

    def train(self,show_every_iteration=False):
        #filter current sequence => unify dimensions => cals
        self._current_mode = ProcessMode.ASSIGN
        self._assign_process_prediction()
        self._current_mode = ProcessMode.PROCESS
        for epoch in range(self.meta_epoch):
            min_sequence = min(self.all_predict_tensors.get_min_sequence_length(TTPType.INPUT),self.all_label_tensors.get_min_sequence_length(TTPType.DEFAULT))

            for sequence_ind in range(0,min_sequence,self.meta_data_per_iteration):
                pred_tensors = self.all_predict_tensors[sequence_ind:sequence_ind+self.meta_data_per_iteration]
                lab_tensors = self.all_label_tensors[sequence_ind:sequence_ind+self.meta_data_per_iteration]

                self._pred_unsqueezed,max_dim = pred_tensors.unsqueeze_tensors()
                self._lab_unsqueezed,_ = lab_tensors.unsqueeze_tensors(max_dim)

                pred = self._assign_process_prediction()
                loss = self._train_one_step_by_equation(self._lab_unsqueezed.tensors[0].tensor,pred)

            
                if show_every_iteration:
                    print(f'Epoch : {epoch}/{self.meta_epoch}\tIteration : {sequence_ind}/{min_sequence}\tLoss : {loss}')
                
        return lambda **kwarg: self.predict(**kwarg)
    
    def predict(self,**kwarg):
        self._current_mode = ProcessMode.PROCESS

        for key in kwarg:
            self.all_predict_tensors.change_tensor(key,kwarg[key])
        
        min_sequence = self.all_predict_tensors.get_min_sequence_length(TTPType.INPUT)

        ret = []
        for sequence_ind in range(0,min_sequence):
            pred_tensors = self.all_predict_tensors[sequence_ind]
            self._pred_unsqueezed,_ = pred_tensors.unsqueeze_tensors()
            pred = self._assign_process_prediction()
            ret.append(pred)

        
        return ret
    
    def input(self:Self,name:str,tensor:torch.Tensor,axis_sequence=0):
        if self._current_mode == ProcessMode.ASSIGN:
            self.all_predict_tensors.new_tensor(name=name,ttype=TTPType.INPUT,axis_sequence=axis_sequence,tensor=tensor)
            return tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._pred_unsqueezed.get_tensor(name).tensor 

    def parameter(self:Self,name:str,tensor:torch.Tensor,axis_sequence=-1):
        if self._current_mode == ProcessMode.ASSIGN:
            self.all_predict_tensors.new_tensor(name=name,ttype=TTPType.PARAMETER,axis_sequence=axis_sequence,tensor=tensor)
            return tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._pred_unsqueezed.get_tensor(name).tensor 

    def label(self:Self,name:str,tensor:torch.Tensor,axis_sequence=0):
        if self._current_mode == ProcessMode.ASSIGN:
            self.all_label_tensors.new_tensor(name=name,ttype=TTPType.DEFAULT,axis_sequence=axis_sequence,tensor=tensor)
            return tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._lab_unsqueezed.get_tensor(name).tensor 

    def get_parameters(self:Self)->Dict:
        return self.all_predict_tensors.get_all_params()

