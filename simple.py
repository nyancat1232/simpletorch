from simpletorch.internal.internal import TensorsManager,ProcessMode,MetaTensorType,MetaDataType
import torch
from typing import Dict,ClassVar,Any,Self,List,Tuple,Callable
from dataclasses import dataclass,field

#works on pytorch 2.1.1


@dataclass
class RecommendInit:
    DEFAULT : ClassVar[torch.nn.init] = torch.nn.init.xavier_normal_
    RELU : ClassVar[torch.nn.init] = torch.nn.init.kaiming_normal_

@dataclass
class CurrentStateInformation:
    current_epoch : int
    max_epoch : int
    current_iteration : int
    len_iteration : int
    current_loss : float


#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=lambda:{'lr':1e-4})
    meta_epoch : int = 3000
    meta_data_per_iteration : int = 1
    meta_error_measurement : Any = torch.nn.MSELoss
    
    all_predict_tensors : TensorsManager = field(init=False,default_factory=TensorsManager)
    all_label_tensors : TensorsManager = field(init=False,default_factory=TensorsManager)

    def __post_init__(self):
        if not hasattr(self, 'process'):
            raise NotImplementedError("Please, override \ndef process(self):\n\n function in this class.")
        
        self._current_mode = ProcessMode.ASSIGN
        self.process()
        self._current_mode = ProcessMode.PROCESS
        
        if self.all_label_tensors.is_empty():
            raise NotImplementedError('Please, set labels by using \n self.label(..) \n in "def process(self):".')

    def _train_one_step_by_equation(self,label:torch.Tensor,prediction_quation:torch.Tensor):
        optim = self.meta_optimizer(self.all_predict_tensors.get_all_params().values(),**self.meta_optimizer_params)
        
        loss = self.meta_error_measurement()(prediction_quation,label)
        loss.backward()
        optim.step()
        optim.zero_grad()

        return loss

    def train(self,show_every_iteration=False):
        #filter current sequence => unify dimensions => cals

        for epoch in range(self.meta_epoch):
            min_sequence = min(self.all_predict_tensors.get_min_sequence_length(MetaTensorType.INPUT),self.all_label_tensors.get_min_sequence_length(MetaTensorType.DEFAULT))

            for sequence_ind in range(0,min_sequence,self.meta_data_per_iteration):
                self._current_tensors_prediction = self.all_predict_tensors[sequence_ind:sequence_ind+self.meta_data_per_iteration]
                lab_tensors = self.all_label_tensors[sequence_ind:sequence_ind+self.meta_data_per_iteration]

                pred = self.process()
                loss = self._train_one_step_by_equation(lab_tensors.tensors[0].tensor,pred)

                if show_every_iteration:
                    self._current_state = CurrentStateInformation(current_epoch=epoch,max_epoch=self.meta_epoch,
                                         current_iteration=sequence_ind,len_iteration=min_sequence,
                                         current_loss=loss)
                    self.iteration_event_function()
                
        return lambda **kwarg: self.predict(**kwarg)
    
    def predict(self,**kwarg):
        self._current_mode = ProcessMode.PROCESS

        for key in kwarg:
            self.all_predict_tensors.change_tensor(key,kwarg[key])
        
        min_sequence = self.all_predict_tensors.get_min_sequence_length(MetaTensorType.INPUT)

        ret = []
        for sequence_ind in range(0,min_sequence):
            self._current_tensors_prediction = self.all_predict_tensors[sequence_ind]
            pred = self.process()
            ret.append(pred)

        
        return ret
    
    def input(self:Self,data:List,meta_data_type:MetaDataType,name:str,axis_sequence=0)->torch.Tensor:
        if self._current_mode == ProcessMode.ASSIGN:
            ret = self.all_predict_tensors.new_tensor(name=name,
                                                meta_data_type=meta_data_type,
                                                meta_tensor_type=MetaTensorType.INPUT,
                                                axis_sequence=axis_sequence,
                                                tensor_data=data)
            return ret
        elif self._current_mode == ProcessMode.PROCESS:
            return self._current_tensors_prediction.get_tensor(name).tensor 

    def parameter(self:Self, 
                  size:Tuple,name:str,
                  init_func:RecommendInit=RecommendInit.DEFAULT,
                  axis_sequence=-1) -> torch.Tensor:

        if self._current_mode == ProcessMode.ASSIGN:

            new_tensor = torch.randn(*size)
            try:
                new_tensor = init_func(new_tensor)
            except:
                pass

            self.all_predict_tensors.new_tensor(name=name,
                                                meta_tensor_type=MetaTensorType.PARAMETER,
                                                axis_sequence=axis_sequence,
                                                tensor_data=new_tensor)
            return new_tensor
        elif self._current_mode == ProcessMode.PROCESS:
            return self._current_tensors_prediction.get_tensor(name).tensor 

    def label(self:Self,data:List,meta_data_type:MetaDataType,axis_sequence=0)->torch.Tensor:
        if self._current_mode == ProcessMode.ASSIGN:
            ret = self.all_label_tensors.new_tensor(meta_data_type=meta_data_type,
                                              axis_sequence=axis_sequence,
                                              tensor_data=data)
            return ret

    def get_parameters(self:Self)->Dict:
        return self.all_predict_tensors.get_all_params()

