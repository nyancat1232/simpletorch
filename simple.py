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
    current_epoch : int = -1
    max_epoch : int = -1
    current_iteration : int = -1
    len_iteration : int = -1
    current_loss : float = 0.0
    all_features : TensorsManager = field(default_factory=TensorsManager)
    all_labels : TensorsManager = field(default_factory=TensorsManager)


#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@dataclass
class TorchPlus:
    meta_optimizer : torch.optim.Optimizer = torch.optim.SGD
    meta_optimizer_params : Dict = field(default_factory=lambda:{'lr':1e-4})
    meta_data_per_iteration : int = 1
    meta_error_measurement : Any = torch.nn.MSELoss
    
    all_predict_tensors : TensorsManager = field(init=False,default_factory=TensorsManager)
    all_label_tensors : TensorsManager = field(init=False,default_factory=TensorsManager)

    def train_init(self):
        if not hasattr(self, 'process'):
            raise NotImplementedError("Please, override \ndef process(self):\n\n function in this class.")
        
        self._current_mode = ProcessMode.ASSIGN
        self.process()
        self._current_mode = ProcessMode.PROCESS

        if self.all_label_tensors.is_empty():
            raise NotImplementedError('Please, set labels by using \n self.label(..) \n in "def process(self):".')

        print('init complete')
        
        epoch = -1
        while True:
            epoch += 1
            min_sequence = min(self.all_predict_tensors.get_min_sequence_length(MetaTensorType.INPUT),self.all_label_tensors.get_min_sequence_length(MetaTensorType.DEFAULT))

            for sequence_ind in range(0,min_sequence,self.meta_data_per_iteration):
                self._current_tensors_prediction = self.all_predict_tensors[sequence_ind:sequence_ind+self.meta_data_per_iteration]
                lab_tensors = self.all_label_tensors[sequence_ind:sequence_ind+self.meta_data_per_iteration]

                pred = self.process()

                optim = self.meta_optimizer(self.all_predict_tensors.get_all_params().values(),**self.meta_optimizer_params)
        
                loss = self.meta_error_measurement()(pred,lab_tensors.tensors[0].tensor)
                loss.backward()
                optim.step()
                optim.zero_grad()

                csi=CurrentStateInformation()
                csi.current_epoch = epoch
                csi.current_iteration = sequence_ind
                csi.len_iteration = min_sequence
                csi.current_loss = loss
                csi.all_features = {tensor.name : tensor.tensor for tensor in self.all_predict_tensors.tensors}
                csi.all_labels = self.all_label_tensors.tensors[0].tensor
                self.show_progress(csi)
            
            yield lambda **kwarg: self.predict(**kwarg)


    def train(self,epoch:int=1000):
        ret=None
        for epoch,func in zip(range(epoch),self.train_init()):
            ret = func
        return ret

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

