import torch
from process.BC_ResNet import  bc_resnet_model
import tensorflow_addons as tfa
import tensorflow as tf
class PretrainedModel:
    _instance = None
    def __new__(cls, cfg=None, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PretrainedModel, cls).__new__(cls, *args, **kwargs)
            # model BC-RESNET
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = bc_resnet_model.BcResNetModel(n_class=35,scale=2,dropout=0.1,use_subspectral=True).to(device)
            model.load_state_dict(torch.load(cfg['bc_resnet'], map_location=torch.device('cpu')))
            cls.bc_resnet = model.eval()
            # model KWT
            cls.kwt_model = tf.keras.models.load_model(cfg['kwt'], custom_objects={"optimizer": tfa.optimizers.AdamW(
                learning_rate=0.001, weight_decay=0.0001
            )})
        return cls._instance
