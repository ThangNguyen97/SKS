# import os
# import copy
# import click
# import torch
# import torch.utils.data
#
# import bc_resnet_model
# import get_data
# import train
# import apply
# import util
import logging
from flask import Flask, request
from config.config import get_config
from process.PretrainModel import PretrainedModel
import tensorflow as tf
from flask_restful import Api
app = Flask(__name__)
api = Api(app)


config_app = get_config()
# logging.basicConfig(filename=config_app['log']['app'],
#                     format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     except RuntimeError as e:
#         print(e)
models = PretrainedModel(config_app['model_weight'])
from service.kws import KWS

api.add_resource(KWS, '/keyword_spotting')
app.run(host=config_app['server']['ip_address'], port=config_app['server']['port'], debug=False, threaded=True)

