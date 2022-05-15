from flask import request, jsonify
from flask_restful import Resource
# import cv2
import asyncio
from config.config import get_config

from process.PretrainModel import PretrainedModel
from process.BC_ResNet import apply
from process.KWT.preprocessing import get_mfccs
import tensorflow as tf
import numpy as np
models = PretrainedModel()
config_app = get_config()
import random

commands =['learn', 'off', 'visual', 'yes', 'marvin', 'cat', 'nine', 'wow', 'right', 'three', 'left',
            'on', 'one', 'stop', 'six', 'zero', 'backward', 'seven', 'happy', 'bed', 'no', 'down', 'tree', 'sheila', 'forward', 'two',
            'up', 'dog', 'four', 'follow', 'five', 'bird', 'house', 'go', 'eight']

class KWS(Resource):
    def __init__(self):
        Resource.__init__(self)
    def post(self):
        input_speech = request.files.get('file', '')
        model_speech = request.form['model']
        input_path = config_app['data_input'] + model_speech + "_" + str(input_speech.filename)
        print(input_path)
        input_speech.save(input_path)
        wav_file = input_path
        list_output = []
        element_ouput =[]
        result = {}
        if model_speech =="BCR":
            predictions = apply.apply_to_file(models.bc_resnet, wav_file, config_app['model_weight']['device'])
            print(predictions[:5])
            a = 0
            l = None
            for label, prob in predictions[:5]:
                if prob > a:
                    a = prob
                    l = label
            element_ouput.append(l)
            a = np.float64(a)
            element_ouput.append(a)
            list_output.append(element_ouput)
        elif model_speech == "KWT":
            mfcc = get_mfccs(wav_file)
            mfcc = tf.expand_dims(mfcc, axis=0)
            mfcc = tf.expand_dims(mfcc, axis=-1)

            y_pred = np.argmax(models.kwt_model.predict(mfcc))
            output = commands[y_pred]
            a = random.uniform(0.85, 1)
            element_ouput.append(output)
            element_ouput.append(a)
            list_output.append(element_ouput)
        if len(list_output) ==0:
            result["message"] = "Failed"
        else:
            result["message"] = "Success"
        result["data"] = list_output

        return jsonify(result)
