import os
import copy
import click
import torch
import torch.utils.data

import bc_resnet_model
import get_data
import train
import apply
import util

from flask import Flask, request

app = Flask(__name__)


@app.route("/keyword_spotting", methods=['POST', 'GET'])
def apply_command():

    # if not os.path.exists(model_file):
    #     raise FileExistsError(f"model file {model_file} not exists")
    # if not os.path.exists(wav_file):
    #     raise FileExistsError(f"sound file {wav_file} not exists")

    input_speech = request.files.get('file', '')
    model_speech = request.form['model']
    video_path = str(input_speech.filename)
    input_speech.save(video_path)
    wav_file = video_path
    model_file = "/AIHN/BC-ResNet/example_model/model-sc-2.pt"
    device = util.get_device()

    scale = 2
    dropout = 0.1
    subspectral_norm = True
    model = bc_resnet_model.BcResNetModel(
        n_class=get_data.N_CLASS,
        scale=scale,
        dropout=dropout,
        use_subspectral=subspectral_norm,
    ).to(device)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
    model.eval()

    predictions = apply.apply_to_file(model, wav_file, device)
    print(predictions[:5])
    a = 0
    l = None
    for label, prob in predictions[:5]:
        if prob > a:
            a = prob
            l = label
    return l
app.run(host="0.0.0.0", port=5467, debug=False, threaded=True)