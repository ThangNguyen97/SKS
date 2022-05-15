# Keyword Spotting Transformer
This is the unofficial TensorFlow  implementation of the Keyword Spotting Transformer model. This model is used to train on the 35 words speech command dataset

Paper : [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/pdf/2104.00769v2.pdf)

## Model architecture
![alt text](https://github.com/Ahmad-Omar-Ahsan/keyword_spotting_transformer/blob/main/KWS_transformer.png)

## Download the dataset
To download the dataset use the following command

```
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data
mv ./speech_commands_v0.02.tar.gz ./data
cd ./data
tar -xf ./speech_commands_v0.02.tar.gz
cd ../
```
## Setup virtual environment
```
virtualenv -p python3 venv
source ./venv/bin/activate
```

## Install dependencies 

```
pip install -r requirements.txt
```

## Training the model
To train the model run this command

```
python3 train.py --data_dir ${Path to data directory} \
                 --logdir ${Path to log directory} \
                 --num_layers ${Number of sequential encoder layers} \
                 --d_model ${Dimension of the encoder layers} \
                 --num_heads ${Number of heads in multi head attention layer} \
                 --mlp_dim ${Dimension of mlp layers} \
                 --lr ${Learning rate} \
                 --weight_decay ${Weight decay} \
                 --batch_size ${Batch size} \
                 --epochs ${Number of epochs} \
                 --save_dir ${Directory to save the model weights}

```
To track your training metrics 
```
tensorboard --logdir  ${Path to log directory}
```
## Predicting keyword of audio file
To predict the keyword of the audio file

```
python3 test.py --model_dir ${Saved model directory} \
                --file_path ${Audio file}
```
