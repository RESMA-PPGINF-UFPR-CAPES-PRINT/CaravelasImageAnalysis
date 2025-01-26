# -*- coding: utf-8 -*-
"""Caravelas - Treina CNN com cross validation

Resultados da dissertação foram obtidos por meio desse script

Fontes:
 * https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/
 * https://stackoverflow.com/questions/71676222/how-apply-kfold-cross-validation-using-tf-keras-utils-image-dataset-from-directo

# Setup
"""
import os
#os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
import timeit
import random

SEED = 88

# A cada rodada de treinamento é guardado historico
RODADA = '_R134'

# MODOS: us = UNDERSAMPLING ou cw = class_weight
MODO = 'us'

#ROTULO = 'rotulo_adaptado2'
ROTULO = 'rotulo_adaptado1'
#ROTULO = 'rotulo_original'

PATH_IMG = '/home/hfrocha/dados/'
FILE_HISTORICO = 'historico/hiperparam_'+MODO+RODADA

# parametros usados na CNN
params = [
#        {'BATCH_SIZE':32,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':8,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':8,'LR':1e-3,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},

#        {'BATCH_SIZE':16,'LR':1e-4,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-4,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-5,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-3,'DECAY':1e-4,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-3,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-4,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-4,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-5,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-4,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':16,'LR':1e-3,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-4,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
        {'BATCH_SIZE':32,'LR':1e-4,'DECAY':1e-5,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-5,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-4,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-5,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#         {'BATCH_SIZE':64,'LR':1e-4,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#                {'BATCH_SIZE':64,'LR':1e-4,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#         {'BATCH_SIZE':64,'LR':1e-5,'DECAY':None,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#                {'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-4,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#                {'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-5,'RELU':256,'POOL':(2,2),'TREINAVEL':True},
#        {'BATCH_SIZE':64,'LR':1e-4,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':64,'LR':1e-4,'DECAY':1e-5,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#         {'BATCH_SIZE':64,'LR':1e-5,'DECAY':None,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-4,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#        {'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-5,'RELU':256,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':2,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':64,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':512,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':1024,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':2,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':64,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':512,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':16,'LR':1e-3,'DECAY':None,'RELU':1024,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':None,'RELU':2,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':None,'RELU':64,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':None,'RELU':512,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':None,'RELU':1024,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-5,'RELU':2,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-5,'RELU':64,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-5,'RELU':512,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-3,'DECAY':1e-5,'RELU':1024,'POOL':(7,7),'TREINAVEL':True},

#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':2,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':64,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':512,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':1024,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-4,'DECAY':1e-5,'RELU':2,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-4,'DECAY':1e-5,'RELU':64,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-4,'DECAY':1e-5,'RELU':512,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-4,'DECAY':1e-5,'RELU':1024,'POOL':(7,7),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':2,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':64,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':512,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':None,'RELU':1024,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-5,'RELU':2,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-5,'RELU':64,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-5,'RELU':512,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-3,'DECAY':1e-5,'RELU':1024,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-5,'DECAY':None,'RELU':2,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-5,'DECAY':None,'RELU':64,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-5,'DECAY':None,'RELU':512,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':64,'LR':1e-5,'DECAY':None,'RELU':1024,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-5,'DECAY':None,'RELU':2,'POOL':(2,2),'TREINAVEL':True},
#{'BATCH_SIZE':32,'LR':1e-5,'DECAY':None,'RELU':512,'POOL':(2,2),'TREINAVEL':True},
        ]

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.model_selection import StratifiedShuffleSplit
from statistics import mean, stdev

# para calar o WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolut ....
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

historico = []

for param in params:

    BATCH_SIZE = param['BATCH_SIZE']
    LR = param['LR']
    DECAY = param['DECAY']
    POOL = param['POOL']
    RELU = param['RELU']
    TREINAVEL = param['TREINAVEL']

    IMAGE_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    EPOCH = 50
    FOLDS = 5
    TEST_SIZE = 30

    """# Carregando os Dados"""

    #BASE = '../../bases/base_exp_rotulo_v2.csv' # USAR COM EXP DOS ROTULOS ORIGINAL X ADAPTADO
    BASE = '../../bases/base_exp_bruta.csv'
    df_train = pd.read_csv(BASE, sep=';')
    df_train['rotulo'] = df_train[ROTULO].apply(lambda r: 1 if r=='ACEITA' else 0)

    def get_class_weight(y):
      n_samples = len(y)
      n_classes = 2
      weight_for_0, weight_for_1 = n_samples / (n_classes * np.bincount(y))
      class_weight = {0: weight_for_0, 1: weight_for_1}
      return class_weight

    class_weight = None
    if MODO=='cw':
        class_weight = get_class_weight(df_train['rotulo'])
        print('CLASS WEIGHT', class_weight)

    # o correto é fazer undersampling apenas na base de treino
    if MODO=='us':
        qta = len(df_train[df_train[ROTULO]=='ACEITA'])
        qtr = len(df_train[df_train[ROTULO]=='REJEITADA'])
        diff = qtr-qta
        df_train.drop(df_train.sample(diff, random_state=SEED).index, inplace=True)
        print('UNDERSAMPLING ficou',len(df_train), ' excluidas', diff)
    
    auto = tf.data.AUTOTUNE
    columns = ["filename", "rotulo"]

    def preprocess_image(sample):
      image_path = PATH_IMG + sample["filename"]
      image = tf.io.read_file(image_path)
      image = tf.image.decode_jpeg(image, 3)
      image = tf.image.resize(image, IMAGE_SIZE)
      return {"image": image}


    def dataframe_to_dataset(dataframe):
        dataframe = dataframe[columns].copy()
        labels = dataframe.pop('rotulo')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        return ds

    def prepare_dataset(dataframe):
        ds = dataframe_to_dataset(dataframe)
        ds = ds.map(lambda x, y: (preprocess_image(x), y)).cache()
        return ds

    dataset = prepare_dataset(df_train)


    base_model = ResNet50(weights="imagenet", input_shape=INPUT_SHAPE, include_top=False,)
    base_model.trainable = TREINAVEL

    """### Adapta o modelo"""

    # data augmentation, aplicando rotação, zoom e espelhamento horizontal de forma aleatória nas imagens
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = preprocess_input(inputs)
    x = data_augmentation(x)
    x = base_model(x, training=False)

    x1 = keras.layers.AveragePooling2D(POOL)(x)
    x1 = keras.layers.Flatten() (x1)
    if RELU>0:
        x1 = keras.layers.Dense(RELU, activation='relu')(x1)
    x1 = keras.layers.Dropout(0.3)(x1)

    outputs1 = keras.layers.Dense(1, activation='sigmoid')(x1)

    model1 = keras.Model(inputs, outputs1)



    callbacks = []

    earlystop = EarlyStopping(monitor="val_loss", patience=10, verbose=1,)
    callbacks.append(earlystop)


    def calcf1(precision, recall):
      recall = float(recall)
      precision = float(precision)
      f1 = 0
      if (precision+recall)>0:
          f1 = 2 * (precision*recall / (precision+recall))
      return round(f1,3)

    X = np.array(list(dataset.map(lambda x, y:x['image'])))
    y = np.array(list(dataset.map(lambda x, y:y)))

    kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=SEED)

    scores = {'precision': [], 'recall': [], 'f1': [], 'loss': []}
    for k, (train_index, val_index) in enumerate(kf.split(X, y)):
      print('FOLD',k)
      tf.keras.backend.clear_session()

      #compila modelo
      model1.compile(
        optimizer=keras.optimizers.Adam(LR, weight_decay=DECAY),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[BinaryAccuracy(), Precision(), Recall()],
      )

      history = model1.fit(X[train_index], y[train_index], validation_data=(X[val_index], y[val_index]),
              batch_size=BATCH_SIZE, shuffle=False, use_multiprocessing=False,
              epochs=EPOCH, callbacks=callbacks, class_weight=class_weight, verbose=1)

      dfh = pd.DataFrame.from_dict(history.history)
      dfh['f1'] = dfh.apply(lambda x: calcf1(x.val_precision, x.val_recall), axis=1)
      result = dfh.sort_values(by=['f1','val_precision'], ascending=False).head(1)

      scores['loss'].append(result['val_loss'].item())
      scores['precision'].append(result['val_precision'].item())
      scores['recall'].append(result['val_recall'].item())
      scores['f1'].append(result['f1'].item())

    historico.append([mean(scores['precision']), stdev(scores['precision']),
              mean(scores['recall']), stdev(scores['recall']),
              mean(scores['f1']), stdev(scores['f1']),
              mean(scores['loss']), stdev(scores['loss']),
              MODO,BATCH_SIZE,LR,DECAY,POOL,RELU,SEED,TREINAVEL,EPOCH,RODADA,'base_exp_bruta.csv'])

    dfh = pd.DataFrame.from_dict(historico)
    columns={0:'precision', 1:'precision_stdev',
         2:'recall', 3:'recall_stdev',
         4:'f1', 5:'f1_stdev',
         6:'loss', 7: 'loss_stdev',
         8:'modo',9:'bs',
         10:'lr',11:'decay',
         12:'pool',13:'relu',
         14:'seed',15:'trainable',
         16:'epoch',17:'r',18:'base'}
    dfh.rename(columns=columns, inplace=True)
    dfh.to_csv(FILE_HISTORICO, index=False)

print(MODO, RODADA)
print(dfh.head())

