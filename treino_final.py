# -*- coding: utf-8 -*-
"""Caravelas - 2 - Treino CNN.ipynb

Treino com base dev (treino + validação) e validação na base de teste

Resultados obtidos estão na dissertação

"""

"""## Setup"""

import os
#os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
import timeit
import random

inicio1 = timeit.default_timer()
SEED = 88
RODADA = '_R205'
LR = 1e-4
DECAY = None
RELU = 256
POOL = (2,2)
BATCH_SIZE = 64
TRAINABLE =True

#MODO = 'cw' #class weight
MODO = 'us' #undersamlin
#ROTULO = 'rotulo_adaptado1'
ROTULO = 'rotulo_adaptado2'

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, F1Score

# para calar o WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolut ....
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

BASE_TRAIN = '../../bases/base_exp_bruta.csv'
df_train = pd.read_csv(BASE_TRAIN, sep=';')
df_train['rotulo'] = df_train[ROTULO].apply(lambda r: 1 if r=='ACEITA' else 0)

# undersampling
if MODO=='us':
    qta = len(df_train[df_train[ROTULO]=='ACEITA'])
    qtr = len(df_train[df_train[ROTULO]=='REJEITADA'])
    diff = qtr-qta
    df_train.drop(df_train.sample(diff, random_state=SEED).index, inplace=True)
    print('UNDERSAMPLING ficou',len(df_train), ' excluidas',diff)

BASE_TEST = '../../bases/base_teste_bruta_filename.csv'
df_test = pd.read_csv(BASE_TEST, sep=';')
IMAGE_SIZE = (224, 224)
CLASS_NAME = ['REJEITADA', 'ACEITA']
PATH_IMG = '/home/hfrocha/dados/'

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_ds = datagen.flow_from_dataframe(df_train, directory = PATH_IMG, x_col = 'filename', 
        y_col = ROTULO, 
        class_mode='binary',
        classes=CLASS_NAME,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True)

test_ds = datagen.flow_from_dataframe(df_test, directory = PATH_IMG, x_col = 'filename', 
        y_col = ROTULO,
        class_mode='binary',
        classes=CLASS_NAME,
        target_size=IMAGE_SIZE,
        shuffle=False)

PATH = '/home/hfrocha/envImagemCP/main/'
INPUT_SHAPE = (224, 224, 3)


# data augmentation, aplicando rotação, zoom e espelhamento horizontal de forma aleatória nas imagens
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)



base_model = ResNet50(weights="imagenet", input_shape=INPUT_SHAPE, include_top=False,)
base_model.trainable = TRAINABLE


inputs = keras.Input(shape=INPUT_SHAPE)
x = preprocess_input(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)


# opção 1 - essa é mais provavel de ser a certa
# camada de average pooling
x1 = keras.layers.AveragePooling2D()(x)
# camada flattening
x1 = keras.layers.Flatten() (x1)
# camada fully-connected que utiliza a função de ativação ReLu
x1 = keras.layers.Dense(RELU, activation='relu')(x1)
# camada de regularização do tipo dropout com porcentagem de 30%
x1 = keras.layers.Dropout(0.3)(x1)

outputs1 = keras.layers.Dense(1, activation='sigmoid')(x1)

model1 = keras.Model(inputs, outputs1)

model1.compile(
  optimizer=keras.optimizers.Adam(LR),
  loss=keras.losses.BinaryCrossentropy(),
  metrics=[BinaryAccuracy(), Precision(), Recall()],
)



callbacks = []

# salva o modelo com menor perda
MDLBESTLOSS = "modelos/bestprecision_"+MODO+RODADA
checkpoint2 = ModelCheckpoint(MDLBESTLOSS, monitor='val_precision', verbose=0, save_best_only=True, mode='max')
callbacks.append(checkpoint2)

# Early Stopping baseado na perda obtida na base da validação (ela não detalha)
earlystop = EarlyStopping(monitor="val_loss", patience=25, verbose=1)
callbacks.append(earlystop)

# salva o modelos com maior F1
MDLBESTF1 = "modelos/bestf1_"+MODO+RODADA
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        current_precision = logs.get("val_precision")
        current_recall = logs.get("val_recall")
        current_f1 = 0
        if (current_precision+current_recall)>0:
            current_f1 = 2 * (current_precision*current_recall / (current_precision+current_recall))
            #print("\nEnd epoch {} of training f1 {}".format(epoch, current_f1))
        if np.greater(current_f1, self.best_f1):
            self.best_f1 = current_f1
            print("\nBest F1 now is {}".format(current_f1))
            self.model.save(MDLBESTF1, save_format='tf')

callbacks.append(CustomCallback())

def get_class_weight(y):
    n_samples = len(y)
    n_classes = 2
    weight_for_0, weight_for_1 = n_samples / (n_classes * np.bincount(y))
    class_weight = {0: weight_for_0, 1: weight_for_1}

    return class_weight

class_weight = None
if MODO=='cw':
    df_train['rotulo'] = df_train[ROTULO].apply(lambda r: 1 if r=='ACEITA' else 0)
    class_weight = get_class_weight(df_train['rotulo'])
    print('CLASS WEIGHT', class_weight)


def calcf1(precision, recall):
    recall = float(recall)
    precision = float(precision)
    f1 = 0
    if (precision+recall)>0:
        f1 = 2 * (precision*recall / (precision+recall))
    return f1


epochs = 50



history1 = model1.fit(train_ds, epochs=epochs, callbacks=callbacks, 
        use_multiprocessing=False,shuffle=False,
        validation_data=test_ds, class_weight=class_weight, verbose=1)
fim1 = timeit.default_timer()

dfh1 = pd.DataFrame.from_dict(history1.history)
dfh1['f1'] = dfh1.apply(lambda x: calcf1(x.val_precision, x.val_recall), axis=1)
FILE_HISTORICO = PATH + '/historico/modelo_final_'+MODO+RODADA+'.csv'
dfh1.to_csv(FILE_HISTORICO, index=False)

result = dfh1.sort_values(by=['f1'], ascending=False).head(1)
print(result)

duracao = fim1 - inicio1
dft1 = pd.DataFrame.from_dict({'inicio':[inicio1],'fim':[fim1],'duracao_seg':[duracao]})
# dft1.head()
FILE_TEMPO = PATH + '/tempo/modelo_final_'+ROTULO+MODO+'.csv'
dft1.to_csv(FILE_TEMPO, index=False)



model = keras.models.load_model(MDLBESTLOSS)
result = model.evaluate(test_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(MDLBESTLOSS, result,'f1',f1)


model = keras.models.load_model(MDLBESTF1)
result = model.evaluate(test_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(MDLBESTF1,result,'f1',f1)


