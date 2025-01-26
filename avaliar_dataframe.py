"""
Avalia o modelo usando keras.eveluate
Usa lote de Imagens carregadas com dataframe
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
import numpy as np
import pandas as pd
import random

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score

PATH = '/home/hfrocha/envImagemCP/main/'
PATH_IMG = '/home/hfrocha/dados/'
IMAGE_SIZE = (224, 224)
CLASS_NAME = ['REJEITADA', 'ACEITA']

BASE_TESTE = '../../bases/base_teste_bruta_filename.csv'
#BASE_TESTE = '../../bases/base_teste_bruta_v2.csv'
#BASE_TESTE = '../../bases/base_teste_simulada.csv'
df_test = pd.read_csv(BASE_TESTE, sep=';')
len(df_test)

ROTULO = 'rotulo_adaptado1'
#ROTULO = 'rotulo_original'

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = datagen.flow_from_dataframe(df_test, directory = PATH_IMG, x_col = 'filename',
                                      y_col = ROTULO,
                                      class_mode='binary',
                                      classes=CLASS_NAME,
                                      target_size=IMAGE_SIZE,
                                      shuffle=False)
RODADA = '_R10'
MODO = 'cw'
#METRIC = 'bestf1'
METRIC = 'precision'
PATH_MODEL = 'modelos/best'+METRIC+'_'+MODO+RODADA
model = keras.models.load_model(PATH_MODEL)
result = model.evaluate(test_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(PATH_MODEL, result,'f1',f1)

"""
#METRIC = 'bestloss'+RODADA
#PATH_MODEL = PATH + 'model-'+MDL+METRIC
model = keras.models.load_model(PATH_MODEL)
result = model.evaluate(test_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(MDL,METRIC,result,'f1',f1)
"""


