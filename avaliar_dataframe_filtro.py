"""
Avalia o modelo usando keras.eveluate
Usa lote de Imagens carregadas com dataframe
Aplica filtro simulando produção
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

BASE_TESTE = '../../bases/base_teste_bruta_filename_motivo.csv'
#BASE_TESTE = '../../bases/base_teste_bruta_v2.csv'
#BASE_TESTE = '../../bases/base_teste_simulada.csv'
df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test = df_test[df_test['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]
print(len(df_test))

ROTULO = 'rotulo_adaptado1'
#ROTULO = 'rotulo_adaptado2'

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = datagen.flow_from_dataframe(df_test, directory = PATH_IMG, x_col = 'filename',
                                      y_col = ROTULO,
                                      class_mode='binary',
                                      classes=CLASS_NAME,
                                      target_size=IMAGE_SIZE,
                                      shuffle=False)

PATH_MODEL = 'modelos/bestprecision_us_R13'
PATH_MODEL = 'modelos/bestf1_us_R131'
PATH_MODEL = 'modelos/bestprecision_us_R132'
PATH_MODEL = 'modelos/bestf1_us_R134'
model = keras.models.load_model(PATH_MODEL)
result = model.evaluate(test_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(PATH_MODEL, result)
print(result['precision'],',',result['recall'],',',f1)

