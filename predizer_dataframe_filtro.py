""" 
Prediz lote de imagens carregadas com dataframe
Usa dataset que simula producao
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
import numpy as np
import pandas as pd
import random

SEED = 88

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score

PATH = '/home/hfrocha/envImagemCP/main/modelos/'
PATH_IMG = '/home/hfrocha/dados/'
IMAGE_SIZE = (224, 224)
CLASS_NAME = ['REJEITADA', 'ACEITA']

ROTULO = 'rotulo_adaptado1'
#ROTULO = 'rotulo_adaptado2'

BASE_TESTE = '../../bases/base_teste_bruta_filename_motivo.csv'
df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test['rotulo'] = df_test[ROTULO].apply(lambda r: 1 if r=='ACEITA' else 0)
#df_test = df_test[df_test['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]
print(len(df_test))

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = datagen.flow_from_dataframe(df_test, directory = PATH_IMG, x_col = 'filename',
                                      y_col = ROTULO,
                                      class_mode='binary',
                                      classes=CLASS_NAME,
                                      target_size=IMAGE_SIZE,
                                      shuffle=False)

#MDL = 'bestprecision_us_R205'
MDL = 'bestprecision_us_R132'
PATH_MODEL = PATH + MDL

model = keras.models.load_model(PATH_MODEL, compile=False)
predictions = model.predict(test_ds, verbose=0)
#y_pred = tf.where(predictions<=0.5,0,1)
#y_pred = y_pred.numpy()

# essa conta que ele faz no evaluate
y_pred = []
for prediction in predictions:
    score = round(prediction[0],2)
    y_pred.append(0 if score <=0.5 else 1)

df_test['proba'] = predictions
df_test['predito'] = y_pred

FILE_HISTORICO = 'historico/prediction_'+MDL+'.csv'
df_test.to_csv(FILE_HISTORICO, index=False)

y_true = df_test['rotulo']

cmpred = confusion_matrix(y_true, y_pred, normalize='pred')
cmtrue = confusion_matrix(y_true, y_pred, normalize='true')
cm = confusion_matrix(y_true, y_pred)
print('CM abs',cm)
print('CM pred',cmpred)
print('CM true',cmtrue)

precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1 = 2 * (precision*rec / (precision+rec))
print('PRE',precision,'REC', rec, 'f1',f1)

