import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
from keras import Input
from keras.applications import InceptionV3
from keras.layers import Dense, Flatten, AveragePooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import cv2

tstart = datetime.now()
base_dir = './BAA/rsna-bone-age/'

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (500, 500))
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def feature_extraction(model, image_path, df, save_dir):
    file_list = os.listdir(image_path)
    file_list.sort()
    datas = []
    l = len(df)
    for i in range(0, l):
        data_dict = dict()
        filepath = image_path+df['id'][i]
        image = load_image(filepath)
        image = image/255.0
        gender = df['male'][i]
        #gender = tf.Variable(gender, dtype = tf.float32, shape = [1,])
        layer = K.function([model.layers[0].input, model.layers[-7].input], [model.layers[-2].output])
        print(image)
        tensors = layer([image,gender])
        data_dict['id'] = df['id'][i]
        data_dict['features'] = tensors
        data_dict['boneage'] = df['boneage'][i]
        datas.append(data_dict)
    new_df = pd.DataFrame(datas)
    new_df.to_csv(save_dir, index=False, header=True)
    print('********** Done ***********')


print('-'*20, 'data loading', '-'*20)

train = pd.read_csv("./BAA/rsna-bone-age/boneage-training-dataset/boneage-training-dataset2.csv")
train['id'] = train['id'].astype(str)
train['id'] = train['id']+'.png'
# train['male'].replace({False : 0, True : 1}, inplace=True)
train['male'] = train['male'].map(lambda x: np.array([1]) if x else np.array([0]))

validation = pd.read_csv("./BAA/rsna-bone-age/boneage-validation-dataset/boneage-validation-dataset.csv")
validation['id'] = validation['id'].astype(str)
validation['id'] = validation['id'] + '.png'
# validation['male'].replace({"FALSE" : 0, "TRUE" : 1}, inplace=True)
validation['male'] = validation['male'].map(lambda x: np.array([1]) if x=='TRUE' else np.array([0]))

test = pd.read_csv("./BAA/rsna-bone-age/boneage-testing-dataset/boneage-testing-dataset.csv")
test['id'] = test['id'].astype(str)
test['id'] = test['id'] + '.png'
# test['male'].replace({"F" : 0, "M" : 1}, inplace=True)
test['male'] = test['male'].map(lambda x: np.array([1]) if x=='M' else np.array([0]))


print('==================================================')
print('================= Building Model =================')
print('==================================================')

print('current time: %s' % str(datetime.now()))

i1 = Input(shape=(500, 500, 3), name='input_img')
i2 = Input(shape=(1,), name='input_gender')
base = InceptionV3(input_tensor=i1, input_shape=(500, 500, 3), include_top=False, weights=None)
feature_img = base.get_layer(name='mixed10').output
feature_img = AveragePooling2D((2, 2))(feature_img)
feature_img = Flatten()(feature_img)
feature_gender = Dense(32, activation='relu')(i2)
feature = concatenate([feature_img, feature_gender], axis=1)
o = Dense(1000, activation='relu')(feature)
o = Dense(1000, activation='relu')(o)
o = Dense(1)(o)
model = Model(inputs=[i1, i2], outputs=o)
optimizer = Adam(lr=1e-3)
model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae'])

print('==================================================')
print('======= Training Model on Boneage Dataset ========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

weight_path = base_dir + "{}_weights_16bit.best.hdf5".format('bone_age')
model.load_weights(weight_path)
tend = datetime.now()
print('current time: %s' % str(datetime.now()))
print('elapsed time: %s' % str((tend - tstart)))

feature_extraction(model, './BAA/rsna-bone-age/boneage-training-dataset/boneage-training-dataset2/', train , './BAA/training_data/train_features.csv')
feature_extraction(model, './BAA/rsna-bone-age/boneage-validation-dataset/boneage-validation-dataset/', validation , './BAA/validation_data/val_features.csv')
feature_extraction(model, './BAA/rsna-bone-age/boneage-testing-dataset/boneage-testing-dataset/', test , './BAA/testing_data/test_features.csv')
