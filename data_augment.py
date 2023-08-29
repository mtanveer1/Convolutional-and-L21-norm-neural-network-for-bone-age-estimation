import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import os

df = pd.read_csv("./BAA/rsna-bone-age/boneage-training-dataset/boneage-training-dataset.csv")

data_aug = tf.keras.Sequential([
   layers.RandomRotation(0.2),
   layers.RandomZoom(0.2, 0.2)
 ]) 

path = "./BAA/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/"
new_path = "./BAA/rsna-bone-age/boneage-training-dataset/boneage-training-dataset2/"
new_data = []
for i in range(0, len(df), 2):
    new_row = dict()
    image_path = f"{path}{df.id[i]}.png"
    new_row['id'] = 100000+df.id[i]
    new_row['boneage'] = df.boneage[i]
    new_row['male'] = df.male[i]
    img = tf.keras.utils.load_img(image_path)
    aug_img = data_aug(img)
    tf.keras.utils.save_img(f"{new_path}{df.id[i]+100000}.png", aug_img)
    new_data.append(new_row)
      
for i in range(0, len(df)):
    image_path = f"{path}{df.id[i]}.png"
    img = tf.keras.utils.load_img(image_path)
    tf.keras.utils.save_img(f"{new_path}{df.id[i]}.png", img)

df2 = pd.DataFrame(new_data)
final_df = pd.concat([df, df2])
final_df.to_csv('./BAA/rsna-bone-age/boneage-training-dataset/boneage-training-dataset2.csv', index=False, header=True)
