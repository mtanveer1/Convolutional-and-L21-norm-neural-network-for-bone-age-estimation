# BoneAge

Download the data from the [link](https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739).

Download the weights of InceptionV3 of imagenet using the following [link](https://github.com/kohpangwei/influence-release/blob/master/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5) in the same directory as that of the code file.

Make the data directory tree as follows:
- BAA
    - features
    - models
    - rsna-bone-age
        - boneage-training-dataset
            - boneage-training-dataset (containing 12616 images)
                - (12616 images)
            - boneage-training-dataset2 (containing 18917 images)
                - (18917 images)
            - boneage-training-dataset.csv
            - boneage-training-dataset2.csv
        - boneage-validation-dataset
            - boneage-validation-dataset (containing 1425 images)
                - (1245 images)
            - boneage-validation-dataset.csv
        - boneage-testing-dataset
            - boneage-testing-dataset (containing 200 images)
                - (200 images)
            - boneage-testing-dataset.csv


### Augmented data 
For Augmented data generation run the file data_augment.py. This will add 18k images to the boneage-training-dataset2 and generate the corresponding csv file boneage-training-dataset2.csv.

---

## Boneage Prediction

### 1. Winner Architecture (uses just the original images) (python env: 3.7 and tf-version==1.5)
run the Winner_architecture.py. The best weights will be saved as **bone_age_weights_16bit.best.hdf5**
For evaluation refer to the evaluation section.

---

## Features Generation of the penultimate (1000 dense) layer
1. run Features_extraction.py to generate the csv containing features. (python env: 3.7 and tf-version==1.5)
2. run features_to_format.py to convert to the required format (image_id , (1000 columns of features) , target)

---

## Evaluation of the Model
1. Winner_architecture.py (python env: 3.7 and tf-version==1.5) : To evaluate this model architecture on Testing(200 images) and Validation(1425 images) run Winner_architecture_evaluate.py. This will generate csv files (test.csv and val.csv) with the corresponding predictions.

---
