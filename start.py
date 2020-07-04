import segmentation_models as sm
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas
import numpy as np
import pickle

SM_FRAMEWORK = keras

sm.set_framework('keras')

keras.backend.set_image_data_format('channels_last')

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

DATADIR1 = "/home/zain/Downloads/Thesis/unet-master/data/membrane/train/image"
DATADIR2 = "/home/zain/Downloads/Thesis/unet-master/data/membrane/train/label"
DATADIR3 = "/home/zain/Downloads/Thesis/unet-master/data/membrane/train/validation"
DATADIR4 = "/home/zain/Downloads/Thesis/unet-master/data/membrane/train/validationlabel"

IMG_SIZE = 320


x_train = []
y_train = []
x_val = []
y_val = []

for id, img in enumerate(os.listdir(DATADIR1)):
    img_arr = cv2.imread(os.path.join(DATADIR1, img), cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    x_train.append([new_arr])

for id, img in enumerate(os.listdir(DATADIR2)):
    img_arr = cv2.imread(os.path.join(DATADIR2, img), cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    y_train.append([new_arr])

for id, img in enumerate(os.listdir(DATADIR3)):
    img_arr = cv2.imread(os.path.join(DATADIR3, img), cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    x_val.append([new_arr])

for id, img in enumerate(os.listdir(DATADIR4)):
    img_arr = cv2.imread(os.path.join(DATADIR4, img), cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    y_val.append([new_arr])

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_val = np.array(x_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_val = np.array(y_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(y_train)

with open('abc.pkl', 'wb') as f:
    pickle.dump(y_train, f)


# plt.imshow(x_train[0], cmap='gray')
# plt.show()


model = sm.Unet()
model = sm.Unet(BACKBONE, input_shape=(
    None, None, 1), encoder_weights=None, activation='softmax')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

results = model.fit(
    x=x_train,
    y=y_train,
    batch_size=5,
    epochs=5,
    validation_data=(x_val, y_val),
)


testpath = '/home/zain/Downloads/Thesis/unet-master/data/membrane/test/0.png'
img = cv2.imread(testpath, cv2.IMREAD_COLOR)
test = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

test = np.array(test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

prediction = model.predict(test)

print(prediction)

with open('abc.pkl', 'wb') as f:
    pickle.dump(prediction, f)
