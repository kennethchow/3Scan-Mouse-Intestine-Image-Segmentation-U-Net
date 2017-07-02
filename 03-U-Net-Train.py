from __future__ import print_function
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras import losses, metrics


data_path = '/home/RGD/cleannpy/normalized/'
img_rows = 256
img_cols = 256
smooth = 1.
nb_epoch = 5
batch_size = 8

#face/mask, train/valid lists
trainimgs = [data_path+'train/imgs/'+x for x  in sorted(os.listdir(data_path+'train/imgs/'))]
trainmasks = [data_path+'train/masks/'+x for x  in sorted(os.listdir(data_path+'train/masks/'))]
validimgs = [data_path+'validation/imgs/'+x for x  in sorted(os.listdir(data_path+'validation/imgs/'))]
validmasks = [data_path+'validation/masks/'+x for x  in sorted(os.listdir(data_path+'validation/masks/'))]
holdoutimgs = [data_path+'holdout/imgs/'+x for x  in sorted(os.listdir(data_path+'holdout/imgs/'))]
holdoutmasks = [data_path+'holdout/masks/'+x for x  in sorted(os.listdir(data_path+'holdout/masks/'))]


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jacc_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)
                
def generator2(img_files, truth_files, batch_size):

    batch_images = np.zeros((batch_size, img_rows,img_cols,3))
    batch_truth = np.zeros((batch_size, img_rows,img_cols,3))

    while True:
        for i in range(batch_size):
            index = int(np.random.choice(len(img_files),1))
            batch_images[i] = np.load(img_files[index])
            batch_truth[i] = np.load(truth_files[index])
            yield (batch_images, batch_truth)

def get_unet():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    dropout1 = Dropout(0.5)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(dropout1)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)    
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis = 1)(conv9)
    dropout2 = Dropout(0.5)(conv9)
    
    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(dropout2)
    
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5,beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0), 
                  loss=dice_coef_loss, metrics=[dice_coef,jacc_coef_int])

    return model

print('-'*30)
print('Loading Train and Validation Generator...')
print('-'*30)

traingenerator = generator2(trainimgs,trainmasks,8)
validgenerator = generator2(validimgs,validmasks,8)

print('-'*30)
print('Creating and compiling model...')
print('-'*30)
model = get_unet()
model_checkpoint = ModelCheckpoint(filepath='weights.h5', monitor='val-loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, epsilon=0.002, cooldown=2)
csvlogger = CSVLogger('training.log')

print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit_generator(traingenerator,epochs=nb_epoch,steps_per_epoch=512,verbose=1, validation_data=validgenerator,
                    validation_steps=256,callbacks=[model_checkpoint,reduce_lr,csvlogger])

model.save_weights('/home/RGD/3Scan_Image_Segmentation/modelweights.h5')

model.save('/home/RGD/3Scan_Image_Segmentation/fullmodel.h5')
