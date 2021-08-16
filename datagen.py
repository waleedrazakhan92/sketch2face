from glob import glob
import random
import numpy as np
import cv2
import os

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

def get_paths(path_dataset):
    all_paths = glob(path_dataset+'/*')
    return all_paths


def load_and_preprocess(img_path, out_shape):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img,(out_shape[0],out_shape[1]))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = load_img(img_path, target_size=(out_shape[0],out_shape[1]))
    img = img_to_array(img)
    img = (img - 127.5) / 127.5
    return img

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape[0], patch_shape[1], 1))
    return X, y

# def my_datagen(all_paths, path_labels, patch_shape ,batch_size=64, out_shape=(256,256,3)):
    
#     while True: 
#         selected_paths = random.sample(all_paths, batch_size)
        
#         batch_images = []
#         batch_labels = []
#         for i in range(0,len(selected_paths)):
#             img_name = selected_paths[i].split('/')[-1]
#             lab_path = path_labels + img_name[:-4] +'_map_'+ img_name[-4:]
          
#             img = load_and_preprocess(selected_paths[i], out_shape=out_shape)
#             lab_img = load_and_preprocess(lab_path, out_shape=out_shape)
            
#             batch_images.append(img)
#             batch_labels.append(lab_img)
            
#         y = np.ones((batch_size, patch_shape[0], patch_shape[1], 1))
#         yield np.array(batch_images), np.array(batch_labels), np.array(y)

def my_datagen(path_dataset, path_labels, patch_shape, batch_size=64, out_shape=(512,512,3)):
    
    while True: 
        all_paths = get_paths(path_dataset)
        selected_paths = random.sample(all_paths, batch_size)
        
        batch_images = []
        batch_labels = []
        for i in range(0,len(selected_paths)):
            
            img_name = selected_paths[i].split('/')[-1]
            lab_name = path_labels+'/'+img_name

            if os.path.isfile(lab_name):
            

                img = load_and_preprocess(selected_paths[i], out_shape)
                lab = load_and_preprocess(lab_name, out_shape)
                
                batch_images.append(np.array(img))
                batch_labels.append(np.array(lab))
            else:
                print('no file ',lab_name)
        
        y = np.ones((batch_size, patch_shape[0], patch_shape[1], 1))
        yield np.array(batch_images), np.array(batch_labels), np.array(y)
