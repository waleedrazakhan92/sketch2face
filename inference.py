from models import *
from datagen import *
from utils import *
from config import *
import os
import numpy as np

import matplotlib.pyplot as plt

# generate samples and save as a plot and save the model
def my_inference(path_inference, g_model, X_realA, X_realB, X_fakeB, n_samples):

    n_samples = len(X_realA)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        #plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        #plt.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        #plt.imshow(X_realB[i])
    
    

    

def main():
    if not os.path.isdir(path_inference):
        os.mkdir(path_inference)

    d_model, g_model, gan_model = define_gan(image_shape)
    #d_model.load_weights(path_results+'d_model.h5')
    g_model.load_weights(path_results+'g_model.h5')
    #gan_model.load_weights(path_results+'gan_model.h5')

    n_patch = (d_model.output_shape[1],d_model.output_shape[2])
    X_realA, X_realB, _ = next(my_datagen(path_images, path_labels, patch_shape=n_patch, batch_size=infer_batch_size, out_shape=image_shape))
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, patch_shape=n_patch)
    #my_inference(path_inference, g_model,  X_realA, X_realB, X_fakeB, n_samples=len(X_realA))

    for i in range(0,len(X_realA)):
        img_infer = np.concatenate((X_realA[i],X_realB[i], X_fakeB[i]),axis=1)
        img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_inference+str(i)+'.png', img_infer*255)

if __name__=="__main__":
    main()
