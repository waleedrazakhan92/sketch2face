from models import *
from datagen import *
from utils import *
from config import *
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datagen import load_and_preprocess

if not os.path.isdir(path_inference):
        os.mkdir(path_inference)

d_model, g_model, gan_model = define_gan(image_shape)
#d_model.load_weights(path_results+'d_model.h5')
g_model.load_weights(path_results+'g_model.h5')
#gan_model.load_weights(path_results+'gan_model.h5')

n_patch = (d_model.output_shape[1],d_model.output_shape[2])
# X_realA, X_realB, _ = next(my_datagen(path_images, path_labels, patch_shape=n_patch, batch_size=infer_batch_size, out_shape=image_shape))

X_realA = []
test_img = load_and_preprocess('test_img.png',image_shape)

X_realA.append(test_img)
print(np.shape(X_realA))
X_fakeB, _ = generate_fake_samples(g_model, np.array(X_realA), patch_shape=n_patch)
#my_inference(path_inference, g_model,  X_realA, X_realB, X_fakeB, n_samples=len(X_realA))

for i in range(0,len(X_realA)):
    img_infer = np.concatenate((X_realA[i], (X_fakeB[i]+1.0)/2.0),axis=1)
    img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path_inference+str(i)+'.png', img_infer*255)

cv2.imshow('results', img_infer)
cv2.waitKey(0)
cv2.destroyAllWindows()
