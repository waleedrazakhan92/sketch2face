from models import *
from datagen import *
from utils import *
from config import *
import os


# train pix2pix models
def train(d_model, g_model, gan_model, path_images, path_labels, n_epochs=100, batch_size=1, path_results='model_performance/', out_shape=(512,512,3)):
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    
    # determine the output square shape of the discriminator
    n_patch = (d_model.output_shape[1],d_model.output_shape[2])
    all_paths = get_paths(path_images)
    bat_per_epo = int(len(all_paths)/batch_size)
    n_steps = bat_per_epo * n_epochs
    #n_steps = 5
    print('Batch per epochs = ', bat_per_epo)
    print('Total Steps = ', n_steps)
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, X_realB, y_real = next(my_datagen(path_images, path_labels, patch_shape = n_patch, batch_size=batch_size,out_shape=out_shape))
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, patch_shape=n_patch)
        
       
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if i%(bat_per_epo*4)==0 and i>1:
            print('---------------------------------------')
            print('Saving and evaluating model...')
            g_model.save(path_results+'g_model.h5')
            d_model.save(path_results+'d_model.h5')
            gan_model.save(path_results+'gan_model.h5')
            # select a sample of input images
            X_realA, X_realB, _ = next(my_datagen(path_images, path_labels, patch_shape=n_patch, batch_size=3,out_shape=out_shape))
            # generate a batch of fake samples
            X_fakeB, _ = generate_fake_samples(g_model, X_realA, patch_shape=n_patch)
            summarize_performance(path_results, i, g_model,  X_realA, X_realB, X_fakeB, n_samples=3)




d_model, g_model, gan_model = define_gan(image_shape)
d_model.load_weights(path_results+'d_model.h5')
g_model.load_weights(path_results+'g_model.h5')
gan_model.load_weights(path_results+'gan_model.h5')

train(d_model, g_model, gan_model, path_images, path_labels,batch_size=batch_size, path_results=path_results, out_shape=image_shape)
