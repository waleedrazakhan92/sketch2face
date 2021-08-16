import matplotlib.pyplot as plt

# generate samples and save as a plot and save the model
def summarize_performance(path_results, step, g_model, X_realA, X_realB, X_fakeB, n_samples):

    n_samples = len(X_realA)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i])
    # save plot to file
    filename1 = path_results+'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    #filename2 = path_results+'model_%06d.h5' % (step+1)
    filename2 = path_results+'model.h5'
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

