# sketch2face
An extension of https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation. 
The theme of the project is as follows:
1. The user draws a sketch of a human face using mouse as an input.
2. That input sketch is taken as an input to the pix2pix GAN.
3. The GAN then outputs a synthetic human face based on the input sketch.

# Dataset
The prepration of dataset is done by first generating synthetic faces using StyleGAN2-ada https://github.com/NVlabs/stylegan2-ada. The checkpoint that I used for face generation is ffhq.pkl. The checkpoint can be obtained from https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/. 
Clone the stylegan2-ada repository and generate as many faces as you like using the ffhq.pkl checkpoint. 
Once the faces are generated put them in a folder and use generate_face_edges.ipynb to generate edge images of the correspinding faces. 
Set the path_labels variable to the face images folder and set the path_write to the path you want to store the edge images.
Once the file is run a folder with the edge images is created. And now the pix2pix model is ready to train.

# Training
