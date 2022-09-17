# fashion-generator
This is the code repository of our final project as part of CSE-676 Deep Learning Course

# Objective
With the advent of the Big Data era, approach to fashion designing has changed. Fashion style cannow be analyzed by machine learning tools from images and textual descriptions of clothing. It is possible to mine attributes of clothing through deep learning. These attributes are then utilized to generate outfit designs. The improvement of computer vision architectures, particularly that of Convolutional Neural Networks (CNN) have given rise to multiple applications of deep learning
in the field of fashion. For instance, we have seen fashion-related convolutional neural network implementations such as clothing description [6] and fashion apparel detection. However, there seems to be limited work in fashion design generation.We propose to build an AI-powered fashion design generator by training a Generative Adversarial Network (GAN). At each training iteration, the generator (G) generates a set of images and tries to make them as realistic as possible. Simultaneously, the discriminator (D) tries to determine whethereach of G’s images are real or not. As both G and D train against each other, the generated imagesshould become more and more realistic. We intend to make a fashion design generator that adapts to consumer trends. We will need to be able to input parameters that modify our output. We will be
able to do this with a Conditional GAN (CGAN) [8]. With CGAN, we can feed in data to condition both the Discriminator and Generator to design creative fashion designs For our AI-powered fashion generator, we will be trying out models of varying complexity starting
from simple vanilla GANs to Deep Convolutional GANs [11]. We envision a conditional DCGAN.


# Architecture
![Conditional DCGAN](https://user-images.githubusercontent.com/99614234/190838961-f5122cfe-0d11-474b-9756-b985641036d3.PNG)

We upgrade the previously seen GAN models by adding convolutional layers in the architecture.The discriminator has 5 convolutional layers. The first layer takes in a leaky ReLU activation.The next three layers take a leaky ReLU activation function in tandem with a batch normalization layer. The final layer takes in a sigmoid activation function since the discriminator basically needs to differentiate between real and fake images. The generator also has 5 convolutional layers. The
first 4 layers use a ReLU activation in tandem with a batch normalization layer. The final layer is aconvolutional layer with a tanh activation function.
Since our training images are huge, we resized each of them to 64*64 before sending them into the model. This is because it is difficult for any GAN to converge when provided with large images.The architecture from the DCGAN paper (Figure 1) was mainly followed. Since the base code is based on MNIST dataset, the architecture was modified to incorporate inputs specific to the DeepFashion dataset

Model Results
![GAN images](https://user-images.githubusercontent.com/99614234/190838555-8076a8b4-f406-411b-84a7-58b124019d9f.PNG)

# Conclusion
The DeepFashion dataset produced promising results to build a powerful fashion generator model.The dataset is packed with high volume of images which are annotated and labelled properly. We saw promising results from the vanilla GANs and DCGANs in our experiments. The addition of the convolution operator in DCGAN substantially improved the model’s ability to learn the sharp edges of the images. The output from the DCGAN fashion generator is close to reality. For the future, the DCGAN needs to be implemented on a higher number of images and for higher number of epochs.Conditional DCGAN provide a very good usecase for fashion generators since they allow a user’s input as a label or an attribute. But the model had poor learning due to computing constraints. Future researchers should focus tuning the hyperparameters of the conditional DCGAN. The model offers a lot of promise since if follows the same architecture as the DCGAN.

# Instructions to run code
1. Download the DeepFashion Dataset from the official page. Pick the Category and Attribute Prediction Benchmark folder to download
2. The downloaded file(s) are compressed zip files.
3. Run the preprocess.py to unzip and preprocess the data. The preprocess.py file has a command to unzip a single file. The same line could be repeated for unzipping multiple files
4. The preprocess.py step also performing the feature extraction phase. That is the cloth coordinates are taken and the image is cropped with just the clothes, removing the extra noise.
5. The preprocess.py step also categorises the existing folder structure into a labelled folder structure - 'img_folder/label_name/img_file.png'. This is done so that it is easier for the ImageFolder() function to read and assign labels to each image
6. The preprocess.py function creates the cropped images as well as the labelled_imgs folder structure
7. This folder structure is used in the three models built as part of this experiment
8. Run the following .py files to execute each model
    a. vanilla_GAN.ipynb - To build a vanilla_GAN generator model that can generate new clothing ideas
    b. dcgan.py - To build a Deep Convolutional GAN generator model that can generate new clothing ideas
    c. cdcgan.py - To build a Conditional DCGAN generator model that can generate new clothing ideas

