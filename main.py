import pandas as pd;
import numpy as np;
import os
from skimage.transform import resize
from imageio import imread
import datetime
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from Generator import Generator
from ImageTensor import ImageTensor
from PlotShowData import PlotShowData
from CNN import CNN

def main(train_flag):

    # get the Image Tensor
    imgTensor = ImageTensor(n_frames=n_frames)

    image_tensor = imgTensor.getImgTensor()

    print ('# img_tensor =', image_tensor)

    ######### test code to check if generator works fine  #############
    # generator = Generator(folder_path, train_dir, val_dir, train_flag, batch_size=20, imgTensor=image_tensor, augmentation=augmentation)
    # sample_generator = generator.generator()

    # sample_batch_data, sample_batch_labels = next(sample_generator)

    # print(len(sample_batch_data), len(sample_batch_labels))
    # print(sample_batch_data.shape, sample_batch_labels.shape)
    # print("****Batch Data****")
    # print(sample_batch_data[:5])
    # print("\n****Batch labels****")
    # print(sample_batch_labels[:5])

    # pltShow = PlotShowData()
    # pltShow.plotSampleImage(sample_batch_data)

    ############ End of test code ####################
    
    # initialize the training and validation data
    generator_trn = Generator(folder_path, train_dir, val_dir, train_flag, batch_size=batch_size, imgTensor=image_tensor, augmentation=augmentation)
    train_generator = generator_trn.generator()
    #num_train_sequences = generator_trn.no_of_sequence

    generator_val = Generator(folder_path, train_dir, val_dir, False, batch_size=batch_size, imgTensor=image_tensor, augmentation=augmentation)
    val_generator = generator_val.generator()
    #num_val_sequences = generator_val.no_of_sequence

    # compute the training an validation steps
    steps_per_epoch = generator_trn.calculateSteps()
    validation_steps = generator_val.calculateSteps()


    # initialize and compile the model
    model = CNN(image_tensor)
    model.initializeModel()
    model.compileModel()

    # set up the callback for the model fit
    callback_list = model.callbackSetup()

    # fit the model
    model.fitModel(train_generator=train_generator, steps_per_epoch=steps_per_epoch,num_epochs=num_epocs, callbacks_list=callback_list, 
                   val_generator=val_generator, validation_steps=validation_steps )

    pltShow = PlotShowData()
    pltShow.plotModelHistory(model.history)


if __name__ == '__main__':
    folder_path = os.getcwd() 
    #print(folder_path)
    
    train_dir = os.path.join(folder_path, "input", "train")
    val_dir = os.path.join(folder_path, "input", "val")

    # print(os.path.join(val_dir,"val.csv"))
    augmentation = False
    n_frames = 20
    batch_size = 20
    num_epocs = 20

    main(train_flag=False)