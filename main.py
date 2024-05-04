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

def main(train_flag):

    # get the Image Tensor
    n_frames = 20
    imgTensor = ImageTensor(n_frames=n_frames)

    image_tensor = imgTensor.getImgTensor()

    print ('# img_tensor =', image_tensor)
    
    if train_flag:

        generator = Generator(folder_path, train_dir, val_dir, train_flag, batch_size=20, imgTensor=image_tensor, augmentation=augmentation)
        sample_generator = generator.generator()
        # print(sample_generator)

        sample_batch_data, sample_batch_labels = next(sample_generator)

        print(len(sample_batch_data), len(sample_batch_labels))
        # print("****Batch Data****")
        # print(sample_batch_data[:5])
        # print("\n****Batch labels****")
        # print(sample_batch_labels[:5])


if __name__ == '__main__':
    folder_path = os.getcwd() 
    #print(folder_path)
    
    train_dir = os.path.join(folder_path, "input", "train")
    val_dir = os.path.join(folder_path, "input", "val")

    # print(os.path.join(val_dir,"val.csv"))
    augmentation = False
    main(train_flag=True)