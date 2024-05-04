import pandas as pd;
import numpy as np;
import os
from skimage.transform import resize
from imageio import imread
import datetime
import os
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings('ignore')

class Generator:
    '''
    Class to generator the data for the model training
    '''

    def __init__(self, folder_path, train_dir, val_dir, train_flag, batch_size, imgTensor, augmentation):
        '''
        Initialization of different parameters for custom data preparation
        '''

        self.folder_path = folder_path

        self.train_dir = train_dir
        self.val_dir = val_dir

        self.train_flag = train_flag

        self.train_file = os.path.join(self.folder_path, "input","train.csv")
        self.val_file = os.path.join(self.folder_path, "input","val.csv")

        #self.train_doc = np.random.permutation(open(self.train_file)).readlines()
        #self.val_doc = np.random.permutation(open(self.val_file)).readlines()

        if (self.train_flag):
            self.source_path = self.train_dir
            self.folder_list = np.random.permutation(open(self.train_file).readlines())
        else:
            self.source_path = self.val_dir
            self.folder_list = np.random.permutation(open(self.val_file).readlines())

        self.batch_size = batch_size
        self.img_tensor = imgTensor
        self.augmentation = augmentation

    def cropResize(self, image, y, z):
        h, w = image.shape
    
        # if smaller image crop at center for 120x120
        if w == 160:
            image = image[:120, 20:140]

        # resize every image
        return resize(image, (y,z))
    
    def normalizeImage(self, image):
        # applying normalization
        return image/255.0  
    
    def preprocessImage(self,image, y, z):
        return self.normalizeImage(self.cropResize(image, y, z))
    
    def make3dFilter(self,x):
        return tuple([x]*3)

    def make2dFilter(self, x):
        return tuple([x]*2)
    
    def generator(self):
        '''
        Function to generate the data for model
        '''

        print( 'Source path = ', self.source_path, '; batch size =', self.batch_size)
        while True:
            #print(self.folder_list)
            t = np.random.permutation(self.folder_list)
            num_batches = int(len(self.folder_list)/self.batch_size)

             # we iterate over the number of batches
            for batch in range(num_batches):
                yield self.getBatchData(t, batch)
            
            # write the code for the remaining data points which are left after full batches
            # checking if any remaining batches are there or not
            if len(self.folder_list)%self.batch_size != 0:
                # updated the batch size and yield
                self.batch_size = len(self.folder_list)%self.batch_sizebatch_size
                yield self.getBatchData(t, batch)

    
    def getBatchData(self, t, batch):

        img_tensor = self.img_tensor

        [x,y,z] = [len(img_tensor[0]),img_tensor[1], img_tensor[2]]
        img_idx = img_tensor[0]
        batch_data = np.zeros((self.batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
        batch_labels = np.zeros((self.batch_size,5)) # batch_labels is the one hot representation of the output
        
        if (self.augmentation): 
            batch_data_aug = np.zeros((self.batch_size,x,y,z,3))
        
        for folder in range(self.batch_size): # iterate over the batch_size
            imgs = os.listdir(self.source_path+'/'+ t[folder + (batch*self.batch_size)].split(';')[0]) # read all the images in the folder
            for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                image = imread(self.source_path+'/'+ t[folder + (batch*self.batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                #crop the images and resize them. Note that the images are of 2 different shape 
                #and the conv3D will throw error if the inputs in a batch have different shapes

                # separate preprocessImage function is defined for cropping, resizing and normalizing images
                batch_data[folder,idx,:,:,0] = self.preprocessImage(image[:, :, 0], y, z)
                batch_data[folder,idx,:,:,1] = self.preprocessImage(image[:, :, 1], y, z)
                batch_data[folder,idx,:,:,2] = self.preprocessImage(image[:, :, 2], y, z)
                if (self.augmentation):
                    shifted = cv2.warpAffine(image, 
                                        np.float32([[1, 0, np.random.randint(-30,30)],[0, 1, np.random.randint(-30,30)]]),
                                        (image.shape[1], image.shape[0]))
                        
                    gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)

                    x0, y0 = np.argwhere(gray > 0).min(axis=0)
                    x1, y1 = np.argwhere(gray > 0).max(axis=0) 
                    # cropping the images to have the targeted gestures and remove the noise from the images.
                    cropped=shifted[x0:x1,y0:y1,:]
                    image_resized=resize(cropped,(y,z,3))
                        
                    #shifted = cv2.warpAffine(image_resized, 
                    #np.float32([[1, 0, np.random.randint(-3,3)],[0, 1, np.random.randint(-3,3)]]), 
                    #(image_resized.shape[1], image_resized.shape[0]))
                
                    batch_data_aug[folder,idx,:,:,0] = (image_resized[:,:,0])/255
                    batch_data_aug[folder,idx,:,:,1] = (image_resized[:,:,1])/255
                    batch_data_aug[folder,idx,:,:,2] = (image_resized[:,:,2])/255

            batch_labels[folder, int(t[folder + (batch*self.batch_size)].strip().split(';')[2])] = 1
            
            if (self.augmentation):
                batch_data=np.concatenate([batch_data,batch_data_aug])
                batch_labels=np.concatenate([batch_labels,batch_labels])
        return batch_data, batch_labels