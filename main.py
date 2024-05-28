# import pandas as pd;
# import numpy as np;
# import os
# from skimage.transform import resize
# from imageio import imread
# import datetime
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
# from Generator import Generator
# from ImageTensor import ImageTensor
# from PlotShowData import PlotShowData
# from Report import showReport
# from CNN import CNN
# warnings.filterwarnings("ignore")
# from sklearn.metrics import classification_report, confusion_matrix

# def main(train_flag):

#     # get the Image Tensor
#     imgTensor = ImageTensor(n_frames=n_frames)

#     image_tensor = imgTensor.getImgTensor()

#     print ('# img_tensor =', image_tensor)

#     ######### test code to check if generator works fine  #############

#     # generator = Generator(folder_path, train_dir, val_dir, train_flag, batch_size=20, imgTensor=image_tensor, augmentation=augmentation)
#     # sample_generator = generator.generator()

#     # sample_batch_data, sample_batch_labels = next(sample_generator)

#     # print(len(sample_batch_data), len(sample_batch_labels))
#     # print(sample_batch_data.shape, sample_batch_labels.shape)
#     # print("****Batch Data****")
#     # #print(sample_batch_data[:1])
#     # print("\n****Batch labels****")
#     # print(sample_batch_labels[:5])

#     # pltShow = PlotShowData()
#     # pltShow.plotSampleImage(sample_batch_data, sample_batch_labels)

#     ############ End of test code ####################

#     if train_flag:
#         # initialize the training and validation data
#         generator_trn = Generator(folder_path, train_dir, val_dir, train_flag, batch_size=batch_size, imgTensor=image_tensor, augmentation=augmentation)
#         train_generator = generator_trn.generator()
#         #num_train_sequences = generator_trn.no_of_sequence

#         generator_val = Generator(folder_path, train_dir, val_dir, False, batch_size=batch_size, imgTensor=image_tensor, augmentation=augmentation)
#         val_generator = generator_val.generator()
#         #num_val_sequences = generator_val.no_of_sequence

#         # compute the training and validation steps
#         steps_per_epoch = generator_trn.calculateSteps()
#         validation_steps = generator_val.calculateSteps()


#         # initialize and compile the model
#         model = CNN(image_tensor)
#         model.initializeModel(modelName)
#         model.compileModel()

#         model.model.summary()

#         # set up the callback for the model fit
#         callback_list = model.callbackSetup()

#         # fit the model
#         model.fitModel(train_generator=train_generator, steps_per_epoch=steps_per_epoch,num_epochs=num_epocs, callbacks_list=callback_list, 
#                        val_generator=val_generator, validation_steps=validation_steps )

#        # model.
#         pltShow = PlotShowData()
#         pltShow.plotModelHistory(model.history,modelName)

#         #predictions = model.predictModel(val_generator)

#     else:
#         generator_val = Generator(folder_path, train_dir, val_dir, False, batch_size=batch_size, imgTensor=image_tensor, augmentation=augmentation)
#         val_generator = generator_val.generator()
#         validation_steps = generator_val.calculateSteps()

#         #model = CNN.loadModel('/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/model_init_2024-05-1522_45_47.146370/model-keras')
#         #model = CNN.loadModel('/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/model_init_2024-05-2316_56_40.851725/model-keras')
#         model = CNN.loadModel('/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/model_init_2024-05-2423_33_16.378358/model-keras.keras')

#         model.summary()

#         # Create empty lists to store the actual and predicted labels
#         actual_labels = []
#         predicted_labels = []

#         # Iterate over the generator to obtain the actual labels
#         for i in range(validation_steps):
#             # Get the next batch of images and labels from the generator
#             batch_x, batch_y = next(val_generator)
    
#             # Convert one-hot encoded labels to class indices
#             actual_labels.extend(np.argmax(batch_y, axis=1))
    
#             # Make predictions for the batch
#             predictions = model.predict(batch_x)
    
#             # Convert predicted probabilities to class indices
#             predicted_labels.extend(np.argmax(predictions, axis=1))

#         # Convert the lists to NumPy arrays for compatibility with confusion_matrix
#         actual_labels = np.array(actual_labels)
#         predicted_labels = np.array(predicted_labels)

#         # show the classification report and confusion matrix
#         showReport(actual_labels, predicted_labels, modelName)


# if __name__ == '__main__':
#     folder_path = os.getcwd() 
#     #print(folder_path)
    
#     train_dir = os.path.join(folder_path, "input", "train")
#     val_dir = os.path.join(folder_path, "input", "val")

#     # print(os.path.join(val_dir,"val.csv"))
#     augmentation = False
#     n_frames = 30
#     batch_size = 30
#     num_epocs = 30
#     modelName = 'MODEL2'

#     main(train_flag=False)