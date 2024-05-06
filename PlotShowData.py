import matplotlib.pyplot as plt

class PlotShowData:

    def plotSampleImage(self, sample_batch_data):
        '''
        Function to show sample images from batch/validation data'''
        # plot generated sample images
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(sample_batch_data[16,10,:,:,:])   
        ax[1].imshow(sample_batch_data[19,10,:,:,:])
        plt.show()

    def plotModelHistory(h):
        '''
        Function to plot the model history details
        '''
        fig, ax = plt.subplots(1, 2, figsize=(15,4))
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('No. of Epochs')
        ax[0].plot(h.history['loss'])   
        ax[0].plot(h.history['val_loss'])
        ax[0].legend(['loss','val_loss'])
        ax[0].title.set_text("Train loss vs Validation loss")

        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('No. of Epochs')
        ax[1].plot(h.history['categorical_accuracy'])   
        ax[1].plot(h.history['val_categorical_accuracy'])
        ax[1].legend(['categorical_accuracy','val_categorical_accuracy'])
        ax[1].title.set_text("Train accuracy vs Validation accuracy")
        plt.show()

        print("Max. Training Accuracy", max(h.history['categorical_accuracy']))
        print("Max. Validaiton Accuracy", max(h.history['val_categorical_accuracy']))