import matplotlib.pyplot as plt

class PlotShowData:

    def plotSampleImage(self, sample_batch_data, sample_batch_labels, train_flag):
        '''
        Function to show sample images from batch/validation data'''
        if(train_flag == 2):
            image_count = 1
        else:
            image_count = 6
        for cnt in range(image_count):
            # plot generated sample images
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(sample_batch_data[cnt,10,:,:,:])  
            ax[0].title.set_text(f"label - {sample_batch_labels[cnt,:]}") 
            ax[1].imshow(sample_batch_data[cnt,10,:,:,:])
            ax[1].title.set_text(f"label - {sample_batch_labels[cnt,:]}") 
            plt.show()

    def plotModelHistory(self, h, name):
        '''
        Function to plot the model history details
        '''
        fig, ax = plt.subplots(1, 2, figsize=(15,4))
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('No. of Epochs')
        ax[0].plot(h.history['loss'])   
        ax[0].plot(h.history['val_loss'])
        ax[0].legend(['loss','val_loss'])
        ax[0].title.set_text(f"Train loss vs Validation loss - {name}")

        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('No. of Epochs')
        ax[1].plot(h.history['categorical_accuracy'])   
        ax[1].plot(h.history['val_categorical_accuracy'])
        ax[1].legend(['categorical_accuracy','val_categorical_accuracy'])
        ax[1].title.set_text(f"Train accuracy vs Validation accuracy - {name}")
        plt.savefig(f'report/acc_loss_{name}.png')
        plt.show()

        #print("Training Accuracy", h.history['categorical_accuracy'][-1])
        #print("Validaiton Accuracy", h.history['val_categorical_accuracy'][-1])