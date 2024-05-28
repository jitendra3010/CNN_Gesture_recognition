import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout, LSTM
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.applications import mobilenet
import datetime
import os

class CNN:
    def __init__(self, img_tensor):
        # specify the input shape for the image tensor
        self.input_shape = (len(img_tensor[0]), img_tensor[1], img_tensor[2], img_tensor[3])
        self.optimizer = optimizers.legacy.Adam()
        self.loss = 'categorical_crossentropy'
        self.metrics = ['categorical_accuracy']
        self.curr_dt_time = datetime.datetime.now()
        self.mobilenet = mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=self.input_shape[1:])
        


    def initializeModel(self , modelName):
        '''
            Model1:
            ***************
            First layer is 16 filter with each filter shape as 3D filter with relu activaiton 
            We also use a maxpooling with a 3D filter of 2,2,2 with same padding
            we also normalize the data here

            second layer we use 32 filter and other details same 
            but the pooling we use a pool size of 1,2,2
            with normalization

            third layer is again 64 filters with 3 filters as 3,3,3 having relu activation 
            max pooling and batch normalizaiton

            We then flatten the output of the 3rd layer to a dence layer of 128 filters , relu activation 
            having batch normalizatiom , we use a droput of 0.25 to combat overfitting

            the above step is again repeated in one more layer haing 64 filters

            and then the final output layer of 5 filter with softmax activation

            Model2:
            **********
            Model 2 is using a Trasfer Learning mobilenet with LSTM
            We perform BatchNormalization , maxpooling with a 2D filter
            Next layer we flatten them pass them to LSTM layer, we do a dropout of 0.2
            and then the ouput layer is a dense layer with softmax activaiton

        '''
        if modelName.upper() == 'MODEL1': 
            self.model = Sequential([
                Conv3D(16, self.make3dFilter(5), activation='relu', input_shape=self.input_shape),
                MaxPooling3D(self.make3dFilter(2), padding='same'),
                BatchNormalization(),

                Conv3D(32, self.make3dFilter(3), activation='relu'),
                MaxPooling3D(pool_size=(1,2,2), padding='same'),
                BatchNormalization(),

                Conv3D(64, self.make3dFilter(3), activation='relu'),
                MaxPooling3D(pool_size=(1,2,2), padding='same'),
                BatchNormalization(),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.25),

                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.25),

                Dense(5, activation='softmax')
            ])
        elif modelName.upper() == 'MODEL2': 
            
            ##### model using a mobilenet with LSTM
            # with a layer of timedistributed batch normalization
            # maxpooling 2D with filter of 2
            # flatten the data in next layer 
            # LSTM in next layer iwth dropout of 0.2
            # then output layer with 5 neuron , activation as softmax
            self.model = Sequential([TimeDistributed(self.mobilenet, input_shape=self.input_shape)], name="mobilenet_lstm")
            self.model.add(TimeDistributed(BatchNormalization()))
            self.model.add(TimeDistributed(MaxPooling2D(self.make2dFilter(2))))
            self.model.add(TimeDistributed(Flatten()))
            self.model.add(LSTM(256))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(5, activation='softmax'))



    def compileModel(self):
        # compile the model
        self.model.compile(optimizer= self.optimizer, loss=self.loss, metrics=self.metrics)

    def fitModel(self, train_generator, steps_per_epoch, num_epochs, callbacks_list, val_generator, validation_steps):
        '''
        function to fit the model
        '''

        self.history = self.model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                                      verbose=1, callbacks=callbacks_list, validation_data=val_generator,
                                      validation_steps=validation_steps, class_weight=None, initial_epoch=0)
    
    def loadModel(filepath):

        return load_model(filepath)
        
    def predictModel(self, test_generator):
        '''Funciton to predict the test data'''

        return self.model.predict(test_generator, steps=len(test_generator), verbose=1)

    def make3dFilter(self,x):
        '''
        function to return a tuple of 3 dimension
        '''
        return tuple([x]*3)

    def make2dFilter(self, x):
        '''
        function to return a tuple of 2 dimension
        '''
        return tuple([x]*2)
    
    def callbackSetup(self, modelName):
        '''
        Function to set the call back setup
        '''
        # model name 
        model_name = modelName + '_' + str(self.curr_dt_time).replace(' ','').replace(':','_') + '/'
        
        # if do not exist create one
        if not os.path.exists(model_name):
            os.mkdir(model_name)

        # filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'
        filepath = model_name + 'model-keras'

        # ModelCheckpoint is a callback function in TensorFlow/Keras used for saving the model's weights during training. 
        # It allows you to save the model's parameters to disk at certain intervals, such as at the end of each epoch or 
        # when the model achieves the best performance on a validation set.
        #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

        # ReduceLROnPlateau is a callback function in TensorFlow/Keras used for reducing the learning rate when a metric has stopped improving. 
        # This can help to fine-tune the learning process and improve convergence when training deep learning models.
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=4)

        # as of now lets use the reducedLR as the call back function
        callbacks_list = [checkpoint, LR]
        #callbacks_list = [checkpoint]

        return callbacks_list