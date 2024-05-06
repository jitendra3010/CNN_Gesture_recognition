from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout, LSTM
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers

class CNN:
    def __init__(self, img_tensor):
        # specify the input shape for the image tensor
        self.input_shape = (len(img_tensor[0]), img_tensor[1], img_tensor[2], img_tensor[3])
        self.optimizer = optimizers.Adam()
        self.loss = 'categorical_crossentropy'
        self.metrics = ['categorical_accuracy']


    def initializeModel(self):
        '''
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
        '''
        self.model = Sequential([
            Conv3D(16, self.make3dFilter(5), activation='relu', input_shape=self.inputShape),
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
    
    def compileModel(self):
        # compile the model
        self.model.compile(optimizer= self.optimizer, loss=self.loss, metrics=self.metrics)

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