import cv2
import time
from CNN import CNN
from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_class_label(pred_idx):
    labels = ['LeftSwipe', 'RightSwipe', 'Stop', 'ThumbsDown', 'ThumbsUp'] # Customize based on labels
    return labels[pred_idx]

def loadModel():# load the existing model
    model = CNN.loadModel(modelToLoad)
    model.summary()

    return model

def cropResize(image, y, z):
    h, w = image.shape

    # if smaller image crop at center for 120x120
    if w == 160:
        image = image[:120, 20:140]

    # resize every image
    return resize(image, (y,z))

def normalizeImage(image):
    # applying normalization
    return image/255.0  

def preprocessImage(image, y, z):
    return normalizeImage(cropResize(image, y, z))


def main():
    #print("Start*******")
    # load the model
    model = loadModel()

    #print("Model Load Done*******")
    #model.summary()

    #infer = model.signatures["serving_default"]

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # '0' is typically the default value for the webcam
 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    # Buffer to hold frames
    frame_buffer = []

    #c = 0
    # Read frames from the webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        #c +=1

 
        #print("Shape of frame::",frame.shape)
        #print(frame[:, :, 0])
        # frame_processed = np.zeros((100, 100 ,3))

        # frame_processed[: , :, 0] = preprocessImage(frame[: , :, 0], 100, 100)
        # frame_processed[: , :, 1] = preprocessImage(frame[: , :, 1], 100, 100)
        # frame_processed[: , :, 2] = preprocessImage(frame[: , :, 2], 100, 100)

        # Preprocess the frame
        frame_resized = cv2.resize(frame, input_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_processed = frame_rgb.astype(np.float32) / 255.0

        #print("Shape of Processed Frame:::", frame_processed.shape)
        # Convert frame to float32
        #frame_processed = frame_processed.astype(np.float32) / 255

        # Add the frame to the buffer
        frame_buffer.append(frame_processed)

        # Maintain only the last 20 frames
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)

        #rgb_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
        
        if len(frame_buffer) == sequence_length:
            input_sequence = np.array(frame_buffer)
            input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
            #input_tensor = tf.image.per_image_standardization(input_tensor)
            input_tensor = tf.expand_dims(input_tensor, axis=0)

            #print("*************************",input_tensor.shape)

            # Perform inference
            #predictions = infer(input_tensor)
            predictions = model.predict(input_tensor)
            #print(predictions)
            #output = predictions["dense"].numpy()
            output = predictions

            
            #print("******** Predicitons and prediction label***********")
            #print(output)
            label = np.argmax(output, axis=-1)
            #print("Label Data::", label)
            labelText = get_class_label(label[0])
            #print("Prediction Label:::",labelText)

            label_text = f"Prediction: {labelText}"
            cv2.putText(frame, label_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)


 
        # Display the resulting frame
        #cv2.imshow('Video', cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        cv2.imshow('Video', frame)
        
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    input_size = (128, 128)
    sequence_length = 30
    #modelToLoad = '/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/model_init_2024-05-1522_45_47.146370/model-keras'
    modelToLoad = '/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/MODEL2_2024-05-3013_09_41.027784/model-keras'

    # Parameters
    #input_size = (128, 128)
    #sequence_length = 30
    #modelToLoad = '/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/MODEL2_2024-05-2423_33_16.378358/model-keras.keras'

    # input_size = (100,100)
    # sequence_length = 30
    # modelToLoad = '/Users/jiten/Masters/Compute vision - CSC 528/CNN_Gesture_recognition/ErikModel/model-keras.keras'



    main()