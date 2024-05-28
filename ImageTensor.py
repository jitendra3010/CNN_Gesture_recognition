import numpy as np

class ImageTensor:

    def __init__(self, n_frames):
        # sepecify the no of frames we have 30 fps but can work with different numbers
        self.n_frames = n_frames
    
    def getImgTensor(self):
        # since 30fls we define the linespace of 0, 29 and get n_frames data
        img_idx = np.round(np.linspace(0, 29, self.n_frames)).astype(int)
        # 100 , 100 is the dimension of image with 3 channels
        return [img_idx, 128, 128, 3]