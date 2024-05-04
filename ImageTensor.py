import numpy as np

class ImageTensor:

    def __init__(self, n_frames):
        self.n_frames = n_frames
    
    def getImgTensor(self):
        img_idx = np.round(np.linspace(0, 29, self.n_frames)).astype(int)
        return [img_idx, 100, 100, 3]