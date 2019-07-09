import torchvision
from tricks import tricks
import numpy as np
from PIL import Image

mean_rgb = np.array([0.0912, 0.8827, 0.4953])

face_jpg = Image.open('face.jpg')
face_erase = tricks.RandomErasing(EPSILON=1.0, mean=mean_rgb)(face_jpg)
face_erase.save('face_erase1.jpg')
