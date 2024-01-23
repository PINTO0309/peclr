import cv2
import numpy as np

numpy_image = np.load('image.npy')
cv2.imshow('Image', numpy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
