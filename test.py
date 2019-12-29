import cv2
import numpy as np


bw_image = np.eye(100, dtype=np.float32)
cv2.imshow("Color Image",bw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
