import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
plt.style.use('seaborn')
def denoise(img_path):
  image = cv2.imread(img_path)
  dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
  dst=cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
  dst = np.reshape(dst,dst.shape)
  data = im.fromarray(dst)
  data.save('data/images/denoised.png')
