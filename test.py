
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('model.keras')
image=cv2.imread(r'D:\A PROJECT MSC 2\pappya\Project code\Project code\dataset\yes\023.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)

input_img=np.expand_dims(img,axis=0)

result=model.predict(input_img)

print(result)