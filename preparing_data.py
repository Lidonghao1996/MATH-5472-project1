# resize the data to save time.
import PIL
from PIL import Image
import numpy as np
import os

def processing(path):
    im=Image.open(path).resize((64,64),PIL.Image.BILINEAR).convert("RGB")
    im=np.array(im).transpose(2,0,1)
    return im


output_path="3dchairs"
folder_path="images/"
image_names=os.listdir(folder_path)
# print(len(image_names))
image_paths=[folder_path+i for i in image_names]
arrays=[ processing(i) for i in image_paths ]
output=np.array(arrays)
# print(output.shape)


np.save(output_path,output)
