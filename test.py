import pickle
import numpy as np
import cv2

actions_list = []
images_list = []

with open('_out/images_new2_balanced.pkl','rb') as af:
    images_list = pickle.load(af)
    
print("Length of overall list: ", len(images_list))
print("Shape of each entry: ", len(images_list[0]))

print("RGB shape: ", images_list[0][0].shape)
print("Sem shape: ", images_list[0][1].shape)

#cv2.imshow("Semantic Segmentation", images_list[0][0])
#cv2.imshow("Aleatoric Uncertainty", images_list[0][1])
cv2.imwrite("RGB_ali.png",images_list[0][0])
cv2.imwrite("Segmentation_ali.png",images_list[0][1])
