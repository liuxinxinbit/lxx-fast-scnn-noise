from SCNN import SCNN
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import imgviz
import time




scnn = SCNN()
# scnn.train(epochs=10, steps_per_epoch=500, batch_size=4)
# scnn.save()


# scnn.load()
# score = scnn.BatchEvaluate(batch_size=50)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# for flag in range(50):
#     images,truths = get_train_data(5)
#     image = images[0,:,:,:]
#     label = truths[0,:,:]   
    
#     plt.subplot(1, 3, 1)
#     plt.title("label")
#     plt.imshow(label)
#     prediction = rtnet.predict(image)

#     plt.subplot(1, 3, 2)
#     plt.title("prediction")
#     plt.imshow(prediction[0,:,:,0]>0.5)
#     plt.subplot(1, 3, 3)
#     plt.title("image")
#     plt.imshow(image[:,:,0]+label)
#     plt.pause(0.1)
    
#     plt.savefig("img/"+str(flag)+".png",dpi=300,)
#     plt.clf()
#     # plt.show()