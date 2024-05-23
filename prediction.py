import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import load_model
# from tensorflow.keras.models import load_model
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model_path = "Har_model_MobilenetLarge.h5"
model = load_model(model_path)


# def read_img(fn):
#     img = Image.open(fn)
#     return np.asarray(img.resize((160,160)))

# labels = ['sitting', 'using_laptop', 'hugging', 'sleeping', 'drinking',
#        'clapping', 'dancing', 'cycling', 'calling', 'laughing', 'eating',
#        'fighting', 'listening_to_music', 'running', 'texting', 'smoking',
#        'weapon', 'knife']

# def test_predict(test_image):
#     result = model.predict(np.asarray([read_img(test_image)]))
#     print("result:",result)
#     print("npmax result: ",np.max(result))
#     if np.max(result)>0.6:
#        itemindex = np.where(result==np.max(result))
#        print("Itemindex: ",itemindex)
#        prediction = itemindex[1][0]
#        print("prediction:",prediction)
#     #    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)
#        print("Lable: ",labels[prediction])
#        image = img.imread(test_image)
#        plt.imshow(image)
#        plt.title(prediction)
#        return labels[prediction]
#     else:
#         return "Unable to Predict"

# test_predict("Image_6.jpg")
