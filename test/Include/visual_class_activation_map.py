
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from path_generator import *
import keras.backend as K
from draw_data import *
import cv2

model = VGG16(weights='imagenet')

img_path = join_path(base_dir, 'creative_commons_elephant.png')
print(img_path)
img = image.load_img(img_path, target_size=(224,224))
x = image.array_to_img(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
# print('preds', preds)
print(np.argmax(preds[0]))
# print(preds[0])

african_elephant_output = model.output[:, 386]

'''last conv layer of vgg16'''
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0,1,2))
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

'''convert to 0~1'''
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

img = cv2.imread(img_path)
print(img.shape)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

elephant_cam_img_path = join_path(base_dir, 'elephant_cam.jpg')
cv2.imwrite(elephant_cam_img_path, superimposed_img)

plt.imshow(image.load_img(elephant_cam_img_path))
plt.show()