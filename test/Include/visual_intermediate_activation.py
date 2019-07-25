from path_generator import *
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from draw_data import *
from keras import models
np.seterr(divide='ignore', invalid='ignore')

model_save_path_2 = join_path(base_dir, 'cats_and_dogs_small_2.h5')

model = load_model(model_save_path_2)
model.summary()

img_path = join_path(test_cats_dir, 'cat.1700.jpg')

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

plt.imshow(img_tensor[0])
# plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')
plt.matshow(first_layer_activation[0,:,:,7], cmap='viridis')
# plt.show()


layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

print(layer_names)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    # print(n_features)
    size = layer_activation.shape[1]  # (1,size,size,n_features)
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channnel_image = layer_activation[0,:,:,col*images_per_row+row]
            channnel_image -= channnel_image.mean()
            channnel_image /= channnel_image.std()
            channnel_image *= 64
            channnel_image += 128
            channnel_image = np.clip(channnel_image,0,255)
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size] = channnel_image
    scale = 1./size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()