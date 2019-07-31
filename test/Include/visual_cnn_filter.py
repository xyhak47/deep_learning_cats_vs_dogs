from keras.applications import VGG16
from keras import backend as K
import numpy as np
from draw_data import *



model = VGG16(weights='imagenet',
              include_top=False)

model.summary()



def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([model.input],[loss, grads])
    # loss_value, grades_value = iterate([np.zeros((1,150,150,3))])

    input_img_data = np.random.random((1,size,size,3))*20+128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value*step

    img = input_img_data[0]
    return deprocess_image(img)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x,0,1)

    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x



# layer_name = 'block3_conv1'
# filter_index = 0
# plt.imshow(generate_pattern(layer_name, filter_index, size=64))
# plt.show()



layer_names = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
size = 64
margin = 5

def draw_blocks_conv_first():
    for layer_name in layer_names:
        print(layer_name)

        '''uint8类型的数据转为0 - 1之间的float类型，本质就是img = img / 255.'''
        results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3)).astype('uint8')
        for i in range(8):
            for j in range(8):
                filter_img = generate_pattern(layer_name, i+(j*8), size=size)
                horzontal_start = i*size+i*margin
                horzontal_end = horzontal_start+size
                vertical_start = j*size+j*margin
                vertical_end = vertical_start+size
                results[horzontal_start:horzontal_end,
                        vertical_start:vertical_end, :] = filter_img

        plt.figure(figsize=(20,20))
        plt.imshow(results)
        plt.show()


draw_blocks_conv_first()
