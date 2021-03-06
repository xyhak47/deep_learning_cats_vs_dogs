import os, shutil
from path_generator import *

debug = 1
def debug_print1(string):
    if(debug == 1):
        print(string)

def debug_print(string1, string2):
    if(debug == 1):
        print(string1, string2)



def copy_file_if_needed(src, dst):
    if(os.path.exists(dst)):
        # debug_print('skip : ', dst)
        return
    shutil.copyfile(src, dst)


fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = join_path(train_original_dataset_dir, fname)
    dst = join_path(train_cats_dir, fname)
    copy_file_if_needed(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = join_path(train_original_dataset_dir, fname)
    dst = join_path(validation_cats_dir, fname)
    copy_file_if_needed(src, dst)


fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = join_path(train_original_dataset_dir, fname)
    dst = join_path(test_cats_dir, fname)
    copy_file_if_needed(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = join_path(train_original_dataset_dir, fname)
    dst = join_path(train_dogs_dir, fname)
    copy_file_if_needed(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = join_path(train_original_dataset_dir, fname)
    dst = join_path(validation_dogs_dir, fname)
    copy_file_if_needed(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = join_path(train_original_dataset_dir, fname)
    dst = join_path(test_dogs_dir, fname)
    copy_file_if_needed(src, dst)


# debug_print('total training cat images:' , len(os.listdir(train_cats_dir)))
# debug_print('total training dog images:' , len(os.listdir(train_dogs_dir)))
# debug_print('total validation cat images:' , len(os.listdir(validation_cats_dir)))
# debug_print('total validation cat images:' , len(os.listdir(validation_dogs_dir)))
# debug_print('total test cat images:' , len(os.listdir(test_cats_dir)))
# debug_print('total test cat images:' , len(os.listdir(test_cats_dir)))


from keras import layers
from keras import models

model = models.Sequential();
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')


for data_batch, labels_batch in train_generator:
    debug_print('data batch shape:', data_batch.shape)
    debug_print('labels batch shape:', labels_batch.shape)
    break


from draw_data import *

model_save_path_1 = join_path(base_dir, 'cats_and_dogs_small_1.h5')

if not os.path.exists(model_save_path_1):
    # there are 2000 images in train_dir, batch_size is 20, so steps_per_epoch = 2000 / 20 = 100
    # there are 1000 images in validation_dir, batch_size is 20, so validation_steps = 1000 / 20 = 50
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
    model.save(model_save_path_1)
    draw_data(history)





#---------------------------------------------------------------------------------
#we will do it again with data augmentation:
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# here is an example of data augmentation:
from keras.preprocessing import image


def draw_data2():
    fnames = [join_path(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

    img_path = fnames[3] #pick up one image
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img) # -> (150,150,3) numpy array
    x = x.reshape((1,) + x.shape) # -> (1,150,150,3)

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i%4 == 0:
            break

    plt.show()


# draw_data2()


# new model with adding dropout and using augmentation data
model = models.Sequential();
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))   # <--------------------change here
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

model_save_path_2 = join_path(base_dir, 'cats_and_dogs_small_2.h5')

if not os.path.exists(model_save_path_2):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)
    model.save(model_save_path_2)
    draw_data()
