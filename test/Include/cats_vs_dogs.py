import os, shutil

debug = 1
def debug_print1(string):
    if(debug == 1):
        print(string)

def debug_print(string1, string2):
    if(debug == 1):
        print(string1, string2)


def mkdir_if_needed(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

def join_path(origin, tail):
    return os.path.join(origin, tail)


original_dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "../../dogs-vs-cats"))
base_dir = join_path(original_dataset_dir, 'cats_and_dogs_small')
# debug_print(base_dir)
train_original_dataset_dir = join_path(original_dataset_dir, 'train')
test_original_dataset_dir = join_path(original_dataset_dir, 'test')

mkdir_if_needed(base_dir)

train_dir = join_path(base_dir, 'train')
mkdir_if_needed(train_dir)

validation_dir = join_path(base_dir, 'validation')
mkdir_if_needed(validation_dir)

test_dir = join_path(base_dir, 'test')
mkdir_if_needed(test_dir)

train_cats_dir = join_path(train_dir, 'cats')
mkdir_if_needed(train_cats_dir)

train_dogs_dir = join_path(train_dir, 'dogs')
mkdir_if_needed(train_dogs_dir)

validation_cats_dir = join_path(validation_dir, 'cats')
mkdir_if_needed(validation_cats_dir)

validation_dogs_dir = join_path(validation_dir, 'dogs')
mkdir_if_needed(validation_dogs_dir)

test_cats_dir = join_path(test_dir, 'cats')
mkdir_if_needed(test_cats_dir)

test_dogs_dir = join_path(test_dir, 'dogs')
mkdir_if_needed(test_dogs_dir)


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
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')


for data_batch, labels_batch in train_generator:
    debug_print('data batch shape:', data_batch.shape)
    debug_print('labels batch shape:', labels_batch.shape)
    break


# there are 2000 images in train_dir, batch_size is 20, so steps_per_epoch = 2000 / 20 = 100
# there are 1000 images in validation_dir, batch_size is 20, so validation_steps = 1000 / 20 = 50
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save(join_path(base_dir, 'cats_and_dogs_small_1.h5'))


#draw to show
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()

plt.show()



#---------------------------------------------------------------------------------
#we will do it again with data augmentation:


# here is an example of data augmentation:
from keras.preprocessing import image

fnames = [join_path(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3] #pick up one image
img = image.load_img(img_path, target_size=(150,150))
x = image.img_to_array(img) # -> (150,150,3) numpy array
x = x.reshape((1,) + x.shape) # -> (1,150,150,3)

