import os

def mkdir_if_needed(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

def join_path(origin, tail):
    return os.path.join(origin, tail)


original_dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "../../dogs-vs-cats"))
base_dir = join_path(original_dataset_dir, 'cats_and_dogs_small')
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
