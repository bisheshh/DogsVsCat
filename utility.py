import os
import glob
import shutil
from sklearn import model_selection
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def separate_train_test(path_to_data, path_to_save_train, path_to_save_test, split_size=0.2):

    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        img_path = glob.glob(os.path.join(full_path, '*.jpg'))

        x_train, x_test = model_selection.train_test_split(img_path, test_size=split_size)

        for x in x_train:

            path_to_train_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_train_folder):
                os.makedirs(path_to_train_folder)

            shutil.copy(x, path_to_train_folder)   

        for x in x_test:

            path_to_test_folder = os.path.join(path_to_save_test, folder)

            if not os.path.isdir(path_to_test_folder):
                os.makedirs(path_to_test_folder)

            shutil.copy(x, path_to_test_folder)


def separate_train_val(path_to_data, path_to_val, split_size = 0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        img_path = glob.glob(os.path.join(full_path, '*.jpg'))

        _ , x_val = model_selection.train_test_split(img_path, test_size=split_size)  

        for x in x_val:
            
            path_to_save_val = os.path.join(path_to_val, folder)

            if not os.path.isdir(path_to_save_val):
                os.makedirs(path_to_save_val)

            shutil.move(x, path_to_save_val)


def data_gen(batch_size, train_path, test_path, val_path):

    train_process = ImageDataGenerator(
        rescale = 1/255., 
        rotation_range = 15,
        shear_range=0.1,
        zoom_range=0.2,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True
    )
    test_process = ImageDataGenerator(
        rescale= 1/255.
    )
    
    train_gen = train_process.flow_from_directory(
        directory=train_path,
        target_size = (128,128),
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size
    )

    test_gen = test_process.flow_from_directory(
        directory=test_path,
        target_size=(128,128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = test_process.flow_from_directory(
        directory=val_path,
        target_size=(128,128),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_gen, test_gen, val_gen
