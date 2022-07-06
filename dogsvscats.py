from utility import separate_train_test, separate_train_val, data_gen
from my_model import function_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers, models
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == "__main__":

    train_test = False
    if train_test:
        data = "/home/bishesh/Desktop/DogsVsCats/PetImages"
        train = "/home/bishesh/Desktop/DogsVsCats/prepare_data/train"
        test = "/home/bishesh/Desktop/DogsVsCats/prepare_data/test"

        separate_train_test(path_to_data=data, path_to_save_train=train, path_to_save_test=test)

    train_val = False
    if train_val:
        data = "/home/bishesh/Desktop/DogsVsCats/prepare_data/train"
        val = "/home/bishesh/Desktop/DogsVsCats/prepare_data/val"

        separate_train_val(path_to_data=data, path_to_val=val)

    x = True
    
    if x:
        train_path = '/home/bishesh/Desktop/DogsVsCats/prepare_data/train'
        test_path = '/home/bishesh/Desktop/DogsVsCats/prepare_data/test'
        val_path = '/home/bishesh/Desktop/DogsVsCats/prepare_data/val'
        batch_size=32
        epochs = 15
        learning_rate = 0.0001

        train_data, test_data, val_data = data_gen(batch_size=batch_size, train_path=train_path, test_path=test_path, val_path=val_path)

        n_class = train_data.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        path_to_save_model = './Models'

        checkpnt = ModelCheckpoint(
            path_to_save_model, 
            monitor = "val_accuracy",
            verbose = 1,
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch'
        )

        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        )

        early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

        callbacks = [checkpnt, early_stop, learning_rate_reduction]

        model = function_model(n_class)

        #optimizer = optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model = model.fit(
            train_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks
        )
        for key in model.history:
            print(key)


    if TEST:

        model = models.load_model("./Models")
        model.summary()

        print("Evaluating Validation set:")
        model.evaluate(val_data)

        print("Evaluating test set:")
        model.evaluate(test_data)       
