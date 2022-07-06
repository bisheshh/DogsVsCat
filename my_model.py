from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras import Model

def function_model(n_class):

    my_input = Input(shape=(128,128,3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    

    x = Conv2D(64, (3,3), activation='relu')(x)    
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.25)(x)


    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.25)(x)

    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

if __name__ == '__main__':

    model = function_model(2)
    model.summary()

