import json

import tensorflow as tf

def augment_images(x, config):
    if config['use_contrast'] == "True":
        x = tf.keras.layers.experimental.preprocessing.RandomContrast(
            config['contrast_factor']
        )(x)

    if config['use_rotation'] == "True":
        x = tf.keras.layers.experimental.preprocessing.RandomRotation(
            config['rotation_factor']
        )(x)
    
    if config['use_flip'] == "True":
        x = tf.keras.layers.experimental.preprocessing.RandomFlip(
            config['flip_mode']
        )(x)

    return x

def FCN_model(config, len_classes=5):
    
    input = tf.keras.layers.Input(shape=(None, None, 3))

    x = augment_images(input, config)

    x = tf.keras.layers.Conv2D(filters=config['conv_block1_filters'], kernel_size=3, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)


    x = tf.keras.layers.Conv2D(filters=config['conv_block2_filters'], kernel_size=3, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)


    x = tf.keras.layers.Conv2D(filters=config['conv_block3_filters'], kernel_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)


    x = tf.keras.layers.Conv2D(filters=config['conv_block4_filters'], kernel_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)


    x = tf.keras.layers.Conv2D(filters=config['conv_block5_filters'], kernel_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    if config['fc_layer_type'] == 'dense':
        if config['pool_type'] == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Fully connected layer 1
        x = tf.keras.layers.Dense(units=config['fc1_units'])(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Fully connected layer 2
        x = tf.keras.layers.Dense(units=len_classes)(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        predictions = tf.keras.layers.Activation('softmax')(x)

    else:
        # Fully connected layer 1
        x = tf.keras.layers.Conv2D(filters=config['fc1_units'], kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)


        # Fully connected layer 2
        x = tf.keras.layers.Conv2D(filters=len_classes, kernel_size=1, strides=1)(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if config['pool_type'] == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        predictions = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)
    
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

    return model

if __name__ == "__main__":
    # config =   {"lr": 0.0001,
    #             "batch_size": 8, 
    #             "use_contrast": "False",
    #             "use_rotation": "False",
    #             "use_flip": "False",
    #             "contrast_factor": 0.2,
    #             "rotation_factor": 0.2,
    #             "flip_mode": 0.2,
    #             "dropout_rate": 0.2,
    #             "conv_block1_filters": 32,
    #             "conv_block2_filters": 64,
    #             "conv_block3_filters": 128,
    #             "conv_block4_filters": 256,
    #             "conv_block5_filters": 512,
    #             "fc_layer_type": 'convolution',
    #             "pool_type": 'max',
    #             "fc1_units": 64}
    with open('./snapshots/config.json', 'r') as f:
        config = json.load(f)
    model = FCN_model(config, len_classes=5)
    model.load_weights('./snapshots/train_model.h5')
    