import tensorflow as tf



def down_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

        p = tf.keras.layers.MaxPool2D((2,2), (2,2))(c)
    return c, p 

def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        up_sampling = tf.keras.layers.UpSampling2D((2,2))(x)
        concat = tf.keras.layers.Concatenate()([up_sampling, skip])

        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(concat)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def UNet(feature_channels, image_size):
    feature_maps = feature_channels  #[64,128,256, 512, 1024]
    inputs = tf.keras.layers.Input( (image_size, image_size, 1) )

    pool_0 = inputs
    conv_1, pool_1 = down_block(pool_0, feature_maps[0]) #512 -> 256
    conv_2, pool_2 = down_block(pool_1, feature_maps[1]) #256 -> 128 
    conv_3, pool_3 = down_block(pool_2, feature_maps[2]) #128 -> 64
    conv_4, pool_4 = down_block(pool_3, feature_maps[3]) #64 -> 32

    bn = bottleneck(pool_4, feature_maps[4])

    ups_1 = up_block(bn, conv_4, feature_maps[3]) #32 -> 64
    ups_2 = up_block(ups_1, conv_3, feature_maps[2]) #64 -> 128
    ups_3 = up_block(ups_2, conv_2, feature_maps[1]) #128 -> 256
    ups_4 = up_block(ups_3, conv_1, feature_maps[0]) #256 -> 512

    outputs = tf.keras.layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(ups_4)

    model = tf.keras.models.Model(inputs, outputs)
    return model

