import tensorflow as tf

def model_7_3_12channel(size,L2_factor):

    inputs = tf.keras.Input(shape=(size,size,12))
    x = tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(7,activation='softmax')(x)

    return tf.keras.Model(inputs,outputs)




def model_7_3(size,L2_factor):
    inputs = tf.keras.Input(shape=(size,size,3))
    x = tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2_factor))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(7,activation='softmax')(x)
    return tf.keras.Model(inputs,outputs)