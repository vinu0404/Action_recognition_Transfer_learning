import tensorflow as tf
from config import CLASSES

def create_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze the base model

    inputs = tf.keras.Input(shape=(10, 224, 224, 3))
    x = tf.keras.layers.TimeDistributed(base_model)(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model