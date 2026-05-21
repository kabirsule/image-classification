# model.py - MobileNet Engine
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(num_classes=3):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0015),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

if __name__ == "__main__":
    model, _ = build_model()
    model.summary()
    print("MobileNet architecture configured safely for hardware profiles.")
