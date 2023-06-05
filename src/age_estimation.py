from PIL import Image
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def estimate_age(image_path):
    # Define model parameters
    width = 8
    height = 8
    face_size = 224

    # Load model and weights
    model = get_wide_resnet(input_shape=(face_size, face_size, 3))
    model.load_weights('weights.29-3.76_utk.hdf5')

    # Load image
    img = Image.open(image_path)
    img = img.resize((face_size, face_size))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    # Predict age
    results = model.predict(img)
    estimated_age = results[0].argmax(axis=-1)

    return estimated_age

def get_wide_resnet(input_shape=(224, 224, 3)):
    # Load pretrained model
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        input_shape=input_shape,
        pooling="avg"
    )
    
    features = base_model.output
    
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    
    model = Model(inputs=base_model.input, outputs=[pred_age])
    
    return model