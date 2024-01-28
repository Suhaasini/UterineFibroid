import os
import numpy as np
from skimage.segmentation import chan_vese
from skimage.measure import regionprops
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score

train_dir = "D:\\Badhra\\Sem 7\\FinalProject\\uterine fibroid ultrasound images\\datasets\\train"
val_dir = "D:\\Badhra\\Sem 7\\FinalProject\\uterine fibroid ultrasound images\\datasets\\test"

batch_size = 32
image_size = (224, 224) 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=5,  
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size
)


train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
print(f'Validation Accuracy: {val_accuracy}')


def chan_vese_segmentation(img):
    segmented_img = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200)
    segmented_img = (segmented_img * 255).astype(np.uint8)
    return segmented_img


def extract_convex_hull_features(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull_areas = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        convex_hull_area = cv2.contourArea(hull)
        convex_hull_areas.append(convex_hull_area)
    return convex_hull_areas


val_features = []
val_labels = []
for _ in range(val_generator.n // batch_size):
    batch_images, batch_labels = next(val_generator)
    batch_features = []
    for i in range(batch_images.shape[0]):
        img = batch_images[i]
        segmented_img = chan_vese_segmentation(img)
        features = extract_convex_hull_features(segmented_img)
        batch_features.append(features)
    batch_features = np.array(batch_features)
    val_features.append(batch_features)
    val_labels.append(batch_labels)

val_features = np.concatenate(val_features)
val_labels = np.concatenate(val_labels)


print(f'Updated Validation Accuracy: {updated_val_accuracy}')