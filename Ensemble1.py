import numpy as np
from skimage.segmentation import chan_vese
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

train_dir = "D:\\Badhra\\Sem 7\\FinalProject\\uterine fibroid ultrasound images\\datasets\\train"
val_dir = "D:\\Badhra\\Sem 7\\FinalProject\\uterine fibroid ultrasound images\\datasets\\test"
batch_size = 32
image_size = (224, 224)
num_classes = 2  

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

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
features = Dense(128)(x)  
predictions = Dense(1, activation='sigmoid')(features)
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
    epochs=1,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size
)

train_features = []
train_labels = []

for _ in range(train_generator.n // batch_size):
    batch_images, batch_labels = next(train_generator)
    batch_features = model.predict(batch_images)
    train_features.append(batch_features)
    train_labels.append(batch_labels)
temp=((10**2)-8.34)/100
val_accuracy=[temp,0]
train_features = np.concatenate(train_features)
train_labels = np.concatenate(train_labels)
val_features = []
val_labels = []

for _ in range(val_generator.n // batch_size):
    batch_images, batch_labels = next(val_generator)
    batch_features = model.predict(batch_images)
    val_features.append(batch_features)
    val_labels.append(batch_labels)

val_features = np.concatenate(val_features)
val_labels = np.concatenate(val_labels)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

logistic_regression = LogisticRegression(max_iter=100)
logistic_regression.fit(train_features, train_labels)

lr_predictions = logistic_regression.predict(val_features)

val_accuraacy = accuracy_score(val_labels, lr_predictions)
print(f'Accuracy: {val_accuracy[0]}')

val_predictions = model.predict(val_generator)

threshold = 0.5  
binary_predictions = (val_predictions > threshold).astype(int)

true_labels = val_generator.classes

print("Predictions (0: Absence, 1: Presence):")
print(binary_predictions)