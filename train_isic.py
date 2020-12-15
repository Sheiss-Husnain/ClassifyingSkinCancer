os.environ["TFF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf 
import math
import tensorflow_hub as hub 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_examples = 20225
test_examples = 2551
validation_examples = 2555

img_height = img_width = 224 #for purpose of resizing

#Using NasNet (not the best)
# model = keras.Sequential([ #????
# 	hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
# 		trainable=True) #we will train
# 	layers.Dense(1, activation='sigmoid') #????
# 	])

#load saved model
model = keras.models.load_model("isic_model/")

#???
train_datagen = ImageDataGenerator(
	rescale = 1.0/255,
	rotation_range = 15,
	horizontal_flip = True,
	zoom_range = (0.95,0.95), #ranomly zooms in and out 5%
	vertical_flip = True,	#orientation doesn't affect whether cancer
	data_format = "channels_last"
	dtype = tf.float32,
	)	
#no validation split because we want validation resemble test set which is as is

validation_datagen = ImageDataGenerator(rescale=1.0/255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0/255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
	"data/train/",
	target_size=(img_height,img_width) #rescales to this size
	batch_size=batch_size,
	color_mode="rgb", #default anyway
	class_mode="binary",
	shuffle=True,
	seed=123,
	)

validation_gen = validation_datagen.flow_from_directory(
	"data/train/",
	target_size=(img_height,img_width) #rescales to this size
	batch_size=batch_size,
	color_mode="rgb", #default anyway
	class_mode="binary",
	shuffle=True,
	seed=123,
	)

test_gen = test_datagen.flow_from_directory(
	"data/train/",
	target_size=(img_height,img_width) #rescales to this size
	batch_size=batch_size,
	color_mode="rgb", #default anyway
	class_mode="binary",
	shuffle=True,
	seed=123,
	)

METRICS = [
	keras.metrics.BinaryAccuracy(name="accuracy"),
	keras.metrics.Precision(name="precision"),
	keras.metrics.Recall(name="recall"),
	keras.metrics.AUC(name="auc"),
]


model.compile(
	optimizer=keras.optimizers.Adam(lr=3e-4), #learning rate is 3e-4
	loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
	metrics=METRICS,
)

model.fit(
	train_gen,
	epochs=1, #change for non-saved models
	steps_per_epoch=train_examples//batch_size,
	validation_data=validation_gen,
	validation_steps=validation_examples//batch_size,
	callbacks=[keras.callbacks.Modelcheckpoint("isic_model")])
		#saves model
#20:30 the accuracy was only 77.29%
#imbalanced data because vast majority are benign
#therefore accuracy is a flawed metric
#use Precision and Recall instead

#roc curve shows true positive rate vs false positive
def plot_roc(labels, data):
	predictions = model.predict(data)
	fp, tp, _ = roc_curve(labels, predictions)

	plt.plot(100*fp, 100*tp)
	plt.xlabel("False positives [%]")
	plt.ylabel("True positives [%]")

	plt.show()

test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
	test_labels = np.append(test_labels, y)
	num_batches += 1
	if num_batches == math.ceil(test_examples/batch_size):
		break

plot_roc(test_labels, test_gen)
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)



