#!/usr/bin/env python3
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import argparse

"""
Setup logging system
"""
def setup_logging():
	logging.getLogger("tensorflow").setLevel(logging.ERROR)
	tf.get_logger().setLevel('CRITICAL')
	fmt = "%(asctime)s - %(levelname)s - %(message)s"
	global my_logger
	my_logger = logging.getLogger()
	my_logger.setLevel(logging.INFO)
	if not my_logger.hasHandlers():
		my_logger.addHandler(logging.StreamHandler())
	my_logger.handlers[0].setFormatter(logging.Formatter(fmt))

"""
Initialize parsing
"""
# def init_parser():
# 	parser = argparse.ArgumentParser(
#                     prog='Training',
#                     description='Training an image classification model on given dataset',
# 					epilog='By Tamigore and Elpastor')
# 	parser.add_argument('-s', '--save', default="model.h5", help='Path to save the model')
# 	parser.add_argument('-d', '--dataset', default="images", help='Path to the dataset')
# 	return parser

"""
Initialize training and validation datasets from a directory of images.
Args:
	path (str): Path to the directory containing image data.
Returns:
	train_ds: Training dataset.
	val_ds: Validation dataset.
"""
def init_datasets(path: str):
	my_logger.info("\n --- Initializing datasets --- \n")
	train_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
		path,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(256, 256),
		batch_size = 32
	)
	val_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
		path,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(256, 256),
		batch_size = 32
	)
	return train_ds, val_ds

"""
Display information about the dataset and show example images.
Args:
	dataset: The dataset to display information about.
"""
def get_info(dataset: tf.data.Dataset):
	my_logger.info("\n --- Dataset Information --- \n")
	my_logger.info(f"Training dataset: {dataset}")
	class_names = dataset.class_names
	my_logger.info(f"Class names: {class_names}")

	for images, labels in dataset.take(1):
		my_logger.info("\n --- Example batch --- \n batch size, image height, image width, channels")
		my_logger.info(f"Image batch shape: {images.shape}")
		my_logger.info(f"Label batch shape: {labels.shape}")
		plt.figure(figsize=(10, 10)) 
		for i in range(25): 
			plt.subplot(5, 5, i + 1)
			plt.imshow(images[i].numpy().astype("uint8")) 
			plt.title(class_names[labels[i].numpy()]) 
			plt.axis("off")
		plt.show(block=True)

"""
Train a model using the provided training and validation datasets.
Args:
	train_ds: Training dataset.
	val_ds: Validation dataset.
"""
def train_model(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
	my_logger.info("\n --- Training model --- \n")
	num_classes = len(train_ds.class_names) 
	my_logger.info(f"num_classes = {num_classes}")
	model: tf.keras.models.Sequential = tf.keras.models.Sequential([ 
		tf.keras.Input(shape=(256, 256, 3)),
		tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'), 
		tf.keras.layers.MaxPooling2D(), 
		tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'), 
		tf.keras.layers.MaxPooling2D(), 
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'), 
		tf.keras.layers.MaxPooling2D(), 
		tf.keras.layers.Flatten(), 
		tf.keras.layers.Dense(128, activation='relu'), 
		tf.keras.layers.Dense(num_classes) 
	])
	model.compile(	optimizer='adam', 
					loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
					metrics=['accuracy']) 
	model.summary()
	epochs = 10
	history: tf.keras.callbacks.History = model.fit(
		train_ds, 
		validation_data=val_ds, 
		epochs=epochs 
	)
	# visualize_result(history, epochs)
	return model


def visualize_result(history: tf.keras.callbacks.History, epochs):
	print(f"--- Show result ---")
	acc = history.history['accuracy'] 
	val_acc = history.history['val_accuracy'] 
	loss = history.history['loss'] 
	val_loss = history.history['val_loss'] 
	epochs_range = range(epochs) 
	plt.figure(figsize=(8, 8)) 
	plt.subplot(1, 2, 1) 
	plt.plot(epochs_range, acc, label='Training Accuracy') 
	plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
	plt.legend(loc='lower right') 
	plt.title('Training and Validation Accuracy') 
	plt.show() 
	plt.subplot(1, 2, 2) 
	plt.plot(epochs_range, loss, label='Training Loss') 
	plt.plot(epochs_range, val_loss, label='Validation Loss') 
	plt.legend(loc='upper right') 
	plt.title('Training and Validation Loss') 
	plt.show() 


if __name__ == "__main__":
	# parser = init_parser()
	# parser.parse_args()

	model_file = "model.keras"
	dataset = "images/Apple"

	setup_logging()
	train_ds, val_ds = init_datasets(dataset)
	# get_info(train_ds)
	# get_info(val_ds)
	model: tf.keras.models.Sequential = train_model(train_ds, val_ds)
	model.save(model_file)
	print("END")
