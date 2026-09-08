from importlib.resources import files
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os, cv2

class YoloObjectDetector:
	def __init__(self):
		# YOLO part
		# Load the exported TensorRT model
		#self.item_extractor_model = YOLO("item_extract_ae.engine")
		#self.item_extractor_model = YOLO("item_extract_ae.pt")

		# Get the path to your model file relative to the package
		model_path = files('ae_semantic_navigation.models').joinpath('item_extract_ae.engine')
		self.item_extractor_model = YOLO(str(model_path))
		self.img_cnt = 0

	def detect_objects_in_image(self, data):
		# 2. Process the images
		received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
		received_img = received_array.reshape(data['shape'])[0]
		# get x images from the received (x, 64, 64, 3) tensor. This will be our path to compare
		pil_image = Image.fromarray(received_img)
		self.store_image(pil_image)

		item_extractor_res = self.item_extractor_model(pil_image)

		# print("AE Classes: ", item_extractor_res[0].boxes.cls)
		# print("AE Classes1: ", item_extractor_res[0].names)
		# print("AE All: ", item_extractor_res[0].boxes)

		item_names = [item_extractor_res[0].names[int(item)] for item in item_extractor_res[0].boxes.cls]
		print("AE: item_names: ", item_names)

		response = {
			'item_names': item_names,
			'success': True
		}

		return response

	def store_image(self, img):
		## debug
		path_id = "tmp_img"
		os.makedirs(path_id, exist_ok=True)
		self.img_cnt += 1
		img = np.stack([img])
		img = img[0]
		cv2.imwrite(os.path.join(path_id, str(self.img_cnt) + ".png"), img)
		print("AE: name: ", str(self.img_cnt), ".png")