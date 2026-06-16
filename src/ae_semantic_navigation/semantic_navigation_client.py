import zmq
import numpy as np
import time, cv2, os
from PIL import Image

class SemanticNavigationClient:
	def __init__(self, jetson_ip, port=5555):
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REQ)  # REQuest socket
		self.socket.connect(f"tcp://{jetson_ip}:{port}")
		print(f"Connected to Jetson at {jetson_ip}:{port}")

	def store_ref_path(self, path_imgs, path_id="?"):
		"""
		Send an a collection of images, representing a reference path, to server.

		Args:
			image_np: numpy array (x, H, W, C) in BGR order (typical from OpenCV/AI2-THOR)

		Returns:
			success flag or None if error
		"""
		# Serialize the images
		data = {
			'shape': path_imgs.shape,
			'dtype': str(path_imgs.dtype),
			'bytes': path_imgs.tobytes(),
			'path_id': path_id,
			'action': "store_ref_path",
			'module': "path_comparator"
		}

		## debug
		path_id = str(path_id)
		os.makedirs(path_id, exist_ok=True)
		cnt = 0
		for img in path_imgs:
			cnt += 1
			cv2.imwrite(os.path.join(path_id, str(cnt) + ".png"), img)
		## /debug

		# Send request
		self.socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.socket.recv_pyobj()
			return response
		except zmq.ZMQError as e:
			print(f"Error receiving response: {e}")
			return None

	def qry_path_similarity(self, path_imgs):
		"""
		Main navigation loop with real-time confidence feedback.

		Args:
			get_image_func: Function that captures current FPV image from AI2-THOR
			max_steps: Maximum number of steps to take
		"""
		data = {
			'shape': path_imgs.shape,
			'dtype': str(path_imgs.dtype),
			'bytes': path_imgs.tobytes(),
			'action': "qry_path_similarity",
			'module': "path_comparator"
		}

		## debug
		path_id = "tmp_cmp"
		os.makedirs(path_id, exist_ok=True)
		cnt = 0
		for img in path_imgs:
			cnt += 1
			cv2.imwrite(os.path.join(path_id, str(cnt) + ".png"), img)
		## /debug

		# Send request
		self.socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.socket.recv_pyobj()
			return response
		except zmq.ZMQError as e:
			print(f"Error receiving response: {e}")
			return None

		# Small delay to avoid overwhelming the system
		time.sleep(0.05)

	def detect_objects_in_image(self, img):
		"""
		Send an a collection of images, representing a reference path, to server.

		Args:
			image_np: numpy array (x, H, W, C) in BGR order (typical from OpenCV/AI2-THOR)

		Returns:
			success flag or None if error
		"""
		# Serialize the images
		data = {
			'shape': img.shape,
			'dtype': str(img.dtype),
			'bytes': img.tobytes(),
			'action': "detect_objects_in_image",
			'module': "yolo_object_detector"
		}

		# Send request
		self.socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.socket.recv_pyobj()
			return response
		except zmq.ZMQError as e:
			print(f"Error receiving response: {e}")
			return None

	def classify_room_by_this_object_set_and_pic(self, obj_set = None, img_bytes = None):
		data = {
			'shape': img_bytes.shape,
			'dtype': str(img_bytes.dtype),
			'bytes': img_bytes.tobytes(),
			'obj_set': obj_set,
			'action': 'classify_room_by_this_object_set_and_pic',
			'module': 'llm_decisions'
		}

		# Send request
		self.socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.socket.recv_pyobj()
			return response
		except zmq.ZMQError as e:
			print(f"Error receiving response: {e}")
			return None

# Example usage
def gen_n_imgs(img_num):
	"""Replace with your actual AI2-THOR frame capture"""
	# Simulate a 64x64 RGB image
	return np.random.randint(0, 255, (img_num, 64, 64, 3), dtype=np.uint8)

def gen_1_img():
	"""Replace with your actual AI2-THOR frame capture"""
	# Simulate a 64x64 RGB image
	return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

if __name__ == "__main__":
	# Create agent and connect to Jetson
	agent = SemanticNavigationClient(jetson_ip="192.168.0.109", port=5555)

	# store reference path images
	# print(agent.store_ref_path(gen_n_imgs(10), "bathroom"))
	# print(agent.store_ref_path(gen_n_imgs(10), "kitchen"))
	# print(agent.store_ref_path(gen_n_imgs(10), "living_room"))
	# print(agent.qry_path_similarity(gen_n_imgs(3)))

	pil_image = Image.open("/home/hp20024/robotics/latent_planning/dreamerv3/scene_pics/8.png")
	img_array = np.stack([pil_image], axis = 0)
	print(agent.detect_objects_in_image(img_array))

	print(agent.classify_room_by_this_object_set_and_pic(obj_set={"Scales", "bathtub", "toothbrush"}, img_bytes = img_array))