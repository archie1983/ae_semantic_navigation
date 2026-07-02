import zmq, glob, re
import numpy as np
import time, cv2, os
from PIL import Image

class SemanticNavigationClient:
	LLM_PORT = 5555
	DR_NAV_PORT = 5556
	RC_NAV_PORT = 5557

	def __init__(self, jetson_ip):
		self.context = zmq.Context()
		# LLM container
		self.llm_socket = self.context.socket(zmq.REQ)  # REQuest socket
		self.llm_socket.connect(f"tcp://{jetson_ip}:{self.LLM_PORT}")
		print(f"Connected to Jetson LLM container at {jetson_ip}:{self.LLM_PORT}")

		# Door navigation container
		self.dr_socket = self.context.socket(zmq.REQ)  # REQuest socket
		self.dr_socket.connect(f"tcp://{jetson_ip}:{self.DR_NAV_PORT}")
		print(f"Connected to Jetson Door navigation container at {jetson_ip}:{self.DR_NAV_PORT}")

		# Room centre navigation container
		self.rc_socket = self.context.socket(zmq.REQ)  # REQuest socket
		self.rc_socket.connect(f"tcp://{jetson_ip}:{self.RC_NAV_PORT}")
		print(f"Connected to Jetson RoomCentre navigation container at {jetson_ip}:{self.RC_NAV_PORT}")

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
		self.llm_socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.llm_socket.recv_pyobj()
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
		self.llm_socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.llm_socket.recv_pyobj()
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
		self.llm_socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.llm_socket.recv_pyobj()
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
		self.llm_socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.llm_socket.recv_pyobj()
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

def extract_number(filename):
	# Extract the number from the filename (assuming it's the step count)
	# This regex looks for digits at the beginning, end, or between non-digits
	numbers = re.findall(r'\d+', filename)
	return int(numbers[-1]) if numbers else 0

def load_images(path):
	imgs_path = glob.glob(path)
	imgs_path = sorted(imgs_path, key=extract_number)
	pil_images = [Image.open(fname).convert('RGB') for fname in imgs_path]
	return pil_images

def load_path(base_dir):
	return np.stack(load_images(base_dir + "/*.png"))

if __name__ == "__main__":
	# Create agent and connect to Jetson
	agent = SemanticNavigationClient(jetson_ip="192.168.0.109")

	# store reference path images
	# print(agent.store_ref_path(gen_n_imgs(10), "bathroom"))
	# print(agent.store_ref_path(gen_n_imgs(10), "kitchen"))
	# print(agent.store_ref_path(gen_n_imgs(10), "living_room"))
	# print(agent.qry_path_similarity(gen_n_imgs(3)))

	#ref_path1 = load_path(
	#	"/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/1")
	#print(ref_path1)
	#img_array = np.stack(ref_path1, axis=0)
	#print(img_array.dtype)
	#exit()

	# Object detection in an image
	pil_image = Image.open("/home/hp20024/robotics/latent_planning/dreamerv3/scene_pics/8.png")
	img_array = np.stack([pil_image], axis = 0)
	obj_det_res = agent.detect_objects_in_image(img_array)
	det_objs = set(obj_det_res['item_names'])
	print("AE: det objs: ", det_objs)

	# Room type inference from a set of objects and/or an image
	print("AE: room type: ", agent.classify_room_by_this_object_set_and_pic(obj_set=det_objs, img_bytes = img_array))

	# Embedding of a path
	ref_path1 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/1")
	ref_path2 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/2")
	ref_path3 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/3")
	ref_path4 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/4")
	ref_path7 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/7")

	ref_cmp_path = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/tmp_cmp")

	agent.store_ref_path(ref_path1, "ref_path1")
	agent.store_ref_path(ref_path2, "ref_path2")
	agent.store_ref_path(ref_path3, "ref_path3")
	agent.store_ref_path(ref_path4, "ref_path4")
	agent.store_ref_path(ref_path7, "ref_path7")

	# Comparison of a path against stored embedded ones
	path_cmp_res = agent.qry_path_similarity(ref_cmp_path)
	print("AE: path_cmp res: ", path_cmp_res)

	#/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/tmp_cmp