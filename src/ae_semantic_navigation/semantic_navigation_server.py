import zmq
from ae_semantic_navigation import YoloObjectDetector, PathComparator, LLMDecisions


class SemanticNavigationServer:
	def __init__(self, port=5555, use_dino = True):
		# Path comparator
		self.path_comparator = PathComparator(use_dino)
		# Yolo item detector
		self.yolo_object_detector = YoloObjectDetector()
		# LLM decisions' maker
		self.llm_decisions = LLMDecisions()

		# Set up ZeroMQ with REP (REPly) socket
		self.port = port
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)  # Changed from PULL to REP
		self.socket.bind(f"tcp://*:{self.port}")

		print(f"Path Compare server running on port {self.port}, waiting for images...")

	# def __init__(self, path_compare_instance, image_port=5555, response_port=5556):
	# 	self.pc = path_compare_instance
	# 	self.context = zmq.Context()
	#
	# 	# Receive images on PULL socket
	# 	self.image_socket = self.context.socket(zmq.PULL)
	# 	self.image_socket.bind(f"tcp://*:{image_port}")
	#
	# 	# Send responses on PUSH socket
	# 	self.response_socket = self.context.socket(zmq.PUSH)
	# 	self.response_socket.bind(f"tcp://*:{response_port}")
	#
	# 	print(f"Server ready: receiving images on {image_port}, sending responses on {response_port}")

	def run(self):
		while True:
			# 1. Receive the image data
			data = self.socket.recv_pyobj()  # This BLOCKS until a request arrives

			if (data['module'] == 'yolo_object_detector'):
				if (data['action'] == 'detect_objects_in_image'):
					response = self.yolo_object_detector.detect_objects_in_image(data)
				else:
					response = None
			elif (data['module'] == 'path_comparator'):
				if (data['action'] == 'store_ref_path'):
					response = self.path_comparator.store_ref_path(data)
				elif(data['action'] == 'qry_path_similarity'):
					response = self.path_comparator.qry_path_similarity(data)
				else:
					response = None
			elif (data['module'] == 'llm_decisions'):
				if (data['action'] == 'classify_room_by_this_object_set_and_pic'):
					response = self.llm_decisions.classify_room_by_this_object_set_and_pic(data)
				elif (data['action'] == 'quick_classify_room_by_this_object_set'):
					response = self.llm_decisions.quick_classify_room_by_this_object_set(data)

			self.socket.send_pyobj(response)


			# # Convert to PIL Image (adjust color channels as needed)
			# if received_image.shape[2] == 3:
			# 	pil_image = Image.fromarray(received_image[:, :, ::-1])  # BGR to RGB
			# else:
			# 	pil_image = Image.fromarray(received_image)

if __name__ == "__main__":
	pcs = SemanticNavigationServer()
	pcs.run()
