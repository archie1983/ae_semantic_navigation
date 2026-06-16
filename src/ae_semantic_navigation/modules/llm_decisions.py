from ae_llm_navigation_decisions import LLMRoomClassifier, LLMType
import numpy as np
from PIL import Image

class LLMDecisions:
	def __init__(self):
		self.lrc = LLMRoomClassifier(LLMType.MINISTRAL_3_3b_instruct_nf4_bnb)

	def classify_room_by_this_object_set_and_pic(self, data):
		received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
		received_img = received_array.reshape(data['shape'])[0]
		# get x images from the received (x, 64, 64, 3) tensor. This will be our path to compare
		pil_image = Image.fromarray(received_img)

		obj_set = None
		img_uri = None
		rt_llm, llm_text = self.lrc.classify_room_by_this_object_set_and_pic(obj_set={"Scales", "bathtub", "toothbrush"}, img_uri=None)
		return rt_llm