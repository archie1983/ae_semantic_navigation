from ae_llm_navigation_decisions import LLMDecisionMaker, LLMType, SVCRoomClassifier, RoomType
import numpy as np
from PIL import Image

class LLMDecisions:
	def __init__(self):
		self.ldm = LLMDecisionMaker(LLMType.MINISTRAL_3_3b_instruct_nf4_bnb)
		self.src = SVCRoomClassifier()

	def classify_room_by_this_object_set_and_pic(self, data):
		received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
		received_img = received_array.reshape(data['shape'])[0]
		# get x images from the received (x, 64, 64, 3) tensor. This will be our path to compare
		pil_image = Image.fromarray(received_img)
		obj_set = data['obj_set']

		rt_llm, llm_text = self.ldm.classify_room_by_this_object_set_and_pic(obj_set=obj_set, img_bytes=pil_image)
		return rt_llm

	def quick_classify_room_by_this_object_set(self, data):
		obj_set = data['obj_set']
		rt_probs = self.src.predict_proba(obj_set)
		top_candidate = sorted(rt_probs.items(), key=lambda x: x[1], reverse=True)[0]
		if top_candidate[1] > 0.67:
			return RoomType.interpret_label(top_candidate[0])
		else:
			return RoomType.NOT_KNOWN