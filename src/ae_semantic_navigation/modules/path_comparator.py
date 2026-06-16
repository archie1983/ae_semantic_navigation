import numpy as np
from PIL import Image
import os, cv2
from ae_path_compare import PathCompare
import pickle
from importlib.resources import files

class PathComparator:
	def __init__(self, use_dino = True):
		self.pc = PathCompare(use_dino)
		self.use_dino = use_dino
		self.PATH_REF_FILE = "path_refs.pkl"

		self.ref_path_store = files('ae_semantic_navigation.models').joinpath(self.PATH_REF_FILE)

		if os.path.exists(self.ref_path_store):
			with open(self.ref_path_store, "rb") as path_ref_store:
				self.path_refs = pickle.load(path_ref_store)
		else:
			self.path_refs = {}

	def store_ref_path(self, data):
		# Process the images
		received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
		received_images = received_array.reshape(data['shape'])
		pil_images = [Image.fromarray(img) for img in received_images]
		path_id = data['path_id']

		mean_path_embedding = self.pc.get_mean_path_embedding(pil_images)

		# if we already have this path_id (e.g., 'LVINGROOM_TO_KITCHEN') in the collection, then
		# append to the set of this type of paths. If not, create the set.
		if path_id in self.path_refs:
			self.path_refs[path_id].add(mean_path_embedding)
		else:
			self.path_refs[path_id] = set()
			self.path_refs[path_id].add(mean_path_embedding)

		# now store this collection
		with open(self.ref_path_store, "wb") as path_ref_store:
			pickle.dump(self.path_refs, path_ref_store)

		# Send the response back
		response = {
			'success': True
		}
		## debug
		path_id = str(path_id)
		os.makedirs(path_id, exist_ok=True)
		cnt = 0
		for img in received_images:
			cnt += 1
			cv2.imwrite(os.path.join(path_id, str(cnt) + ".png"), img)
		## /debug

		return response

	def qry_path_similarity(self, data):
		# 2. Process the images
		received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
		received_images = received_array.reshape(data['shape'])
		# get x images from the received (x, 64, 64, 3) tensor. This will be our path to compare
		pil_images = [Image.fromarray(img) for img in received_images]
		# compare that path against all reference paths
		mean_path_embedding = self.pc.get_mean_path_embedding(pil_images)
		if self.use_dino:
			cmp_res = {k: float(max(self.pc.compare_mean_path_embeddings(v, mean_path_embedding))) for k, v in self.path_refs.items()}
		else:
			cmp_res = {k: self.pc.compare_mean_path_embeddings(v, mean_path_embedding)[1] for k, v in self.path_refs.items()}

		# print(cmp_res)
		# find the best match and return both the score and the reference match buffer
		best_match = max(cmp_res.items(), key=lambda k: k[1])
		response = {
			'best_match_ref': best_match[0],
			'best_match_score': best_match[1],
			'success': True
		}
		## debug
		path_id = "tmp_cmp"
		os.makedirs(path_id, exist_ok=True)
		cnt = 0
		for img in received_images:
			cnt += 1
			cv2.imwrite(os.path.join(path_id, str(cnt) + ".png"), img)
		## /debug
		# for k, v in self.path_refs.items():
		#	cmp_res = self.pc.compare_paths(v, pil_images)

		return response