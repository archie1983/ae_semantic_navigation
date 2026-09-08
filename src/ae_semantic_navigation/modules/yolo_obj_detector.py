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
        self.object_history = {}

    def detect_objects_in_image(self, data):
        # 2. Process the images
        received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
        received_img = received_array.reshape(data['shape'])[0]
        # get x images from the received (x, 64, 64, 3) tensor. This will be our path to compare
        pil_image = Image.fromarray(received_img)

        # # debug
        # img_path = self.store_image(pil_image)
        # reloaded_img = self.load_image(img_path)
        # reloaded_res = self.item_extractor_model(reloaded_img)
        # # /debug

        #item_extractor_res = self.item_extractor_model(pil_image)
        item_extractor_res = self.item_extractor_model.track(pil_image, persist=True)
        self.detect_unstable_item_detections(item_extractor_res)

        # print("AE Classes: ", item_extractor_res[0].boxes.cls)
        # print("AE Classes1: ", item_extractor_res[0].names)
        # print("AE All: ", item_extractor_res[0].boxes)

        item_names = [item_extractor_res[0].names[int(item)] for item in item_extractor_res[0].boxes.cls]
        # # debug
        # print("AE: item_names: ", item_names)
        # item_names_reloaded = [reloaded_res[0].names[int(item)] for item in reloaded_res[0].boxes.cls]
        # print("AE: item_names: ", item_names, " reloaded: ", item_names_reloaded, " path: ", img_path)
        # #/debug

        response = {
            'item_names': item_names,
            'success': True
        }

        return response

    def detect_unstable_item_detections(self, yolo_res):
        if yolo_res[0].boxes is None or yolo_res[0].boxes.id is None:
            return

        for box in yolo_res[0].boxes:
            obj_id = int(box.id)
            cls = int(box.cls)
            conf = float(box.conf)
            bbox = box.xyxy.tolist()
            cur_name = yolo_res[0].names[int(cls)]

            if obj_id not in self.object_history:
                self.object_history[obj_id] = []

            # Append current detection
            self.object_history[obj_id].append({
                'class': cls,
                'confidence': conf,
                'bbox': bbox,
                'name': cur_name
            })

            # Check if class changed compared to last frame
            if len(self.object_history[obj_id]) > 1:
                prev_class = self.object_history[obj_id][-2]['class']
                prev_name = self.object_history[obj_id][-2]['name']
                prev_conf = self.object_history[obj_id][-2]['conf']
                if cls != prev_class:
                    print(f"Object {obj_id} changed from {prev_name} to {cur_name}! CONF {conf} to {prev_conf}. Potentially affected: {len(self.object_history[obj_id])} frames")

    def store_image(self, img):
        ## debug
        path_id = "tmp_img"
        os.makedirs(path_id, exist_ok=True)
        self.img_cnt += 1
        img = np.stack([img])
        img = img[0]
        cv2.imwrite(os.path.join(path_id, str(self.img_cnt) + ".png"), img)
        print("AE: name: ", str(self.img_cnt), ".png")
        return os.path.join(path_id, str(self.img_cnt) + ".png")

    def load_image(self, img_path):
        img = Image.open(img_path)
        return np.stack([img])[0]