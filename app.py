from flask import Flask
from flask_restful import Resource, Api
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import keras
import cv2
import os

def images_load(dir_path):
    img_list = []
    filename_list = []
    for root, _, files in os.walk(dir_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            img = image_load(img_path)
            img_list.append(img)
            filename_list.append(filename)
    img_list = np.array(img_list)
    img_list.reshape((-1, 224, 224, 3))
    return img_list, filename_list

def image_load(img_path):
    img_size = (224, 224)
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img 

def get_feature_vector(model, images):
    feature_map = model.predict(images)
    np_feature_map = np.array(feature_map)
    feature_vector = np_feature_map.reshape((-1, 2048)) 
    return feature_vector

def get_ranked_list(query_vecs, reference_vecs, ref_filenames):
    sim_matrix = np.dot(query_vecs, reference_vecs.T)
    indices = np.argsort(sim_matrix, axis=1)
    indices = np.flip(indices, axis=1)
    ranked_list = [ref_filenames[idx] for idx in indices[0]]
    return ranked_list

app = Flask(__name__)
api = Api(app)

class Inference(Resource):
    def get(self):
        ref_imgs, ref_filenames= images_load('./references')
        query_imgs, _ = images_load('./query')
        resnet = ResNet50(weights='imagenet',
                    include_top=False)
        resnet.summary()
        ref_vecs = get_feature_vector(
                            resnet, ref_imgs)
        query_vecs = get_feature_vector(
                            resnet, query_imgs)
        ranked_list = get_ranked_list(query_vecs, ref_vecs, ref_filenames)
        return {'recommend_list' : ranked_list} 

api.add_resource(Inference, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

