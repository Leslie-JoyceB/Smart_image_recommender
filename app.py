from flask import Flask,render_template,Response, jsonify, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import json
import os
import base64
import shutil
import tempfile
import tensorflow_hub as hub
from recommender import DataLoader, ArticlesLoader, ImagesCaptionLoader, ImageEmbeddingCreator, ImagesPredictor
import os
import pickle
#initialization for flask app
app = Flask(__name__)
api = Api(app)  # type: Api

#load data 
data_loader = DataLoader()
data = data_loader.load()
print("captioning_dataset.json loaded")

#load articles with headlines having more than 5 words
articles_loader = ArticlesLoader()
article_df, all_images = articles_loader.load(data)
print("articles and images with more than 5 words loaded")

#only take articles with captions longer than 5 words
images_loader = ImagesCaptionLoader()
image_df =  images_loader.load(all_images,article_df["idx"].values)
print("image_df loaded")

use_encoder = hub.load('../models/universal-sentence-encoder_4/')
print("use_encoder loaded")

use_img_embedding_file = "../models/use_img_embedding.pkl"
if os.path.isfile(use_img_embedding_file):
    with open(use_img_embedding_file, 'rb') as handle:
        use_img_embedding = pickle.load(handle)
else:
    embedding_creator = ImageEmbeddingCreator()
    use_img_embedding = embedding_creator.create(image_df, use_encoder)
    with open(use_img_embedding_file, 'wb') as handle:
        pickle.dump(use_img_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("use_img_embedding loaded")

images_predictor = ImagesPredictor(use_encoder, use_img_embedding, image_df)

@app.route('/recommend_images', methods=['GET'])
def recommend_images():
    text = request.args.get('text')
    print("text :", text)
    num_images = request.args.get('num_images')
    num_images = int(num_images)
    print("num_images:", num_images)
    print("type(num_images) : ", type(num_images))
    data = images_predictor.predict(text, num_images)
    return jsonify(data)


@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

if __name__ == "__main__":
    app.run()