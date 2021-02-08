
import os
import pickle
import numpy as np
import time
import pandas as pd
import json

class DataLoader:
    def load(self, path='../data/captioning_dataset_part1.json'):
        with open(path) as json_file:
            data = json.load(json_file)
        return data

class ArticlesLoader(object):
    def load(self, data):
        min_words = 5
        ids, headlines, urls, all_images, articles, lens = [], [], [], [], [], []

        for i,idx in enumerate(data):
            try:
                headline = data[idx]['headline']['main'].strip()
                num_words = len(headline.split(' '))
                
                # removing article headlines if the number of words in the headline is less than 5
                if num_words < min_words:
                    continue

                headlines.append(headline)
                lens.append(num_words)
                all_images.append(data[idx]['images'])
                urls.append(data[idx]['article_url'])
                articles.append(data[idx]['article'])
                ids.append(idx)
            
            # deals with situations where articles are missing a headline
            except:
                continue

        # creating a dataframe with our results
        article_df = pd.DataFrame({'idx': ids, 'headline': headlines, 'text': articles, 'url': urls, 'num_words': lens})
        #print(f'Number of Articles: {article_df.shape[0]}')
        #article_df.head()
        return article_df, all_images

class ImagesCaptionLoader(object):
    def load(self, all_images,ids, min_words = 5):
        img_captions, img_article_ids, caption_lens, nums = [], [], [], []

        for i, img in enumerate(all_images):
            for k in img.keys():
                caption = img[k].strip()
                num_words = len(caption.split(' '))
                
                # removing article headlines if the number of words in the headline is less than 5
                if num_words < min_words:
                    continue
                
                nums.append(k)
                img_captions.append(caption)
                caption_lens.append(num_words)
                img_article_ids.append(ids[i])
                
        # creating a dataframe with our results        
        image_df = pd.DataFrame({'article_idx': img_article_ids, 'caption': img_captions, 'num_words': caption_lens, 'number': nums})
        
        return image_df


class ImageEmbeddingCreator(object):
    def create(self, image_df, use_encoder):
        start_time = time.time()
        use_img_embedding = np.zeros((len(image_df),512))
        for i, text in enumerate(image_df.caption.values):
            if i % 100000 == 0 and i > 0:
                print(f'{i} out of {len(image_df.caption.values)} done in {time.time() - start_time:.2f}s')
            emb = use_encoder([text])
            use_img_embedding[i] = emb
        print(f'{i} out of {len(image_df.caption.values)} done')
        #use_img_embedding_normalized = use_img_embedding/np.linalg.norm(use_img_embedding,axis=1).reshape(-1,1)
        return use_img_embedding


class ImagesPredictor(object):
    def __init__(self,use_encoder,use_img_embedding, image_df):
        self.use_encoder = use_encoder
        self.use_img_embedding = use_img_embedding
        self.image_df = image_df


    def predict(self, headline, k = 2):
        """
        Predicts the closest matching image caption given an article headline
        Returns a list of image ids
        """
        # finding the embedding. No pre-processing is needed
        emb =  self.use_encoder([headline])
        
        # normalizing the embeddings
        emb = emb/np.linalg.norm(emb)
        print("emb :", emb.shape)
        print("self.use_img_embedding :", self.use_img_embedding)
        # calculating the cosine distance. 
        # since the embeddings are normalized: this is the dot product of the embedding vector and the matrix
        scores_images = np.dot(emb,self.use_img_embedding.T).flatten()
        print("scores_images :", scores_images)
        print("-scores_images :", -scores_images)
        # predict top k images
        top_k_images = self.image_df.iloc[np.argsort(-scores_images)[:k]]
        idx = top_k_images.article_idx.values
        nums = top_k_images.number.values
        images = []

        for index in range(k):
            images.append("https://smartirstorage.blob.core.windows.net/smartircontainer{0}_{1}.jpg".format(idx[index], nums[index]))
                # images.append("https://imagesrecommender.blob.core.windows.net/data/resized/{0}_{1}.jpg".format(idx[index], nums[index])) -- code original 

        return images
