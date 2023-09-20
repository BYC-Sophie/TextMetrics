from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
import pandas as pd
import numpy  as np
from tqdm.auto import tqdm
from array import array
import torch
import torchvision
import torchvision.transforms as transforms



def semantic_clarity_noise(text_list, embeddings=None, topic_model_path='topic_model', save=True):
    '''
    Params:
        text_list: the list of text for calculation
        embedding: the embedding for bertTopic (if not provided, use "all-mpnet-base-v2")
        topic_model_path: the saving file name of the topic model
        save: whether to save
    Return:
        two lists of value: semantic_clarity and semantic_noise
    Usage:
        semantic_clarity, semantic_noise = semantic_clarity_noise(text_list)
    '''

    if embeddings is None:
        sentence_model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = sentence_model.encode(text_list, show_progress_bar=True)

    hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True, min_samples=150)
    
    #remove stop words
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, hdbscan_model=hdbscan_model, calculate_probabilities=True)

    #topics, probs = topic_model.fit_transform(topic_text, x_squeezed)
    topics, probs = topic_model.fit_transform(text_list, embeddings)

    if save:
        topic_model.save(topic_model_path)

    clarify = []
    pro_list = topic_model.probabilities_
    for ps in tqdm(pro_list):
        # get all probability of the all topics
        n = len(ps)
        # calculate the maximum value in the probability
        max_value = np.max(ps)
        # calculate the expression for clarity
        result = (1/n) * np.sum(max_value - ps)
        clarify.append(result)
    
    noise = []
    for data in tqdm(pro_list):
        excess_kurtosis = calculate_excess_kurtosis(data)
        noise.append(excess_kurtosis)
    
    return clarify, noise

def calculate_excess_kurtosis(data):
    n = len(data)
    mean = np.mean(data)
    deviations_fourth_power = [(x - mean) ** 4 for x in data]
    fourth_central_moment = np.sum(deviations_fourth_power)
    second_central_moment = np.sum([(x - mean) ** 2 for x in data])

    kurtosis = n * (fourth_central_moment / (second_central_moment ** 2))
    return kurtosis
