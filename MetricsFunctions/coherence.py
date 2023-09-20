import pandas as pd
import numpy  as np
from tqdm.auto import tqdm
from array import array
import torch
import torchvision
import torchvision.transforms as transforms
import nltk
import re
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch


def Coherence_Score(df, column_name, model_name=None):
    '''
    Params:
        df: the dataframe
        column_name: the name of the text column
    Return:
        lists of value: coherence
    Usage:
        coherence = Coherence_Score(df, "Text")
    '''
    
    nltk.download('punkt')
    sent_tokenize = nltk.tokenize.sent_tokenize
    # get nltk_sentence_tokenization
    def nltk_sentence_tokenization(texts):
        sentences = sent_tokenize(texts)
        return sentences


    # set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU available. Using GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('GPU not available. Using CPU.')

    # split texts
    df['split_cur'] = df[column_name].apply(nltk_sentence_tokenization)
    split_fur = df['split_cur'].apply(lambda x: [subitem.strip() for item in x for subitem in item.split('\n')])

    if model_name is None:
        model_name = 'bert-large-uncased'
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    model.to(device)

    score_list = []
    loaded_text = split_fur.to_list()
    for instance in tqdm(loaded_text):
        if len(instance) <= 1:
            score_list.append(1)
        else:
            # Example list of texts
            texts = instance
            divide = len(instance) - 1

            # Calculate probability iteratively
            total_probability = 0.0
            previous_sentence = texts[0]

            for i in range(1, len(texts)):
                current_sentence = texts[i]

                # Tokenize the sentence
                inputs = tokenizer.encode_plus(previous_sentence, current_sentence, padding='longest', truncation=True, return_tensors='pt').to(device)

                # Perform Next Sentence Prediction
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs[0]
                probs = softmax(logits, dim=1)
                next_sentence_probability = probs[0, 0].item()

                total_probability += next_sentence_probability
                previous_sentence = current_sentence

            score = total_probability / divide

            score_list.append(score)
    return score_list

