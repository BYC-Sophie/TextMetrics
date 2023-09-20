import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



def perplexity(text_list, model_name=None):
    '''
    Params:
        text_list: the list of text for calculation
        model_name: model used for perplexity calculation (default: gpt2-large)
    Return:
        lists of value: perplexity
    Usage:
        perplexity = perplexity(text_list)
    '''
    if model_name is None:
        model_name = 'gpt2-large'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)

    perplexity = []

    with tqdm(total=len(text_list), desc="Processing") as pbar:
        for index, text in enumerate(text_list):
            inputs = tokenizer(text, return_tensors="pt")

            input_ids = inputs["input_ids"].to(device)
            labels = inputs["input_ids"].to(device)

            with torch.no_grad():
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                loss = model(input_ids=input_ids, labels=labels).loss

            ppl = torch.exp(loss).item()

            perplexity.append(ppl)

            pbar.update(1)

    return perplexity


