from transformers import BertTokenizer, BertForTokenClassification
import torch
import numpy as np

def tokenize_batch(tokenizer, text_batch):

    inputs = tokenizer(text_batch, return_tensors="pt", padding=True)

    # compute seq lens
    s_lens = []
    for att_mask in inputs['attention_mask']:
        index = 0
        found = False
        for index, val in enumerate(att_mask):
            if val == 0:
                found = True
                break
        if not found:
            index += 1
        s_lens.append(index)

    return inputs, s_lens


def feed_through_model(model, inputs):

    logits = model(**inputs).detach().numpy()
    preds = np.argmax(logits, -1)
    return preds, logits

def get_all_batch_tokens(tokenizer, batch_input, s_lens):

    tot_tokens = []
    for s_len, input_ids in zip(s_lens, batch_input['input_ids']):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tot_tokens += tokens[:s_len]
    return tot_tokens

def initialize_tokenizer(model_path):

    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer

def initialize_model(model_path):

    model = BertForTokenClassification.from_pretrained(model_path)
    return model
