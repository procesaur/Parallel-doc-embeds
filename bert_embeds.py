import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import numpy as np
import os

from transformers import BertTokenizer, BertModel
import torch

from helpers import get_langs


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
ALPHABET = "([A-Za-z])"
PREF = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
SUFF = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov|me|edu)"
DIGITS = "([0-9])"


def paragraph_to_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "i.e." in text:
        text = text.replace("i.e.", "i<prd>e<prd>")
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def embed_sentence(sentence, tokenizer, model):
    # Tokenize input
    sentence = tokenizer.tokenize("[CLS] " + sentence + " [SEP]")

    if len(sentence) > 512:
        sentence = sentence[:512]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
    # In our case we only have one sentence, i.e. one segment id
    segment_ids = [0] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    token_tensor = torch.tensor([indexed_tokens]).to(device)
    segment_tensor = torch.tensor([segment_ids]).to(device)

    with torch.no_grad():
        # Output state of last 4 layers
        output = model(token_tensor, segment_tensor, output_hidden_states=True)["hidden_states"][-4:]
        token_embeddings = torch.stack(output, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = torch.sum(token_embeddings, dim=0)
        sentence_embedding_sum = torch.sum(token_embeddings, dim=0)

    return sentence_embedding_sum


def generate_embeddings(documents):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    model.eval()

    embeddings_per_document = []
    embeddings_per_paragraph = []

    with tqdm(documents, unit="document", desc=f"Generating embeddings") as pbar:
        for doc in pbar:

            doc_embedding = torch.zeros(768).to(device)
            par_embeddings = []
            sentence_count = 0

            par = doc

            par_embedding = torch.zeros(768).to(device)
            sentences = paragraph_to_sentences(par)

            for sent in sentences:
                sentence_count += 1
                sent_embedding = embed_sentence(sent, tokenizer, model)
                par_embedding.add_(sent_embedding)

            doc_embedding.add_(par_embedding)
            par_embeddings.append(par_embedding)

            embeddings_per_document.append(doc_embedding / sentence_count)
            embeddings_per_paragraph.append(par_embeddings)

    return embeddings_per_document  # , embeddings_per_paragraph


for lang in get_langs("./data/chunks"):
    chunksx = os.listdir("./data/chunks/" + lang + "/word/")
    chunksnames = sorted(chunksx)
    chunks = []
    for i, chunk in enumerate(chunksnames):
        with open("./data/chunks/" + lang + "/word/" + chunk, "r", encoding="utf-8") as f:
            chunks.append(f.read())
    chunksnames = [x.split(".")[0] for x in chunksnames]
    documents = []
    for chunk in chunks:
        documents.append(chunk.replace("\n", " "))
    document_embedings = generate_embeddings(documents)
    de = [x.cpu() for x in document_embedings]
    document_embedings = np.stack(de)
    pairwise_similarities=cosine_similarity(document_embedings)
    df = pd.DataFrame(pairwise_similarities, columns=chunksnames, index=chunksnames)
    df.to_csv("./data/document_embeds/" + lang + "/bert.csv", sep=" ")
