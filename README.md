# parallel-doc-embeds

this repository is created to ease the replication of results (and further reasearch on the subject) of
Parallel Stylometric Document Embeddings with Deep Learning based Language Models in Literary Authorship Attribution

abstract:This paper explores effectiveness of parallel stylometric document embeddings in solving the authorship attribution task by testing a novel approach on literary texts in seven different languages, totaling in 7,051 unique 10,000-token chunks from 700 PoS and lemma annotated documents. We used these documents to produce four document embedding models using stylo R package (word-based, lemma-based, PoS-trigrams-based and PoS-mask-based) and one document embedding model using mBERT for each of seven languages. We created further derivations of these embeddings in the form of average, product, minimum, maximum and vector norm of these document embedding matrices and tested them both including and excluding the mBERT-based document embeddings for each language. Finally, we trained several perceptrons on the portions of the dataset in order to procure adequate weights for a weighted combination approach. We tested standalone (two baselines) and composite embeddings for classification accuracy, precision, recall, weithed and macro-averaged $F_1$-score, compared them with one another and have found that for each language most of our methods outperform the baselines (with a couple of methods outperforming all baselines for all languages), with or without mBERT inputs which are found to have no positive impact on the results of our combination methods.

in order to reproduce the results:

(0. install python, r, and the package requiremnts)

if you do not want to recreate the embeddings find them in https://github.com/procesaur/parallel-doc-embeds/tree/main/data/document_embeds and skip steps 1, 2 and 3

1. unzip chunks from https://github.com/procesaur/parallel-doc-embeds/tree/main/data/zipped_chunks into folder /data/chunks
2. run https://github.com/procesaur/parallel-doc-embeds/blob/main/data/get_stylo_dist.R in order to produce stylo-based document embeddings
3. run https://github.com/procesaur/parallel-doc-embeds/blob/main/bert_embeds.py in order to produce mbert-based chunks

4. run main function from main.py in order to procure the classification results for each language and each embedding
