# parallel-doc-embeds

this repository is created to ease the replication of results (and further reasearch on the subject) of
Parallel Stylometric Document Embeddings with Deep Learning based Language Models in Literary Authorship Attribution


in order to reproduce the results:

(0. install python, r, and the package requiremnts)

if you do not want to recreate the embeddings find them in https://github.com/procesaur/parallel-doc-embeds/tree/main/data/document_embeds and skip steps 1, 2 and 3

1. unzip chunks from https://github.com/procesaur/parallel-doc-embeds/tree/main/data/zipped_chunks into folder /data/chunks
2. run https://github.com/procesaur/parallel-doc-embeds/blob/main/data/get_stylo_dist.R in order to produce stylo-based document embeddings
3. run https://github.com/procesaur/parallel-doc-embeds/blob/main/bert_embeds.py in order to produce mbert-based chunks

4. run main function from main.py in order to procure the classification results for each language and each embedding
