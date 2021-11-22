from stylo_ga import load_langdata

data = load_langdata("rs")
lemma = 3.22
pos = 3.41
word = 0.28

df = (data["lemma"]+lemma)*(data["pos"]+pos)*(data["word"]+word)

i = 1