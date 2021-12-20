library(stylo)

# ------------------SETTINGS-----------------------------

# set working directory
home <- "../Desktop/stilometrija/novo/chunks/"
langs <- c("deu", "eng", "fra", "hun", "por", "srp", "slv" )

try_times = 10
  
# hyper parameters for text load

h <- 800
ns_pos <- 3
h_pos <- 300
  
# --------------------PIPE------------------------------
for (lang in langs){

  path <- paste(home, lang, sep="")


  # get list of folders containing text incarnations (words, lemmas, POS, etc.) 
  incarnations <- c(paste(path, "word", sep="/"), paste(path, "lemma", sep="/"), paste(path, "pos", sep="/"))
  n_inc <- length(incarnations)
      
  # go through each text incarnations folder
  for (incarnation in incarnations){

    # apply hyper parameters > it goes different for POS
    
    #if it is pos get trigrams and reduce head size
    if (grepl("pos", incarnation, fixed="TRUE")){
      h <- h_pos
	texts <- load.corpus.and.parse(files="all", corpus.dir=incarnation, corpus.lang="Other", ngram.size=ns_pos)
    }
    else {
      texts <- load.corpus.and.parse(files = "all", corpus.dir=incarnation", corpus.lang="Other", ngram.size=1)
    }

    # generate frequencies from text bits
    freq.list <- make.frequency.list(texts, head = h)
    word.frequencies <- make.table.of.frequencies(corpus=texts, features=freq.list)
    rm(texts)

    # save frquencies model name
    freqcsvname <- paste(incarnation, ".csv", sep="_freq")
    write.table(word.frequencies,file=freqcsvname) 


    # calculate distances between each text bit and add to total average
    distances <- as.matrix(dist.wurzburg(word.frequencies))   
 
    # save distancies model name
    csvname <- paste(incarnation, ".csv", sep="")
    write.table(distances,file=csvname)
    
  }
}
