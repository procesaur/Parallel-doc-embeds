library(stylo)

# ------------------SETTINGS-----------------------------

# set working directory
home <- "../Desktop/stilometrija/novo/chunks/"
langs <- c("deu", "eng", "fra", "hun", "por", "srp", "slv" )
  
# function to generate embeds
get_embeds <- function(texts, h, incarnation){

    # generate frequencies from text bits
    freq.list <- make.frequency.list(texts, head = h)
    word.frequencies <- make.table.of.frequencies(corpus=texts, features=freq.list)

    # save frquencies model name
    freqcsvname <- paste(incarnation, ".csv", sep="_freq")
    write.table(word.frequencies,file=freqcsvname) 

    # calculate distances between each text bit and add to total average
    distances <- as.matrix(dist.wurzburg(word.frequencies))   
 
    # save distancies model name
    csvname <- paste(incarnation, ".csv", sep="")
    write.table(distances,file=csvname, format(ttf.all, digits=7))
}

# --------------------PIPE------------------------------
for (lang in langs){

  path <- paste(home, lang, sep="")

  # get list of folders containing text incarnations (words, lemmas, POS, etc.) 
  incarnations <- list.dirs(path = path, full.names = TRUE, recursive = FALSE)
#c(paste(path, "word", sep="/"), paste(path, "lemma", sep="/"), paste(path, "pos", sep="/"))
      
  # go through each text incarnations folder
  for (incarnation in incarnations){

    # apply hyper parameters > it goes different for POS
    
    #if it is pos get trigrams and reduce head size
    if (grepl("pos", incarnation, fixed="TRUE")){
	texts <- load.corpus.and.parse(files="all", corpus.dir=incarnation, corpus.lang="Other", ngram.size=3)
	get_embeds(texts, 300, incarnation)
    }
    else if (grepl("masked", incarnation, fixed="TRUE")){
    	for(i in 1:5){
		texts <- load.corpus.and.parse(files="all", corpus.dir=incarnation, corpus.lang="Other", ngram.size=i)
		get_embeds(texts, 500, paste(incarnation, i, sep="_"))
	}
    }
    else {
      texts <- load.corpus.and.parse(files="all", corpus.dir=incarnation, corpus.lang="Other", ngram.size=1)
	get_embeds(texts, 800, incarnation)
    }

  }
}
