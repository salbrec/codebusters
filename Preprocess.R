library(readr)
library(textmineR)
library(wordcloud)
set.seed(123)
NewsDatasetLabeled <- read_csv("C:/Users/diior/Desktop/NewsDatasetLabeled.csv")
data<-NewsDatasetLabeled
Relevant=data[data$class==1,]
Not_Relevant=data[data$class==0,]

N=dim(data)[1]
N_Rel=dim(Relevant)[1]
N_not_Rel=dim(Not_Relevant)[1]
random_idxs=sample(c(1:(N_Rel)),replace=T,size=(N_not_Rel-N_Rel))
Extended_relevant=Relevant[random_idxs,]

#a <- read.table("Matrix.csv",header = TRUE,sep=" ")








fulldata=rbind(data,Extended_relevant)
dtm <- CreateDtm(doc_vec = fulldata$Text, # character vector of documents
                 doc_names = fulldata$NumArt, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 stem_lemma_function = function(x) SnowballC::wordStem(x, "porter"),
                
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo
                 cpus=2) # default is all available cpus on the system


dtm_origi <- CreateDtm(doc_vec = data$Text, # character vector of documents
                 doc_names = data$NumArt, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 stem_lemma_function = function(x) SnowballC::wordStem(x, "porter"),
                 
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo
                 cpus=2) # default is all available cpus on the system




# for (i in c(1,2,3,4,5,6,7,8,9,10,c(15:5:60))){
# prova<-length(which(colSums(dtm)>i))
# print(c(i,prova))
#   }
#   
  
# dtm=dtm[,colSums(dtm)>60]
# 
# 
# dtm_Relevant= CreateDtm(doc_vec = Relevant$Text, # character vector of documents
#                         doc_names = Relevant$NumArt, # document names
#                         ngram_window = c(1, 2), # minimum and maximum n-gram length
#                         stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
#                                          stopwords::stopwords(source = "smart")), # this is the default value
#                         lower = TRUE, # lowercase - this is the default value
#                         remove_punctuation = TRUE, # punctuation - this is the default
#                         remove_numbers = TRUE, # numbers - this is the default
#                         verbose = FALSE, # Turn off status bar for this demo
#                         cpus=2) # default is all available cpus on the system
# 
# dtm_Not_Relevant= CreateDtm(doc_vec = Not_Relevant$Text, # character vector of documents
#                             doc_names = Not_Relevant$NumArt, # document names
#                             ngram_window = c(1, 2), # minimum and maximum n-gram length
#                             stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
#                                              stopwords::stopwords(source = "smart")), # this is the default value
#                             lower = TRUE, # lowercase - this is the default value
#                             remove_punctuation = TRUE, # punctuation - this is the default
#                             remove_numbers = TRUE, # numbers - this is the default
#                             verbose = FALSE, # Turn off status bar for this demo
#                             cpus=2) # default is all available cpus on the system


Tf=TermDocFreq(dtm)
Tf_orig=TermDocFreq(dtm_origi)
# Tf_Relevant=TermDocFreq(dtm_Relevant)
# Tf_Not_Relevant=TermDocFreq(dtm_Not_Relevant)
# which.max(Tf$term_freq)
# Tf[which.max(Tf$term_freq),1]
# Tf_Relevant[which.max(Tf_Relevant$term_freq),1]
# Tf_Not_Relevant[which.max(Tf_Not_Relevant$term_freq),1]





pal <- brewer.pal(8, "Dark2")




wordcloud(Tf$term,freq = Tf$term_freq,max.words=50,colors=pal)
# wordcloud(Tf_Relevant$term,freq = Tf_Relevant$term_freq,max.words=50,colors =pal)
# wordcloud(Tf_Not_Relevant$term,freq = Tf_Not_Relevant$term_freq,max.words=50,colors=pal)


Tf$term

tfidf <- t(dtm[ , Tf$term ]) * Tf$idf

tfidf <- t(tfidf)

Tf_orig$term

tfidf_orig <- t(dtm_origi[ , Tf_orig$term ]) * Tf_orig$idf

tfidf_orig<- t(tfidf_orig)

idd=which(colSums(tfidf)>2000)
idd_orig=which(colSums(tfidf_orig)>1000)




tfidf <- tfidf[,colSums(tfidf)>2000]
tfidf_orig <- tfidf_orig[,colSums(tfidf_orig)>1000]

a=matrix(0,dim(tfidf)[1],dim(tfidf)[2])
a_orig=matrix(0,dim(tfidf_orig)[1],dim(tfidf_orig)[2])

idx=summary(tfidf)
idx_orig=summary(tfidf_orig)


a[cbind(idx$i,idx$j)]<-idx$x
a_orig[cbind(idx_orig$i,idx_orig$j)]<-idx_orig$x
colnames(a)<-Tf$term[idd]
colnames(a_orig)<-Tf_orig$term[idd_orig]

Reduced=prcomp(a,rank=300)
Full=data.frame(fulldata$class,a)
Full_orig=data.frame(data$class,a_orig)
colnames(Full)[1]<-"class"
colnames(Full_orig)[1]<-"class"

rel=Full[Full$class==1,]
not_rel=Full[Full$class==0,]
write.table(rel,"Relevant.csv",row.names = FALSE,col.names=TRUE)
write.table(not_rel,"Not_Relevant.csv",row.names=FALSE,col.names = TRUE)
write.table(xx,"full.csv",row.names = FALSE,col.names=TRUE)
write.table(Full_orig,"full_orig.csv",row.names = FALSE,col.names=TRUE)


prova=read.table("Relevant.csv",header=TRUE,sep=" ")

