#################Clean one########################


#Library Declaration
library(readr)
library(textmineR)
library(wordcloud)
library(caret)
library(pROC)
library(glmnet)

#Data Input and preliminary stuff
set.seed(123)
data_path="C:/Users/diior/Desktop/NewsDatasetLabeled.csv"
data <- read_csv(data_path) #importing data
data$Date<-as.Date(data$Date)
data=data[data$Date>as.Date("2018-12-31"),] #Removing unlabelled observation
Relevant=data[data$class==1,] #Relevant News data
Not_Relevant=data[data$class==0,] #Not relevant News Data

N_Rel=dim(Relevant)[1]
N_Not=dim(Not_Relevant)[1]
N=dim(data)[1]

Ratio=round(N_Rel/N,2) #Percentage of relevant on the whole data set.

Test_size=200 #Test data size


#Sample from relevant news to obtain relevant part of test set.
sample_rel=sample(1:N_Rel,size=Ratio*Test_size,replace=F) 
test_data_Rel=Relevant[sample_rel,]
train_data_Rel=Relevant[-sample_rel,]

#Sample from relevant news to obtain not relevant part of test set.

sample_not_rel=sample(1:N_Not,size=((1-Ratio)*Test_size),replace=F)
test_data_not_Rel=Not_Relevant[sample_not_rel,]
train_data_not_Rel=Not_Relevant[-sample_not_rel,]

#Merge the two data to obtain one train data set.

test_data=rbind(test_data_Rel,test_data_not_Rel) #this test set has the same ratio of relevant case of the original set
train_data=rbind(train_data_Rel,train_data_not_Rel)

N_Rel_train=dim(train_data_Rel)[1]
N_Not_train=dim(train_data_not_Rel)[1]
N_train=dim(train_data)[1]

fulldata=rbind(train_data,test_data)

#Preprocessing data

dtm <- CreateDtm(doc_vec = fulldata$Text, # character vector of documents
                 doc_names = fulldata$NumArt, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase 
                 remove_punctuation = TRUE, # punctuation 
                 stem_lemma_function = function(x) SnowballC::wordStem(x, "porter"),#some stemming
                 remove_numbers = TRUE, # numbers 
                 verbose = FALSE, # Turn off status bar
                 cpus=2) # parallelizing


Tf=TermDocFreq(dtm) #creating term frequency and tf-idf
tfidf <- t(dtm[ , Tf$term ]) * Tf$idf
tfidf <- t(tfidf)

summary(colSums(tfidf)) #some descriptives about cumulative distribution of tfidf 

# 300 400 600
to_keep=which(colSums(tfidf)>300) #Keep columns with a cumulative tfidf bigger than 300 in order to reduce features.
#it can be cross-validated or chosen in a more smart way. This treshold is strictly data dependent!!!!!!!!!!

tfidf <- tfidf[,to_keep]

dim(tfidf)

#Make the tfidf matrix a full matrix (not a sparse one).
M_tfidf=matrix(0,dim(tfidf)[1],dim(tfidf)[2]) #memory allocation
idx=summary(tfidf)
M_tfidf[cbind(idx$i,idx$j)]<-idx$x 
colnames(M_tfidf)<-Tf$term[to_keep]
Full=data.frame(fulldata$class,M_tfidf)
colnames(Full)[1]<-"class"

train_data_full=Full[1:N_train,]
test_data_full=Full[-(1:N_train),]

#####################AUGMENTING DATA##########################

synt=c()  #allocating some memory

#Following function creates synthetic relevant cases sampling with replacement 
#from empirical distribution of words of relevant news. We are not using relevant cases that are in the test set,
#in order to avoid bias.

synt=apply(train_data_full[train_data_full$class==1,],2,
           function(x) sample(x,size=(N_Not_train-N_Rel_train),replace=T)) #fast way to act on matrices

train_data_full=rbind(train_data_full,synt) #augmented train data

write.table(train_data_full,"train_data_full_685.csv",row.names=FALSE)
write.table(test_data_full,"test_data_full_685.csv",row.names=FALSE)