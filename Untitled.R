#' Title: HW2
#' Purpose: HW2
#' Author: Atalia Horenshtien
  #' Date: Jan 28 2020
#' 

# Set the working directory
setwd("~/Documents/MsBA/textAnalytics/Cases/classification_modeling")

# Libs
library(tm)
library(RTextTools)
library(yardstick)


# Options & Functions
options(stringsAsFactors = FALSE, scipen = 999)
Sys.setlocale('LC_ALL','C')

tryTolower <- function(x){
  y = NA
  try_error = tryCatch(tolower(x), error = function(e) e)
  if (!inherits(try_error, 'error'))
    y = tolower(x)
  return(y)
}

remove_UTF8_dots <- function(x){
  y <- iconv(x, "UTF-8", "ASCII", ".")
  y <- gsub("[[:punct:]]+", ' ', y)
  return(y)
}

cleanCorpus<-function(corpus, customStopwords){
  corpus <- tm_map(corpus, content_transformer(remove_UTF8_dots))
  corpus <- tm_map(corpus, content_transformer(qdapRegex::rm_url))
  corpus <- tm_map(corpus, content_transformer(replace_contraction)) 
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(removePunctuation))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tryTolower))
  corpus <- tm_map(corpus, removeWords, customStopwords)
  return(corpus)
}

# Create custom stop words
stops <- c(stopwords('SMART'))

# load data
trainingTxt <- read.csv("student_tm_case_training_data.csv")
txt <- paste(trainingTxt$rawText)
testTxt <- read.csv("student_tm_case_score_data.csv" )

# Subset to avoid overfitting
set.seed(1234)
idx <- sample(1:nrow(trainingTxt), floor(length(txt)*.70))
train <- trainingTxt[idx, ]
test  <- trainingTxt[-idx, ] 

# Clean, extract text and get into correct object
cleanTrain <- cleanCorpus(VCorpus(VectorSource(train$rawText)), stops)
cleanTrain_df <- data.frame(text = unlist(sapply(cleanTrain, `[`, "content")),
                         stringsAsFactors=F)
### Extract the cleaned text 
training_data <- unlist(pblapply(cleanTrain, content))
### Collapse each into a single "subject" ie a single document
training_data <- paste(training_data, collapse = ' ')
training_data <- VCorpus((VectorSource(training_data)))
##TDM,TDMm
training_dataTDM  <- TermDocumentMatrix(training_data)
training_dataTDMm <- as.matrix(training_dataTDM)

# visualization

## Load the needed libraries
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(ggthemes)

# Frequent words
### frequency analysis
training_dataSums <- rowSums(training_dataTDMm)
training_dataFreq <- data.frame(word=names(training_dataSums),frequency=training_dataSums)

###  Remove the row attributes meta family
rownames(training_dataFreq) <- NULL

### top frequent word
### sunset for a smaller data to plot based on condition
topWords <- subset(training_dataFreq, training_dataFreq$frequency >= 40) 
topWords  <- topWords[order(topWords$frequency, decreasing= T),]
head(topWords, 15)

###  Chg to factor for ggplot
topWords$word <- factor(topWords$word, levels=unique(as.character(topWords$word))) 

### visualization - Simple barplot
ggplot(topWords, aes(x=word, y=frequency)) + 
  geom_bar(stat="identity", fill='#ADD8E6') + 
  coord_flip()+ theme_gdocs() +
  geom_text(aes(label=frequency), colour="black",hjust=1.25, size=3.0)


# Word Cloud
### Make a common cloud
commonality.cloud(training_dataTDMm, 
                  max.words=75, 
                  random.order=FALSE,
                  colors='blue',
                  scale=c(3.5,0.25))


#Module - Training

trainDTMm <- create_matrix(cleanTrain_df, language="english")

# Create the container
container <- create_container(matrix    = trainDTMm,
                              labels    = trainingTxt$label[idx], 
                              trainSize = 1:length(idx), 
                              virgin=FALSE)

# Build Models
models <- train_models(container, algorithms=c("GLMNET","SVM")) 

# Score the original training data
results <- classify_models(container, models)
head(results)

# Append Actuals
results$actual <- trainingTxt$label[idx]

# Confusion Matrix
table(results$GLMNET_LABEL, results$actual)
table(results$SVM_LABEL, results$actual)

# Accuracy GLMNET_LABEL
autoplot(conf_mat(table(results$GLMNET_LABEL, results$actual)))
accuracy(table(results$GLMNET_LABEL, results$actual))

# Accuracy SVM_LABEL
autoplot(conf_mat(table(results$SVM_LABEL, results$actual)))
accuracy(table(results$SVM_LABEL, results$actual))

# from the analysis I will choose the GLMNET module.

# Module - Testing 

# Clean, extract text and get into correct object
cleanTest <- cleanCorpus(VCorpus(VectorSource(test$rawText)), stops)
cleanTest_df <- data.frame(text = unlist(sapply(cleanTest, `[`, "content")),
                        stringsAsFactors=F)

#combine the matrices to the original to get the tokens joined
allDTM <- rbind(cleanTrain_df, cleanTest_df)
allDTMm <- create_matrix(allDTM, language="english")
containerTest <- create_container(matrix    = allDTMm,
                                  labels    = trainingTxt$label, 
                                  trainSize = 1:length(idx),
                                  testSize  = (length(idx)+1):nrow(allDTM),
                                  virgin=T)
# Build Model
testFit <- train_models(containerTest, algorithms=c("GLMNET"))
resultsTest <- classify_models(containerTest, testFit)

# Append Actuals
resultsTest$actual <- trainingTxt$label[-idx]

# Confusion Matrix
summary(resultsTest$GLMNET_PROB)
table(resultsTest$GLMNET_LABEL, resultsTest$actual)


# Module - Build the scoring data

# Clean, extract text and get into correct object
clean_Actual_Test <- cleanCorpus(VCorpus(VectorSource(testTxt$rawText)), stops)
clean_Actual_Test_df <- data.frame(text = unlist(sapply(clean_Actual_Test, `[`, "content")),
                        stringsAsFactors=F)

# combine the matrices to the original to get the tokens joined
allDTM <- rbind(cleanTrain_df, clean_Actual_Test_df)
allDTMm <- create_matrix(allDTM, language="english")
Actual_containerTest <- create_container(matrix    = allDTMm,
                                  labels    = trainingTxt$label, 
                                  trainSize = 1:length(idx),
                                  testSize  = (length(idx)+1):nrow(allDTM),
                                  virgin=T)

# Build module
Actual_testFit <- train_models(Actual_containerTest, algorithms=c("GLMNET"))
Actual_resultsTest <- classify_models(Actual_containerTest, Actual_testFit)

# Add the results to my scored data
testTxt$GLMNET_LABEL <- Actual_resultsTest$GLMNET_LABEL[1:nrow(Actual_resultsTest)]
testTxt$GLMNET_PROB <- Actual_resultsTest$GLMNET_PROB[1:nrow(Actual_resultsTest)]
head(testTxt)

write.csv(testTxt,'scored_data.csv')

# End