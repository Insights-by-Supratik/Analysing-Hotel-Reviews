## INTRODUCTION ----

# Setting working directory
setwd(choose.dir())

# Clear Environment
rm(list = ls())

# Load required packages
library(tm)
library(cld2)
library(dplyr)
library(ldatuning)
library(ggplot2)
library(tokenizers)
library(textstem)
library(wordcloud)
library(LDAvis)
library(syuzhet)
library(topicmodels)

# Set seed for reproducibility
set.seed(467)

## DEFINE FUNCTIONS ----
# Create a custom function to remove symbols and special characters
remove_special_chars <- function(text) {
  cleaned_text <- gsub("[^[:alnum:] ]", "", text)
  return(cleaned_text)
}

# Define a custom function to replace accented characters with ASCII encoding
replace_accented_chars <- function(text) {
  cleaned_text <- iconv(text, to = "ASCII//TRANSLIT")
  return(cleaned_text)
}

## DATA LOADING & EXTRACTION ----

# Read data
df <- read.csv("HotelsData.csv", stringsAsFactors = FALSE)

# Copy to a data frame
data <- df

# Checking missing values
sum(is.na(data))

# Extract data in english
eng_data <- subset(data, detect_language(data$Text.1) == "en")

# Extract a Sample of 2,000 records
sample_data <- sample_n(eng_data, size = 2000)

# Rename the columns
colnames(sample_data)
colnames(sample_data) <- c("Rating", "Review")

## DATA CLEANING & PREPROCESSING ----

# Create a corpus
corpus <- Corpus(VectorSource(sample_data$Review))
corpus$content[8]

# Tokenization of corpus
corpus_token <- tokenize_words(corpus$content)
corpus_token[8]

# Preprocessing steps
corpus <- tm_map(corpus, content_transformer(function(x) gsub("(f|ht)tps?://\\S+|www\\.\\S+", "", x))) # Remove URLs
corpus <- tm_map(corpus, content_transformer(function(x) gsub("([a-z])([A-Z])", "\\1 \\2", x))) # Remove joined words
corpus <- tm_map(corpus, content_transformer(remove_special_chars)) # Remove special characters including punctuation
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation) # Remove punctuation
corpus <- tm_map(corpus, removeNumbers) # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("english")) # Remove stop words
corpus <- tm_map(corpus, content_transformer(replace_accented_chars)) # Replace accented characters
corpus <- tm_map(corpus, stripWhitespace) # Strip whitespace
corpus <- tm_map(corpus, lemmatize_strings) # lemmatization
corpus$content[8]

# Extract preprocessed text from the corpus
cleaned_text <- sapply(corpus, as.character)

# Combine cleaned text with the Rating column
cleaned_data <- data.frame(Rating = sample_data$Rating, Review = cleaned_text)

# Write the combined data to a CSV file
write.csv(cleaned_data, file = "Data/cleaned_data.csv", row.names = FALSE)


## DATA EXPLORATION ----

# Read the cleaned data
data <- read.csv("Data/cleaned_data.csv")

# Create a column Sentiment based on Rating
data <- data %>%
  mutate(Sentiment = ifelse(Rating >= 4, "Positive", "Negative"))

# Create separate data sets for positive and negative reviews and reset their row indices
data_pos <- subset(data, Rating >= 4)
row.names(data_pos) <- NULL
data_neg <- subset(data, Rating < 4)
row.names(data_neg) <- NULL

# Plotting distribution of ratings by sentiment
ggplot(data, aes(x = factor(Rating), fill = Sentiment)) +
  geom_bar(position = "dodge") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2, position=position_dodge(0.9)) +
  labs(x = "Rating", y = "Count", title = "Distribution of Ratings by Sentiment") +
  scale_fill_manual(values = c("Positive" = "skyblue", "Negative" = "lightgreen")) +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

sentiment_counts <- table(data$Sentiment)
sentiment_df <- as.data.frame(sentiment_counts)
names(sentiment_df) <- c("Sentiment", "Count")
sentiment_df <- sentiment_df %>%
  mutate(Percentage = Count / sum(Count) * 100)
sentiment_df$Label <- paste0(sentiment_df$Count, " (", round(sentiment_df$Percentage, 1), "%)")

ggplot(sentiment_df, aes(x = "", y = Count, fill = Sentiment)) +
  geom_bar(stat = "identity", width = 1) +
  geom_text(aes(label = Label), position = position_stack(vjust = 0.5)) +
  coord_polar("y", start = 0) +
  labs(title = "Distribution of Sentiment") +
  theme(axis.title.y = element_blank(), axis.title.x = element_blank(), legend.position = "right", plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = c("Positive" = "skyblue", "Negative" = "lightgreen"))

# Create separate corpus for positive and negative reviews based on ratings
positive_corpus <- Corpus(VectorSource(data_pos$Review))
negative_corpus <- Corpus(VectorSource(data_neg$Review))

# Create document-term matrices DTM TF
positive_dtm <- DocumentTermMatrix(positive_corpus)
positive_dtm
negative_dtm <- DocumentTermMatrix(negative_corpus)
negative_dtm

# Find most commonly used terms
findFreqTerms(positive_dtm,200)
findFreqTerms(negative_dtm,200)

# Dropping terms which occur less than 1 percent in the document
positive_dtm_new <- removeSparseTerms(positive_dtm, 0.99)
positive_dtm_new
negative_dtm_new <- removeSparseTerms(negative_dtm, 0.99)
negative_dtm_new

# Find most commonly used terms again
findFreqTerms(positive_dtm_new,200)
findFreqTerms(negative_dtm_new,200)

# Find most associated term with a term of interest such as "travel"
pos_association <- findAssocs(positive_dtm_new, "travel", corlimit = 0.1)
pos_association$travel[1:10]
neg_association <- findAssocs(negative_dtm_new, "travel", corlimit = 0.1)
neg_association$travel[1:10]

# Plot the top 5 frequent words in the Positive DTM
matrix1 <- as.matrix(positive_dtm_new)
frequencies1 <- colSums(matrix1)
words1 <- sort(frequencies1, decreasing = TRUE)
top_words1 <- head(words1, 5)
barplot(top_words1, main = "Top 5 Frequent Words in the Positive DTM", 
        xlab = "Words", ylab = "Frequency", col = "skyblue")


# Plot the top 5 frequent words in the Negative DTM
matrix2 <- as.matrix(negative_dtm_new)
frequencies2 <- colSums(matrix2)
words2 <- sort(frequencies2, decreasing = TRUE)
top_words2 <- head(words2, 5)
barplot(top_words2, main = "Top 5 Frequent Words in the Negative DTM", 
        xlab = "Words", ylab = "Frequency", col = "lightgreen")

# Creating wordcloud for DTM TF
dev.off()
par(mar = c(1, 1, 1, 1))

# Creating wordcloud for positive corpus
pos_freq = data.frame(sort(colSums(as.matrix(positive_dtm_new)), decreasing=TRUE))
wordcloud(rownames(pos_freq), pos_freq[,1], max.words=300, 
          random.order=FALSE, colors=brewer.pal(8, "Dark2"))

# Creating wordcloud for negative corpus
neg_freq = data.frame(sort(colSums(as.matrix(negative_dtm_new)), decreasing=TRUE))
wordcloud(rownames(neg_freq), neg_freq[,1], max.words=300, 
          random.order=FALSE, colors=brewer.pal(8, "Dark2"))

# Create DTM TF-IDF & Wordcloud for Positive corpus
positive_dtmidf <- DocumentTermMatrix(positive_corpus, 
                                      control = list(weighting =  weightTfIdf, wordLengths=c(1,Inf)))
pos_freq_idf = data.frame(sort(colSums(as.matrix(positive_dtmidf)), decreasing=TRUE))
wordcloud(rownames(pos_freq_idf), pos_freq_idf[,1], max.words=200, 
          random.order=FALSE, colors=brewer.pal(8, "Dark2"))

# Create DTM TF-IDF & Wordcloud for Negative corpus
negative_dtmidf <- DocumentTermMatrix(negative_corpus, 
                                      control = list(weighting =  weightTfIdf, wordLengths=c(1,Inf)))
neg_freq_idf = data.frame(sort(colSums(as.matrix(negative_dtmidf)), decreasing=TRUE))
wordcloud(rownames(neg_freq_idf), neg_freq_idf[,1], max.words=200, 
          random.order=FALSE, colors=brewer.pal(8, "Dark2"))


## SENTIMENT ANALYSIS ----

# Perform sentiment analysis on the whole dataset
syuzhet_vector <- get_sentiment(data$Review, method="syuzhet")
head(syuzhet_vector)
summary(syuzhet_vector)

bing_vector <- get_sentiment(data$Review, method="bing")
head(bing_vector)
summary(bing_vector)

afinn_vector <- get_sentiment(data$Review, method="afinn")
head(afinn_vector)
summary(afinn_vector)

sent_vector <- rbind(
  sign(syuzhet_vector),
  sign(bing_vector),
  sign(afinn_vector)
)

rownames(sent_vector) <- c("Syuzhet","Bing","AFINN")
head(sent_vector, n = c(3, 10))

# Get emotions and sentiments from the reviews
sentiment <- get_nrc_sentiment(data$Review)
head(sentiment)

# Transpose the sentiment data frame
t_sentiment <- data.frame(t(sentiment))

# Calculate the total scores for each sentiment
t_sentiment_new <- data.frame(rowSums(t_sentiment[2:ncol(t_sentiment)]))

# Transformation and cleaning the new sentiment dataframe
names(t_sentiment_new)[1] <- "count"
t_sentiment_new <- cbind("Sentiment" = rownames(t_sentiment_new), t_sentiment_new)
rownames(t_sentiment_new) <- NULL
t_sentiment_new <- transform(t_sentiment_new, percentage = (count / sum(count)) * 100)

par(mar = c(4, 4, 4, 4))

# Plot Distribution of each Sentiment in Reviews
ggplot(t_sentiment_new, aes(x = Sentiment, y = count, fill = Sentiment)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            position = position_stack(vjust = 0), color = "black", 
            vjust = -0.5, size = 3) + 
  ylab("Count") +
  ggtitle("Distribution of Sentiment in Reviews") +
  theme(legend.position = "none", 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank())

## TOPIC MODELLING ~ POSITIVE REVIEWS ----

# Find number of Topics
pos_dtm <- as.matrix(positive_dtm)
frequency <- colSums(pos_dtm)
frequency <- sort(frequency, decreasing = TRUE)
doc_length <- rowSums(pos_dtm)

num_topics_pos_dtm <- FindTopicsNumber(
  pos_dtm,
  topics = seq(from = 2, to = 10, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 467),
  mc.cores = 2L,
  verbose = TRUE
)

# Visually calculating the optimal number of topics
FindTopicsNumber_plot(num_topics_pos_dtm)

# Mathematically calculating the optimal number of topics
num_topics_pos <- num_topics_pos_dtm[,1:5]
num_topics_pos[,2:5] <- scale(num_topics_pos[,2:5])
num_topics_pos$Score <- num_topics_pos$Griffiths2004 - num_topics_pos$CaoJuan2009 - num_topics_pos$Arun2010 + num_topics_pos$Deveaud2014
sorted_topic <- num_topics_pos[order(-num_topics_pos$Score), ]
num_topics <- sorted_topic$topics[1]
num_topics

# Fit topic model using LDA
positive_lda <- LDA(positive_dtm, k = num_topics, method="Gibbs", 
                    control=list(iter=1000,seed=467))

# Probability distribution over terms for a topic
phi <- posterior(positive_lda)$terms %>% as.matrix 

# Probability distribution over topics for a document
theta <- posterior(positive_lda)$topics %>% as.matrix 

# Top 10 terms associated with each topic
positive_lda_terms <- as.matrix(terms(positive_lda, 10))
positive_lda_terms

# Assigning labels to topics based on top terms
topic_labels <- c("Hospitality", 
                  "Facilities & Services", 
                  "Reservation & Check-in", 
                  "Room Facilities", 
                  "Location & Accessibility",
                  "Overall Experience",
                  "Staff Friendliness")
colnames(positive_lda_terms) <- topic_labels

# Find most probable topic for each review
positive_lda_topics <- data.frame(topics(positive_lda))
positive_lda_topics$index <- as.numeric(row.names(positive_lda_topics))
data_pos$index <- as.numeric(row.names(data_pos))
datawithtopic <- merge(data_pos, positive_lda_topics, by='index',all.x=TRUE)
datawithtopic <- datawithtopic[order(datawithtopic$index), ]
datawithtopic[0:10,]

# Association of reviews with each topic
topicProbabilities <- as.data.frame(positive_lda@gamma)
colnames(topicProbabilities) <- topic_labels
topicProbabilities[0:10,]

# Interactive Visualisation with LDAVis
# Vocab list in DTM
vocab <- colnames(phi) 

# Create the JSON object to feed the visualization in LDAvis:
json_lda <- createJSON(phi = phi, theta = theta, 
                       vocab = vocab, doc.length = doc_length, 
                       term.frequency = frequency)

# Visualize the LDA model using LDAvis
serVis(json_lda, out.dir = 'vis', open.browser = TRUE)

# Search for Top 3 factors
average_topic_proportions <- colMeans(theta)
average_topic_df <- data.frame(average_topic_proportions)
average_topic_df$Topic <- topic_labels
sorted_topic_df <- average_topic_df[order(-average_topic_df$average_topic_proportions), ]
sorted_topic_df$top3 <- ifelse(row_number(desc(sorted_topic_df$average_topic_proportions)) <= 3, TRUE, FALSE)

# Plot Top 3 factors
ggplot(sorted_topic_df, aes(x = reorder(Topic, -average_topic_proportions), y = average_topic_proportions, fill = top3)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(average_topic_proportions * 100, 3), "%")), 
            color = "black", size = 3) + 
  scale_fill_manual(values = c("FALSE" = "lightgrey", "TRUE" = "skyblue")) + 
  coord_flip(ylim = c(0.138, 0.147)) +
  labs(x = "Topics", y = "Proportion") +
  ggtitle("Top 3 Topics contributing to customer satisfaction") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) + 
  guides(fill = FALSE)


## TOPIC MODELLING ~ NEGATIVE REVIEWS ----

# Find number of Topics
neg_dtm <- as.matrix(negative_dtm)
frequency2 <- colSums(neg_dtm)
frequency2 <- sort(frequency2, decreasing = TRUE)
doc_length2 <- rowSums(neg_dtm)

num_topics_neg_dtm <- FindTopicsNumber(
  neg_dtm,
  topics = seq(from = 2, to = 10, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 467),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(num_topics_neg_dtm)

num_topics_neg <- num_topics_neg_dtm[,1:5]
num_topics_neg[,2:5] <- scale(num_topics_neg[,2:5])
num_topics_neg$Score <- num_topics_neg$Griffiths2004 - num_topics_neg$CaoJuan2009 - num_topics_neg$Arun2010 + num_topics_neg$Deveaud2014
sorted_topic2 <- num_topics_neg[order(-num_topics_neg$Score), ]
num_topics2 <- sorted_topic2$topics[1]
num_topics2

# Fit topic models
negative_lda <- LDA(negative_dtm, k = num_topics2, method="Gibbs", 
                    control=list(iter=1000,seed=467))

# Probability distribution over terms for a topic
phi2 <- posterior(negative_lda)$terms %>% as.matrix 

# Probability distribution over topics for a document
theta2 <- posterior(negative_lda)$topics %>% as.matrix 

# Top 10 terms associated with each topic
negative_lda_terms <- as.matrix(terms(negative_lda, 10))
negative_lda_terms

# Assigning labels to topics based on top terms
topic_labels2 <- c("Guest Communication", 
                   "Reservation & Check-in", 
                   "Hotel Staff & Service", 
                   "Room Facilities", 
                   "Room Comfort", 
                   "Overall Experience", 
                   "Location & Accessibility")
colnames(negative_lda_terms) <- topic_labels2

# Find most probable topic for each review
negative_lda_topics <- data.frame(topics(negative_lda))
negative_lda_topics$index <- as.numeric(row.names(negative_lda_topics))
data_neg$index <- as.numeric(row.names(data_neg))
datawithtopic2 <- merge(data_neg, negative_lda_topics, by='index',all.x=TRUE)
datawithtopic2 <- datawithtopic2[order(datawithtopic2$index), ]
datawithtopic2[0:10,]

# Association of reviews with each topic
topicProbabilities2 <- as.data.frame(negative_lda@gamma)
colnames(topicProbabilities2) <- topic_labels2
topicProbabilities2[0:10,]

# Interactive Visualisation with LDAVis
# Vocab list in DTM
vocab2 <- colnames(phi2)

# Create the JSON object to feed the visualization in LDAvis:
json_lda2 <- createJSON(phi = phi2, theta = theta2, 
                        vocab = vocab2, doc.length = doc_length2, 
                        term.frequency = frequency2)

# Visualize the LDA model using LDAvis
serVis(json_lda2, out.dir = 'vis', open.browser = TRUE)

# Search for Top 3 factors
average_topic_proportions2 <- colMeans(theta2)
average_topic_df2 <- data.frame(average_topic_proportions2)
average_topic_df2$Topic <- topic_labels2
sorted_topic_df2 <- average_topic_df2[order(-average_topic_df2$average_topic_proportions), ]
sorted_topic_df2$top3 <- ifelse(row_number(desc(sorted_topic_df2$average_topic_proportions)) <= 3, TRUE, FALSE)


# Plot Top 3 factors
ggplot(sorted_topic_df2, aes(x = reorder(Topic, -average_topic_proportions2), y = average_topic_proportions2, fill = top3)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(average_topic_proportions2 * 100, 3), "%")), 
            color = "black", size = 3) + 
  scale_fill_manual(values = c("FALSE" = "lightgrey", "TRUE" = "red")) + 
  coord_flip(ylim = c(0.138, 0.150)) +
  labs(x = "Topics", y = "Proportion") +
  ggtitle("Top 3 Topics contributing to customer dissatisfaction") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) + 
  guides(fill = FALSE)
