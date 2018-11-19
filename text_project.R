# Topic Funnel 2.0 Analyzer #

#### Required Packages ####
library(dplyr)
library(sentimentr)
library(lexicon)
library(tidytext)
library(tokenizers)
library(lexRankr)
library(ggplot2)
library(tm)
library(topicmodels)
library(readr)
library(lda)
library(LDAvis)

#### Files ####
dirName <- "C:/Users/George/Documents/Text Analytics/Cisco/"

fileName <- list.files(dirName, pattern = glob2rx("*.txt"), full.names = F)

cisco_df <- data.frame(text = rep(NA, length(fileName)))

#########################################################
# Run this if I sent you the CSV of the top sentences #
############ Then skip lines 41 - 74 ####################
read_sentences <- read_csv("top_sentence_cisco.csv")
cisco_df$text <- read_sentences$text

#######################################################

#### read files in #####
data <- do.call("rbind", 
                  lapply(fileName, 
                         function(x) 
                           readLines(paste(dirName, x, sep='') 
                           )))
# Put text into a vector #
data_vector <- data[,2]


# USE CSV. THIS WILL TAKE 30 - 40 minutes #
#### Prepare data for LexRank####
for(file in 1:length(data_vector)){
  
  sent <- gsub("=", "", data_vector[file])
  
  sent <- gsub("-", "", sent)
  
  sent <- gsub('\\[+[0-9]{1,3}\\]+', "", sent)
  
  test_sent <- tokenize_sentences(sent)
  
  test_sent <- unlist(test_sent)
  
  #### Lex Rank for locate most important sentences ####
  top_5 = lexRank(test_sent,
                  docId = rep(1, length(test_sent)),
                  n = round((.05 * length(test_sent))),
                  continuous = TRUE, 
                  sentencesAsDocs = F
  )

  #reorder the top  sentences to be in order of appearance in article
  order_of_appearance = order(as.integer(gsub("_","",top_5$sentenceId)))
  #extract sentences in order of appearance
  ordered_top_5 = top_5[order_of_appearance, "sentence"]
  
  top_5_collpased <- paste0(ordered_top_5, collapse = "")
  
  cisco_df$text[file] <- top_5_collpased
}


#### Sentiment Analysis ####

cisco_df$sentiment <- NA
cisco_df$fileName <- fileName

for(file in 1:length(cisco_df$text)){
  
  topic_txt <- iconv(cisco_df$text[file], from = "latin1", to = "ASCII", sub = "")
  topic_txt <- trimws(stripWhitespace(topic_txt), which = "both")
  
  temp <- which( nchar(topic_txt) == 0 )
  if ( length(temp) >0 ) topic_txt <- topic_txt[ -temp ]
  
  topic_txt <- paste(topic_txt, collapse = " ") 
  topic_txt <- stripWhitespace(topic_txt) 
  topic_txt <- trimws(topic_txt, which = "both") 
  topic_txt <- gsub("\\\"", "", topic_txt, fixed = FALSE) 
  topic_txt <- gsub("(\\ \\.\\ )+", " ", topic_txt)
  topic_txt <- gsub("[[:punct:]]", " ", topic_txt)
  topic_txt <- gsub("[[:digit:]]", " ", topic_txt)
  
  doc_sent <- sentiment(topic_txt, polarity_dt = hash_sentiment_loughran_mcdonald, 
                        valence_shifters_dt = lexicon::hash_valence_shifters)
  
  cisco_df$sentiment[file] <- doc_sent[,4]
  
}

#### Topic Analysis on the reduced document   ####
cisco_topic_df <- data.frame(text = rep(NA, length(fileName)))

#### read through important sentences and clean again ####
for(file in 1:length(cisco_df$text)) {
  
  doctext <- iconv(cisco_df$text[file], from = "latin1", to = "ASCII", sub = "")
  doctext <- trimws(stripWhitespace(doctext), which = "both")
  
  temp <- NULL
  temp <- which( nchar(doctext) == 0 ) 
  if (length(temp) >0) doctext <- doctext[-temp]
  
  doctext <- gsub("[[:punct:]]", " ", doctext)
  doctext <- gsub("[[:digit:]]", " ", doctext)
  
  cisco_topic_df$text[file] <- doctext
}

#### Set up DTM ####
cisco_topic_df$fileName <- fileName

cisco_words <- cisco_topic_df %>%
  unnest_tokens(word, text)

word_count <- cisco_words %>%
  anti_join(stop_words, by = "word") %>%
  count(fileName, word, sort = TRUE)

word_count <- word_count %>%
  bind_tf_idf(word, fileName, n)

word_count <- word_count %>%
  filter(tf_idf > quantile(tf_idf, probs = 0.25))

cisco_DTM <- word_count %>%
  cast_dtm(fileName, word, n)

#### Preform LDA ####
cisco_lda <- LDA(cisco_DTM, k = 20, method = "Gibbs", control = list(iter = 2000, seed = 69))

cisco_topics <- tidy(cisco_lda, matrix = "beta")

top_terms <- cisco_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

transcipts_gamma <- tidy(cisco_lda, matrix = "gamma")

transcript_classification <- transcipts_gamma %>%
  group_by(document) %>%
  top_n(1, gamma)

transcript_classification <- merge(transcript_classification, cisco_topic_df, by.x = "document", by.y = "fileName")

illustrative_transcript <- transcript_classification %>%
  group_by(topic) %>%
  top_n(1, gamma)



#### Visulalize the LDA ####

# Bar Chart showing count by topic #

ggplot(transcript_classification, aes(as.factor(topic))) +
  geom_bar(color = "black", fill = "#66ffff") +
  xlab("Topic Group")+
  ylab(NULL) +
  ggtitle("Number of Documents Per Topic") +
  theme(axis.text.y = element_text(size=15),axis.text.x = element_text(size=11, angle = 0))+
  theme(panel.background = element_rect(fill = "#f2f2f2")) + 
  theme(plot.background = element_rect(fill = "White")) + 
  theme(panel.border = element_rect(colour = "Black", fill=NA, size=1.5)) +
  theme(plot.title = element_text(family = "sans", color="Black", size=20, hjust = 0.5))


# Clean up data for looking a sentiment across topic #

colnames(transcript_classification)[1] <- "fileName"
plotaction <-cisco_df %>%
  left_join(transcript_classification, by = "fileName") %>%
  group_by(topic) %>%
  summarise(avg_sent= mean(as.numeric(sentiment)))

# Sent. across topic #

ggplot(plotaction, aes(as.factor(topic), avg_sent)) +
  geom_bar(stat = 'identity', fill = "#66ffff", color = "black")+
  geom_hline(yintercept = 0, size = 1) +
  xlab("Topic Group") +
  ylab("Sentiment Score")+
  ggtitle("Sentiment Across Topics") +
  theme(axis.text.y = element_text(size=15),axis.text.x = element_text(size=11, angle = 0))+
  theme(panel.background = element_rect(fill = "#f2f2f2")) + 
  theme(plot.background = element_rect(fill = "White")) + 
  theme(panel.border = element_rect(colour = "Black", fill=NA, size=1.5)) +
  theme(plot.title = element_text(family = "sans", color="Black", size=20, hjust = 0.5))





# clean text once more #

cisco_vis_df <- data.frame(clean_text = rep(NA, length(fileName)))

for(file in 1:length(cisco_vis_df$clean_text)){
  
  txt <- iconv(data_vector[file], from = "latin1", to = "ASCII", sub = "")
  txt <- trimws(stripWhitespace(txt), which = "both")
  
  temp <- which( nchar(txt) == 0 )
  if ( length(temp) >0 ) txt <- txt[ -temp ]
  
  txt <- paste(txt, collapse = " ") 
  txt <- stripWhitespace(txt) 
  txt <- trimws(txt, which = "both") 
  txt <- gsub("\\\"", "", txt, fixed = FALSE) 
  txt <- gsub("(\\ \\.\\ )+", " ", txt)
  txt <- gsub("[[:punct:]]", " ", txt)
  txt <- gsub("[[:digit:]]", " ", txt)
  
  cisco_vis_df$clean_text[file] <- txt
}


#### Create dashboard to examine LDA ####
doc_list <- strsplit(cisco_vis_df$clean_text, "[[:space:]]+")


term.table <- table(unlist(doc_list))
term.table <- sort(term.table, decreasing = TRUE)

del <- names(term.table) %in% stop_words$word | term.table < 5 | names(term.table) %in% c("I", ",", "(.*?),", "exhibits", "exh") 
term.table <- term.table[!del]
vocab <- names(term.table)

get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}


documents_cisco <- lapply(doc_list, get.terms)

D <- length(documents_cisco)  # number of documents 
W <- length(vocab)  # number of terms in the vocab
doc.length <- sapply(documents_cisco, function(x) sum(x[2, ]))  # number of tokens per document 
N <- sum(doc.length)  # total number of tokens in the data 
term.frequency <- as.integer(term.table)


K <- 20
G <- 2000
alpha <- 0.02
eta <- 0.02

cisco_fit <- lda.collapsed.gibbs.sampler(documents = documents_cisco, K = K, 
                                         vocab = vocab, 
                                         num.iterations = G, alpha = alpha, 
                                         eta = eta, initial = NULL, burnin = 0,
                                         compute.log.likelihood = TRUE)


theta <- t(apply(cisco_fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(cisco_fit$topics) + eta, 2, function(x) x/sum(x)))


cisco_plot_list <- list(phi = phi,
                        theta = theta,
                        doc.length = doc.length,
                        vocab = vocab,
                        term.frequency = term.frequency)

json <- createJSON(phi = cisco_plot_list$phi, 
                   theta = cisco_plot_list$theta, 
                   doc.length = cisco_plot_list$doc.length, 
                   vocab = cisco_plot_list$vocab, 
                   term.frequency = cisco_plot_list$term.frequency)

serVis(json, out.dir = 'C:/Users/George/Documents/Text Analytics/', open.browser = FALSE)




