if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Start
# Create train and test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Matching userId and movieId in both train and test sets
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Adding back rows into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# Data exploration
# Str
str(edx)

# Summary statistics
edx %>% select(-genres) %>% summary()

# Unique users and rated movies
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# Top movies
edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(20, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill = "deepskyblue2", stat = "identity") +
  xlab("Count") +
  ylab(NULL) +
  theme_bw()

# Movie ratings frequency chart
edx %>%
  ggplot(aes(rating, y = ..prop..)) +
  geom_bar(color = "black", fill = "deepskyblue2") +
  labs(x = "Ratings", y = "Relative Frequency") +
  scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) +
  theme_bw()

# Distribution of users
edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", fill = "deepskyblue2", bins = 40) +
  xlab("Ratings") +
  ylab("Users") +
  scale_x_log10() +
  theme_bw()

# RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Model 1
mu_hat <- mean(train_set$rating)
mu_hat

# Model 1 RMSE
naive_rmse <- RMSE(test_set$rating, mu_hat)
results <- tibble(Method = "Model 1: Simply the mean", RMSE = naive_rmse)
results %>% knitr::kable()

# Movie bias
bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

# Movie bias distribution
bi %>% ggplot(aes(b_i)) +
  geom_histogram(color = "black", fill = "deepskyblue2", bins = 10) +
  xlab("Movie Bias") +
  ylab("Count") +
  theme_bw()

# Model 2 RMSE
predicted_ratings <- mu_hat + test_set %>%
  left_join(bi, by = "movieId") %>%
  pull(b_i)
m_bias_rmse <- RMSE(predicted_ratings, test_set$rating)
results <- bind_rows(results, tibble(Method = "Model 2: Mean + movie bias", RMSE = m_bias_rmse))
results %>% knitr::kable()

# User bias
bu <- train_set %>%
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Model 3 RMSE
predicted_ratings <- test_set %>%
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)
u_bias_rmse <- RMSE(predicted_ratings, test_set$rating)
results <- bind_rows(results, tibble(Method = "Model 3: Mean + movie bias + user effect", RMSE = u_bias_rmse))
results %>% knitr::kable()

# Model 4 Regularization and RMSE
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(x){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+x))
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+x))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Lambdas plot and best parameter
qplot(lambdas, rmses, color = I("blue"))
lambda <- lambdas[which.min(rmses)]
lambda

# Model 4 RMSE
results <- bind_rows(results, tibble(Method = "Model 4: Regularized movie and user effects", RMSE = min(rmses)))
results %>% knitr::kable()

# Model 5 Matrix Factorization using recosystem
library(recosystem)
set.seed(1, sample.kind="Rounding")
train_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()

# Recosystem tuning
para_reco <- r$tune(train_reco, opts = list(dim = c(20, 30),
                                            costp_l2 = c(0.01, 0.1),
                                            costq_l2 = c(0.01, 0.1),
                                            lrate = c(0.01, 0.1),
                                            nthread = 4,
                                            niter = 10))
# Recosystem training
r$train(train_reco, opts = c(para_reco$min, nthread = 4, niter = 30))

# Recosystem prediction
results_reco <- r$predict(test_reco, out_memory())

# Model 5 RMSE
factorization_rmse <- RMSE(results_reco, test_set$rating)
results <- bind_rows(results, tibble(Method = "Model 5: Matrix factorization using recosystem", RMSE = factorization_rmse))
results %>% knitr::kable()

# Final validation
set.seed(1, sample.kind="Rounding")
edx_reco <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
validation_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()

para_reco <- r$tune(edx_reco, opts = list(dim = c(20, 30),
                                            costp_l2 = c(0.01, 0.1),
                                            costq_l2 = c(0.01, 0.1),
                                            lrate = c(0.01, 0.1),
                                            nthread = 4,
                                            niter = 10))

r$train(edx_reco, opts = c(para_reco$min, nthread = 4, niter = 30))

# Final prediction
final_reco <- r$predict(validation_reco, out_memory())

# Final RMSE
final_rmse <- RMSE(final_reco, validation$rating)
results <- bind_rows(results, tibble(Method = "Final validation: Matrix factorization using recosystem", RMSE = final_rmse))
results %>% knitr::kable()
