library(readxl)     
library(class)      
library(MASS)       
library(e1071)      
library(nnet)       
library(rpart)      
library(caret)      
library(pROC)       
library(dplyr)      

data <- read_excel("C:/Users/dubey/Desktop/Semester 6/Data Analytics and ML/Final/default of credit card clients.xls", skip = 1)
names(data)[names(data) == "default payment next month"] <- "Default_Payment"

# convert the categorical stuff into factors
data <- data %>%
  mutate(across(c(SEX, EDUCATION, MARRIAGE, PAY_0:PAY_6), as.factor))

#predictors
numeric_cols <- c("LIMIT_BAL", "AGE", 
                  paste0("BILL_AMT", 1:6), 
                  paste0("PAY_AMT", 1:6))
categorical_cols <- c("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")

# split into training and test sets
set.seed(123)
trainIndex <- createDataPartition(data$Default_Payment, p = 0.7, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# scaling numeric factors
preProc <- preProcess(train[, numeric_cols], method = c("center", "scale"))
train_scaled <- predict(preProc, train[, numeric_cols])
test_scaled <- predict(preProc, test[, numeric_cols])

# glue it all back together
train_final <- cbind(train_scaled, train[, c(categorical_cols, "Default_Payment")])
test_final <- cbind(test_scaled, test[, c(categorical_cols, "Default_Payment")])

# smoothing function 
smoothing <- function(pred_prob, actual, n = 50) {
  ordered_indices <- order(pred_prob)
  smoothed <- numeric(length(pred_prob))
  for (i in seq_along(pred_prob)) {
    lower <- max(1, i - n)
    upper <- min(length(pred_prob), i + n)
    smoothed[i] <- mean(actual[ordered_indices[lower:upper]])
  }
  data.frame(pred = pred_prob[ordered_indices], smoothed = smoothed)
}

# train 
models <- list()

# k-nearest neighbors
knn_pred <- knn(train = train_final[, numeric_cols],
                test = test_final[, numeric_cols],
                cl = train_final$Default_Payment,
                k = 5)
models$KNN <- as.numeric(knn_pred == 1)

# logistic regression
logit_model <- glm(Default_Payment ~ ., data = train_final, family = binomial)
models$Logistic <- predict(logit_model, test_final, type = "response")

# linear discriminant analysis
lda_model <- lda(Default_Payment ~ ., data = train_final)
models$LDA <- predict(lda_model, test_final)$posterior[,2]

# naive bayes
nb_model <- naiveBayes(Default_Payment ~ ., data = train_final)
models$NB <- predict(nb_model, test_final, type = "raw")[,2]

# artificial neural network
nnet_tuned <- train(Default_Payment ~ .,
                    data = train_final,
                    method = "nnet",
                    tuneGrid = expand.grid(size = 5, decay = 0.1),
                    trControl = trainControl(method = "cv", number = 5),
                    trace = FALSE)
models$ANN <- predict(nnet_tuned, test_final, type = "raw")

# decision tree
tree_model <- rpart(Default_Payment ~ ., data = train_final, method = "class")
models$Tree <- predict(tree_model, test_final, type = "prob")[,2]

# evaluation 
results <- data.frame(Model = character(),
                      AUC = numeric(),
                      R2 = numeric(),
                      Intercept = numeric(),
                      Slope = numeric(),
                      AreaRatio = numeric())

# area ratio function
calculate_area_ratio <- function(pred, actual) {
  actual <- factor(actual, levels = c(0, 1))
  

  lift_obj <- lift(actual ~ pred, data = data.frame(pred = pred, actual = actual))

  df <- lift_obj$data
  x <- df$cum.pop
  y_model <- df$CumEvent
  y_baseline <- df$cum.rn
  
  area_model <- sum(diff(x) * (head(y_model, -1) + tail(y_model, -1)) / 2)
  area_baseline <- sum(diff(x) * (head(y_baseline, -1) + tail(y_baseline, -1)) / 2)
  
  total_pop <- length(actual)
  total_pos <- sum(actual == 1)
  perfect_y <- c(rep(1, total_pos), rep(0, total_pop - total_pos))
  perfect_y_cum <- cumsum(perfect_y) / total_pos
  perfect_x <- seq(0, 1, length.out = total_pop)
  
  area_perfect <- sum(diff(perfect_x) * (head(perfect_y_cum, -1) + tail(perfect_y_cum, -1)) / 2)
  
  if (area_perfect == area_baseline) return(NA)
  (area_model - area_baseline) / (area_perfect - area_baseline)
}

# loop through each model and collect stats
for (model_name in names(models)) {

  roc_obj <- roc(test_final$Default_Payment, models[[model_name]])
  auc_score <- auc(roc_obj)

  ssm_data <- smoothing(models[[model_name]], test_final$Default_Payment)
  fit <- lm(smoothed ~ pred, data = ssm_data)
  
  # area ratio
  area_ratio <- calculate_area_ratio(models[[model_name]], test_final$Default_Payment)
  
  results <- rbind(results, data.frame(
    Model = model_name,
    AUC = round(auc_score, 3),
    R2 = round(summary(fit)$r.squared, 3),
    Intercept = round(coef(fit)[1], 4),
    Slope = round(coef(fit)[2], 4),
    AreaRatio = round(area_ratio, 3)
  ))
}

print(results)
