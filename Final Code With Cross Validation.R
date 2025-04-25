# ------------------- Libraries -------------------
library(readxl)
library(class)
library(MASS)
library(e1071)
library(nnet)
library(rpart)
library(caret)
library(pROC)
library(dplyr)
library(randomForest)
library(xgboost)
library(pls)
library(purrr)

#VERY IMPORTANT TO CHANGE FILE PATH TO THE DATASETS PATH IN YOUR PC
data <- read_excel("C:/Users/dubey/Desktop/Semester 6/Data Analytics and ML/Final/default of credit card clients.xls", skip = 1)
names(data)[names(data) == "default payment next month"] <- "Default_Payment"

# convert the categorical variables to factors
data <- data %>%
  mutate(across(c(SEX, EDUCATION, MARRIAGE, PAY_0:PAY_6), as.factor))

#predictors
numeric_cols <- c("LIMIT_BAL", "AGE", 
                  paste0("BILL_AMT", 1:6), 
                  paste0("PAY_AMT", 1:6))
categorical_cols <- c("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")

#Convert PAY_0 to PAY_6 to numeric (ordinal). VERY IMPORTANT.
data <- data %>%
  mutate(across(c(SEX, EDUCATION, MARRIAGE), as.factor)) %>%  # nominal factors
  mutate(across(PAY_0:PAY_6, as.numeric))  # ordinal as numeric

# smoothing function for SSM
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

# area ratio function
calculate_area_ratio <- function(pred, actual) {
  df <- data.frame(pred = pred, actual = actual)
  df <- df[order(-df$pred), ]
  total_pop <- nrow(df)
  total_pos <- sum(df$actual)
  df$cum_actual <- cumsum(df$actual)
  model_curve <- df$cum_actual / total_pos
  baseline_curve <- seq(1, total_pop) / total_pop
  df_best <- df[order(-df$actual), ]
  df_best$cum_best <- cumsum(df_best$actual)
  best_curve <- df_best$cum_best / total_pos
  x <- seq(1, total_pop) / total_pop
  trapz <- function(x, y) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
  area_model <- trapz(x, model_curve)
  area_baseline <- trapz(x, baseline_curve)
  area_best <- trapz(x, best_curve)
  if ((area_best - area_baseline) == 0) return(NA)
  (area_model - area_baseline) / (area_best - area_baseline)
}

#K-fold CV
cv_results <- list()

for (k in 5:15) {
  set.seed(123)
  folds <- createFolds(data$Default_Payment, k = k)
  
  fold_metrics <- data.frame(Model = character(),
                             Fold = integer(),
                             AUC = numeric(),
                             R2 = numeric(),
                             Intercept = numeric(),
                             Slope = numeric(),
                             AreaRatio = numeric(),
                             ErrorRate = numeric(),
                             stringsAsFactors = FALSE)
  
  for (i in seq_along(folds)) {
    test_index <- folds[[i]]
    train_data <- data[-test_index, ]
    test_data <- data[test_index, ]
    
    # scale numeric columns
    preProc <- preProcess(train_data[, numeric_cols], method = c("center", "scale"))
    train_scaled <- predict(preProc, train_data[, numeric_cols])
    test_scaled <- predict(preProc, test_data[, numeric_cols])
    
    train_final <- cbind(train_scaled, train_data[, c(categorical_cols, "Default_Payment")])
    test_final <- cbind(test_scaled, test_data[, c(categorical_cols, "Default_Payment")])
    
    train_final$Default_Payment <- as.factor(train_final$Default_Payment)
    test_final$Default_Payment <- as.factor(test_final$Default_Payment)
    
    models <- list()
    
    # logistic regression
    models$Logistic <- predict(glm(Default_Payment ~ ., data = train_final, family = binomial), test_final, type = "response")
    
    # lda
    models$LDA <- predict(lda(Default_Payment ~ ., data = train_final), test_final)$posterior[,2]
    
    # naive bayes
    models$NB <- predict(naiveBayes(Default_Payment ~ ., data = train_final), test_final, type = "raw")[,2]
    
    # ANN
    ann_model <- train(Default_Payment ~ ., data = train_final, method = "nnet",
                       tuneGrid = expand.grid(size = 5, decay = 0.1),
                       trControl = trainControl(method = "cv", number = 3),
                       trace = FALSE)
    models$ANN <- predict(ann_model, test_final, type = "prob")[,2]
    
    # decision tree
    models$Tree <- predict(rpart(Default_Payment ~ ., data = train_final, method = "class"), test_final, type = "prob")[,2]
    
    # random Forest
    models$RF <- predict(randomForest(Default_Payment ~ ., data = train_final, ntree = 100), test_final, type = "prob")[,2]
    
    # xgboost
    xgb_train <- model.matrix(Default_Payment ~ . - 1, data = train_final)
    xgb_test <- model.matrix(Default_Payment ~ . - 1, data = test_final)
    xgb_label <- as.numeric(as.character(train_final$Default_Payment))
    xgb_model <- xgboost(data = xgb_train, label = xgb_label, objective = "binary:logistic",
                         nrounds = 100, max_depth = 4, eta = 0.1, verbose = 0)
    models$XGBoost <- predict(xgb_model, newdata = xgb_test)
    
    # KNN
    knn_train_mat <- model.matrix(~ . -1, data = train_final[, c(numeric_cols, categorical_cols)])
    knn_test_mat <- model.matrix(~ . -1, data = test_final[, c(numeric_cols, categorical_cols)])
    knn_labels <- as.numeric(as.character(train_final$Default_Payment))
    knn_pred <- knn(train = knn_train_mat, test = knn_test_mat, cl = knn_labels, k = 5, prob = TRUE)
    knn_probs <- ifelse(knn_pred == "1", attr(knn_pred, "prob"), 1 - attr(knn_pred, "prob"))
    models$KNN <- knn_probs
    
    # convert actuals to numeric
    test_final$Default_Payment <- as.numeric(as.character(test_final$Default_Payment))
    
    test_final$Default_Payment <- as.numeric(as.character(test_final$Default_Payment))
    
    # Evaluate all models
    for (model_name in names(models)) {
      roc_obj <- roc(test_final$Default_Payment, models[[model_name]])
      auc_score <- auc(roc_obj)
      ssm_data <- smoothing(models[[model_name]], test_final$Default_Payment)
      fit <- lm(smoothed ~ pred, data = ssm_data)
      area_ratio <- calculate_area_ratio(models[[model_name]], test_final$Default_Payment)
      pred_labels <- ifelse(models[[model_name]] >= 0.5, 1, 0)
      error_rate <- mean(pred_labels != test_final$Default_Payment)
      
      fold_metrics <- rbind(fold_metrics, data.frame(
        Model = model_name,
        Fold = i,
        AUC = round(auc_score, 3),
        R2 = round(summary(fit)$r.squared, 3),
        Intercept = round(coef(fit)[1], 4),
        Slope = round(coef(fit)[2], 4),
        AreaRatio = round(area_ratio, 3),
        ErrorRate = round(error_rate, 3)
      ))
    }
  }
  
  avg_metrics <- fold_metrics %>%
    group_by(Model) %>%
    summarise(across(AUC:ErrorRate, ~ round(mean(.), 3))) %>%
    mutate(K = k)
  
  cv_results[[as.character(k)]] <- avg_metrics
}

final_cv_results <- bind_rows(cv_results)
print(final_cv_results)
