# "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (Angelopoulos & Bates)

# 0. LIBRARIES AND SEED

# dslabs is the library that we used only to download/load raw MNIST data as 28x28 pixel images that we will treat as a matrix
if (!require("dslabs", quietly = TRUE)) install.packages("dslabs")
library(dslabs)

set.seed(42)  # Set seed to have the same deterministic results always

# 1. Data Loading and Preprocessing

mnist = read_mnist()  # Download the MNIST dataset from the internet

X_all = rbind(mnist$train$images, mnist$test$images) # We create a big matrix in which every column represents a pixel and every row represents an image
y_all = c(mnist$train$labels, mnist$test$labels) # Take the true numbers that they should represent

total_sample_size = 8000 # We sample, using just 8k images out of the 70k available in the original MNIST dataset
indices_total = sample(1:nrow(X_all), total_sample_size) # Randomly sample the indices
X_all = X_all[indices_total, ] # Keep just the 8,000 sampled images and discard all the others
y_all = y_all[indices_total] # Keep just the true labels of these 8,000 randomly chosen images

X_all = X_all / 255 # Normalize pixel values, scaling them to [0,1]; not scaling them produces errors in the gradient descent later in section 2

num_classes = 10

# One-hot encoding is needed to compute the cross-entropy loss used in gradient descent; we need to be able to multiply the 10 columns to use cross-entropy loss instead of a distance
one_hot = function(y, num_classes_param = num_classes) { # Function definition
  one_hot_matrix = matrix(0, nrow = length(y), ncol = num_classes_param) # Create a matrix of all 0s, with a row for every image and a column for every possible number
  for (i in 1:length(y)) { # Go from the first row to the last one
    one_hot_matrix[i, y[i] + 1] = 1 # Put a 1 in the column corresponding to the true number (0,1,2,3...), remembering, Ire, that indices start from 1 in R
  }
  return(one_hot_matrix)
}
one_hot_labels_all = one_hot(y_all) # We call the function for all 8,000 figures

total_observations = nrow(X_all)
indices_random = sample(1:total_observations) # Shuffle all dataset indices to eliminate bias and order, or ordinal meaning if any
n_train = floor(0.5 * total_observations) # Count how many will go into training as an absolute number, rounding down
n_calib = floor(0.25 * total_observations) # Same for calibration

indices_train = indices_random[1:n_train] # Take the first 50% of the training indices
indices_calib = indices_random[(n_train + 1):(n_train + n_calib)] # Take the subsequent 25% of the indices for calibration
indices_test  = indices_random[(n_train + n_calib + 1):total_observations] # All the remaining indices go to the test set (slightly bigger than calibration if total size is changed, as a consequence of using floor)

# Extract the data given the indices, and split the matrices, creating the 3 split datasets
X_train = X_all[indices_train, ]; one_hot_labels_train = one_hot_labels_all[indices_train, ]; y_train = y_all[indices_train]
X_calib = X_all[indices_calib, ]; one_hot_labels_calib = one_hot_labels_all[indices_calib, ]; y_calib = y_all[indices_calib]
X_test  = X_all[indices_test, ];  one_hot_labels_test  = one_hot_labels_all[indices_test, ];  y_test  = y_all[indices_test]

# Return the dimensions of the datasets
cat("Dimensions -> train:", nrow(X_train),
    " calib:", nrow(X_calib),
    " test:", nrow(X_test), "\n")




# 2. NN taken mainly from the internet and readapted, with fixed hyperparameters !!!NOT CRAFTED BY US!!!, just adapted with modified variables
n_x <- ncol(X_train)      
n_h <- 64                 
n_y <- num_classes        
weight_matrix_pixel_neuron  <- matrix(rnorm(n_x * n_h, sd = 0.01), nrow = n_x, ncol = n_h)  
bias_vector_neuron          <- rep(0, n_h)                                                  
weight_matrix_neuron_output <- matrix(rnorm(n_h * n_y, sd = 0.01), nrow = n_h, ncol = n_y)  
bias_vector_output          <- rep(0, n_y)                                                  
relu <- function(Z) {
  Z[Z < 0] <- 0
  return(Z)
}
softmax <- function(Z) {
  Z <- Z - apply(Z, 1, max)
  exp_Z <- exp(Z)
  return(exp_Z / rowSums(exp_Z))
}
forward <- function(X, W1, b1, W2, b2) {
  Z1 <- X %*% W1 + matrix(b1, nrow = nrow(X), ncol = length(b1), byrow = TRUE)
  A1 <- relu(Z1)
  Z2 <- A1 %*% W2 + matrix(b2, nrow = nrow(A1), ncol = length(b2), byrow = TRUE)
  probabilities_output <- softmax(Z2)
  list(
    Z1 = Z1,
    A1 = A1,
    Z2 = Z2,
    probabilities_output = probabilities_output
  )
}
learning_rate <- 0.1
n_epochs <- 400
m <- nrow(X_train)
loss_history <- numeric(n_epochs)

for (epoch in 1:n_epochs) {
  cache <- forward(X_train, weight_matrix_pixel_neuron, bias_vector_neuron, weight_matrix_neuron_output, bias_vector_output)
  probabilities_output <- cache$probabilities_output
  loss <- -mean(rowSums(one_hot_labels_train * log(probabilities_output + 1e-9)))
  loss_history[epoch] <- loss
  dZ2 <- (probabilities_output - one_hot_labels_train) / m
  dW2 <- t(cache$A1) %*% dZ2
  db2 <- colSums(dZ2)
  dA1 <- dZ2 %*% t(weight_matrix_neuron_output)
  dZ1 <- dA1
  dZ1[cache$Z1 <= 0] <- 0
  dW1 <- t(X_train) %*% dZ1
  db1 <- colSums(dZ1)
  weight_matrix_neuron_output <- weight_matrix_neuron_output - learning_rate * dW2
  bias_vector_output          <- bias_vector_output - learning_rate * db2
  weight_matrix_pixel_neuron  <- weight_matrix_pixel_neuron - learning_rate * dW1
  bias_vector_neuron          <- bias_vector_neuron - learning_rate * db1
  if (epoch %% 50 == 0) {
    cat("Epoch", epoch, "- loss:", round(loss, 4), "\n")
  }
}
yhat_train <- apply(forward(X_train, weight_matrix_pixel_neuron, bias_vector_neuron, weight_matrix_neuron_output, bias_vector_output)$probabilities_output, 1, which.max) - 1
acc_train  <- mean(yhat_train == y_train)
cat("Accuracy on training set:", round(acc_train * 100, 2), "%\n")

#######################################################################################################################################



# Our project focus
# 3. BASE CONFORMAL PREDICTION (Section 1 of the paper)

alpha = 0.10  # We define 90% as our threshold, so for the paper formula we have to define alpha = 1 - 0.9 (threshold)

# Calculate the conformal score of the base classifier
calculate_base_scores = function(probabilities, y) { 
  n = nrow(probabilities) 
  scores = numeric(n) # Empty vector to save the scores, one for each image
  for (i in 1:n) { # For all the images
    scores[i] = 1 - probabilities[i, y[i] + 1]   # Low score = secure and correct
  }
  return(scores)
}

# Calculate the probabilities predicted by the NN for the 25% calibration dataset
probabilities_calib = forward(X_calib, weight_matrix_pixel_neuron, bias_vector_neuron, weight_matrix_neuron_output, bias_vector_output)$probabilities_output   
scores_calib = calculate_base_scores(probabilities_calib, y_calib) # Calculate the conformal scores for all images of the calibration dataset (using the base classifier)

# Function to compute the qhat threshold: conformal quantile that gives us a statistical guarantee that the true class is in the set 90% of the time
conformal_quantile = function(scores, alpha) {
  n = length(scores) # Count how many calibration scores are available
  sorted_scores = sort(scores)
  position = ceiling((n + 1) * (1 - alpha))  # Compute the position, once sorted, to use as the threshold        
  position = min(position, n)  # If the calculated position surpasses the number of available scores, limit it to the last available score to avoid NA 
  # (which would otherwise propagate through build_prediction_set, causing everything to be FALSE or NA)                      
  return(sorted_scores[position])
}

qhat_base = conformal_quantile(scores_calib, alpha) # qhat threshold using the calibration scores already computed before and the alpha that we chose
cat("Conformal threshold (qhat) - base method:", round(qhat_base, 4), "\n") # Mettilo nella presentazione: ci aiuta a capire quanto margine si prende la conformal prediction tramite la soglia calcolata

# Given the probabilities and qhat, build the prediction set for each image. It includes only the numbers whose predicted probability is greater than 1 - qhat
# The result is a matrix of TRUE/FALSE values (TRUE means that this number is included in the prediction set for this image)
build_prediction_set = function(probabilities, qhat) { # It calculates it for each image
  return(probabilities >= (1 - qhat)) 
}

probabilities_test = forward(X_test, weight_matrix_pixel_neuron, bias_vector_neuron, weight_matrix_neuron_output, bias_vector_output)$probabilities_output
prediction_set_base = build_prediction_set(probabilities_test, qhat_base) # Build the prediction set for each image of the test set, using the qhat_base threshold (previously calculated on the calibration dataset)

# Check if the true number is present in the prediction set of that image, and then compute the mean over all test images
# to get the percentage of times in which the true number was present in the set
coverage_base = mean(sapply(1:nrow(X_test), function(i) prediction_set_base[i, y_test[i] + 1]))
cat("Empirical coverage (base method):", round(coverage_base * 100, 2), # Empirical coverage of the test set
    "% (target:", (1 - alpha) * 100, "%)\n") # Theoretical goal for comparison


# 4. ADAPTIVE PREDICTION SETS - APS (Section 2.1 of the paper)

# Compute the conformal score of the APS method for a group of images with known true labels
calculate_aps_scores = function(probabilities, y) {
  n = nrow(probabilities)
  scores = numeric(n) # Create an empty vector to use later to store the APS scores, one for each image
  
  # For each image, sort probabilities and compute the cumulative sum until the true class
  for (i in 1:n) {
    row_probabilities = probabilities[i, ] # Extract 10 probabilities, one for each number, predicted for this specific image
    order_indices = order(row_probabilities, decreasing = TRUE) # Find the order of the numbers from most likely to least; it returns the positions/columns in that order
    sorted_probabilities = row_probabilities[order_indices] # Then it reorders the probability values, given the previous line, from most to least likely
    cumulative_sum = cumsum(sorted_probabilities) # Compute the progressive cumulative sum until 1 (100%)                  
    true_class_rank = which(order_indices == (y[i] + 1)) # Find the position of the true number     
    scores[i] = cumulative_sum[true_class_rank]  # Measure how much cumulative probability is needed to include the correct number                   
  }
  return(scores)
}

scores_calib_aps = calculate_aps_scores(probabilities_calib, y_calib) # APS scores for each image of the calibration set
qhat_aps = conformal_quantile(scores_calib_aps, alpha) # qhat threshold for APS
cat("Conformal threshold (qhat) - APS method:", round(qhat_aps, 4), "\n")


# Having probabilities and qhat, build the prediction set for each image using the APS logic: keep adding numbers, 
# from most to least likely, until the cumulative probability passes qhat
build_aps_set = function(probabilities, qhat) {
  n = nrow(probabilities); k = ncol(probabilities) # n = number of images / k = number of possible numbers (10)
  prediction_set = matrix(FALSE, n, k) # Build an empty matrix, filled with FALSE, which will be filled with TRUE every time a number is included in the prediction set
  
  # For each image, include classes until the cumulative sum reaches qhat (same as in section 3, but for APS)
  for (i in 1:n) {
    row_probabilities = probabilities[i, ]
    order_indices = order(row_probabilities, decreasing = TRUE)
    sorted_probabilities = row_probabilities[order_indices]
    cumulative_sum = cumsum(sorted_probabilities)
    
    # Include classes until the sum is greater than qhat (plus a tiny tolerance)
    num_included_classes = sum(cumulative_sum <= qhat + 1e-9) + 1
    num_included_classes = min(num_included_classes, k)          
    
    included_classes = order_indices[1:num_included_classes]
    prediction_set[i, included_classes] = TRUE
  }
  return(prediction_set)
}

prediction_set_aps = build_aps_set(probabilities_test, qhat_aps)
coverage_aps = mean(sapply(1:nrow(X_test), function(i) prediction_set_aps[i, y_test[i] + 1]))
cat("Empirical coverage (APS method):", round(coverage_aps * 100, 2),
    "% (target:", (1 - alpha) * 100, "%)\n")



# 5. EVALUATION (Section 4 of the paper): Repeat over many splits
# Since everything calculated until now was an empirical coverage from just a single run on one random split (calibration/test),
# to ensure a stable pattern and not a lucky split, we repeat the calibration/evaluation step 50 times (the NN itself is not retrained).

n_repetitions = 50
# Empty vectors to fill
coverages_base = numeric(n_repetitions)
coverages_aps  = numeric(n_repetitions)
sizes_base = numeric(n_repetitions)
sizes_aps  = numeric(n_repetitions)

X_pool = rbind(X_calib, X_test)
y_pool = c(y_calib, y_test)
n_pool = nrow(X_pool)
probabilities_pool = forward(X_pool, weight_matrix_pixel_neuron, bias_vector_neuron, weight_matrix_neuron_output, bias_vector_output)$probabilities_output   

for (r in 1:n_repetitions) {
  indices_repetition = sample(1:n_pool) # Randomly mix all indices of the pool; every repetition will have a different split, independent of the others
  n_calib_split = floor(n_pool / 2) # In this repeated evaluation, the data is split 50/50 between calibration and test
  indices_calib_split = indices_repetition[1:n_calib_split] # Take the first half of the mixed indices as the new calibration group for this repetition
  indices_test_split = indices_repetition[(n_calib_split + 1):n_pool] # Take the second half as the new test group for this repetition
  
  # --- base method ---
  scores_calib_split = calculate_base_scores(probabilities_pool[indices_calib_split, , drop = FALSE], y_pool[indices_calib_split])
  qhat_split = conformal_quantile(scores_calib_split, alpha)
  pred_set_split = build_prediction_set(probabilities_pool[indices_test_split, , drop = FALSE], qhat_split)
  coverages_base[r] = mean(sapply(1:length(indices_test_split), function(i) pred_set_split[i, y_pool[indices_test_split[i]] + 1]))
  sizes_base[r] = mean(rowSums(pred_set_split))
  
  # --- APS method ---
  scores_calib_aps_split = calculate_aps_scores(probabilities_pool[indices_calib_split, , drop = FALSE], y_pool[indices_calib_split])
  qhat_aps_split = conformal_quantile(scores_calib_aps_split, alpha)
  pred_set_aps_split = build_aps_set(probabilities_pool[indices_test_split, , drop = FALSE], qhat_aps_split)
  coverages_aps[r] = mean(sapply(1:length(indices_test_split), function(i) pred_set_aps_split[i, y_pool[indices_test_split[i]] + 1]))
  sizes_aps[r] = mean(rowSums(pred_set_aps_split))
}

cat("\n--- Summary over", n_repetitions, "random splits ---\n")
cat("Average coverage (base):", round(mean(coverages_base) * 100, 2), "%\n")
cat("Average coverage (APS): ", round(mean(coverages_aps) * 100, 2), "%\n")
cat("Average prediction set size (base):", round(mean(sizes_base), 2), "\n")
cat("Average prediction set size (APS): ", round(mean(sizes_aps), 2), "\n")

# ---- Histogram of empirical coverage over splits ----
x_range_coverage = range(c(coverages_base, coverages_aps))

hist(coverages_base, col = rgb(1,0,0,0.5), xlim = x_range_coverage, breaks = 15,
     main = "Distribution of empirical coverage",
     xlab = "Empirical coverage", ylab = "Frequency")
hist(coverages_aps, col = rgb(0,0,1,0.5), breaks = 15, add = TRUE)
abline(v = 1 - alpha, lty = 2, col = "red", lwd = 2)

legend("topleft", legend = c("Base (LAC)", "Adaptive (APS)"), 
       fill = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)))

# ---- Histogram of prediction set sizes ----
sizes_base_test = rowSums(prediction_set_base)
sizes_aps_test  = rowSums(prediction_set_aps)

x_range_size = range(c(sizes_base_test, sizes_aps_test))

hist(sizes_base_test, col = rgb(1,0,0,0.5), xlim = x_range_size, breaks = seq(min(x_range_size)-0.5, max(x_range_size)+0.5, by=1),
     main = "Distribution of prediction set size",
     xlab = "Number of classes included in the set", ylab = "Frequency")
hist(sizes_aps_test, col = rgb(0,0,1,0.5), breaks = seq(min(x_range_size)-0.5, max(x_range_size)+0.5, by=1), add = TRUE)

legend("topright", legend = c("Base (LAC)", "Adaptive (APS)"), 
       fill = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)))

# ---- Visual example: some images with their prediction sets ----
par(mfrow = c(2, 3))
sample_indices = sample(1:nrow(X_test), 6)
for (i in sample_indices) {
  image_matrix = matrix(X_test[i, ], nrow = 28, ncol = 28)
  image_matrix = t(apply(image_matrix, 2, rev))          
  included_classes = which(prediction_set_aps[i, ]) - 1  
  image(image_matrix, col = gray.colors(256), axes = FALSE,
        main = paste0("True: ", y_test[i],
                      "\nAPS Set: {", paste(included_classes, collapse = ","), "}"))
}
par(mfrow = c(1, 1))

# 6. PLOTS FOR THE PRESENTATION (illustrative only)

# ---- 6.1 Calibration scores distribution + qhat (base method) ----
hist(scores_calib, breaks = 30, col = "lightblue",
     main = "Conformal scores (calibration) - base method",
     xlab = "Score = 1 - P(true class)")
abline(v = qhat_base, col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("qhat =", round(qhat_base, 3)),
       col = "red", lty = 2, bty = "n")

# ---- 6.2 Same, for APS method ----
hist(scores_calib_aps, breaks = 30, col = "lightgreen",
     main = "Conformal scores (calibration) - APS method",
     xlab = "Cumulative probability up to true class")
abline(v = qhat_aps, col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("qhat =", round(qhat_aps, 3)),
       col = "red", lty = 2, bty = "n")

# ---- 6.3 Reliability diagram (is the raw network well calibrated?) ----
max_prob_test = apply(probabilities_test, 1, max)
pred_test = apply(probabilities_test, 1, which.max) - 1
correct_test = as.numeric(pred_test == y_test)

bins = cut(max_prob_test, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)
reliability = tapply(correct_test, bins, mean)
bin_centers = seq(0.05, 0.95, by = 0.1)

plot(bin_centers, reliability, type = "b", pch = 19, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted confidence (max prob)", ylab = "Observed accuracy",
     main = "Reliability diagram (before conformal prediction)")
abline(0, 1, col = "red", lty = 2)  # Perfect calibration line

# ---- 6.4 Empirical coverage vs alpha ----
alphas = seq(0.01, 0.4, by = 0.02)
coverage_by_alpha = sapply(alphas, function(a) {
  q = conformal_quantile(scores_calib, a)
  pset = build_prediction_set(probabilities_test, q)
  mean(sapply(1:nrow(X_test), function(i) return(pset[i, y_test[i] + 1])))
})

plot(1 - alphas, coverage_by_alpha, type = "b", pch = 19,
     xlab = "Target coverage (1-alpha)", ylab = "Empirical coverage",
     main = "Empirical coverage vs target, across alpha values")
abline(0, 1, col = "red", lty = 2)

# ---- 6.5 Average set size vs alpha (coverage-efficiency trade-off) ----
size_by_alpha = sapply(alphas, function(a) {
  q = conformal_quantile(scores_calib, a)
  pset = build_prediction_set(probabilities_test, q)
  mean(rowSums(pset))
})
plot(1 - alphas, size_by_alpha, type = "b", pch = 19, col = "darkblue",
     xlab = "Target coverage (1-alpha)", ylab = "Average set size",
     main = "Average set size vs required coverage")

# ---- 6.6 Coverage by digit (base method) ----
coverage_by_class = sapply(0:9, function(d) {
  idx = which(y_test == d)
  mean(sapply(idx, function(i) return(prediction_set_base[i, d + 1])))
})
barplot(coverage_by_class, names.arg = 0:9, col = "coral",
        main = "Coverage by digit (base method, LAC)", ylab = "Coverage", xlab = "True digit")
abline(h = 1 - alpha, col = "red", lty = 2)

# ---- 6.7 Average APS set size by digit ----
size_by_class = sapply(0:9, function(d) {
  idx = which(y_test == d)
  mean(rowSums(prediction_set_aps[idx, , drop = FALSE]))
})
barplot(size_by_class, names.arg = 0:9, col = "seagreen",
        main = "Average APS set size by digit", ylab = "Average set size", xlab = "True digit")

# ---- 6.8 Training loss curve ----
plot(1:n_epochs, loss_history, type = "l", col = "steelblue", lwd = 2,
     xlab = "Epoch", ylab = "Loss (cross-entropy)",
     main = "Training loss over epochs")

# ---- 6.9 LAC vs APS set size, image by image ----
plot(jitter(rowSums(prediction_set_base)), jitter(rowSums(prediction_set_aps)),
     xlab = "Set size (LAC)", ylab = "Set size (APS)",
     main = "Set size comparison, image by image", pch = 16, col = rgb(0, 0, 0, 0.2))
abline(0, 1, col = "red", lty = 2)

# ---- 6.10 Confusion matrix on test set ----
pred_test_confmat = apply(probabilities_test, 1, which.max) - 1
confusion_matrix_test = table(Predicted = pred_test_confmat, True = y_test)
print(confusion_matrix_test)
