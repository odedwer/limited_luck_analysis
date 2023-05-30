LoadPackage <- function(name) {
  if (!require(name, character.only = T)) install.packages(name); library(name, character.only = T)
}

LoadPackage("dplyr")
LoadPackage("plyr")
LoadPackage("tidyverse")
LoadPackage("glmnet")
LoadPackage("ggplot2")
LoadPackage("rjags")
LoadPackage("runjags")
LoadPackage("LaplacesDemon")

RunJagsModel <- function(data_list, filename, parameters, n_chains = 3, burnout = 500, n_iter = 10000) {
  bimodal_jags_model <- jags.model(file = filename, data = data_list, n.chains = n_chains)
  print("burnout...")
  update(bimodal_jags_model, n.iter = burnout)
  print("running model...")
  coda.samples(bimodal_jags_model, variable.names = parameters, n.iter = n_iter)
}

gammaShRaFromModeSD <- function(mode, sd) {
  if (mode <= 0) stop("mode must be > 0")
  if (sd <= 0) stop("sd must be > 0")
  rate <- (mode + sqrt(mode^2 + 4 * sd^2)) / (2 * sd^2)
  shape <- 1 + mode * rate
  return(list(shape = shape, rate = rate))
}

RunModel <- function(dist, model_string, model_filename, params, n_iter = 10000) {
  data_list <- list(distMean = mean(dist),
                    sd = sd(dist),
                    sh = gammaShRaFromModeSD(sd(dist), 5 * sd(dist))$shape,
                    ra = gammaShRaFromModeSD(sd(dist), 5 * sd(dist))$rate,
                    y = dist,
                    N = length(dist))
  writeLines(model_string, model_filename)
  RunJagsModel(data_list, model_filename, params, n_iter = n_iter)
}

Swap <- function(a, b, i)
{
  tmp <- a[i]
  a[i] <- b[i]
  b[i] < tmp
}

GetBimodalComparison <- function(dist, n_iter = 10000) {
  # run the model
  print("running bimodal model...")
  bimodal_posterior <- RunModel(as.numeric(dist), BIMODAL_MODEL_STRING, BIMODAL_MODEL_FILENAME, BIMODAL_PARAMS, n_iter = n_iter)
  print("running unimodal model...")
  unimodal_posterior <- RunModel(as.numeric(dist), UNIMODAL_MODEL_STRING, UNIMODAL_MODEL_FILENAME, UNIMODAL_PARAMS, n_iter = n_iter)
  # calculate the log likelihood of both models
  # extract the center value of the posterior estimation
  biMu1 <- sapply(bimodal_posterior[, "biMu[1]"], median)
  biMu2 <- sapply(bimodal_posterior[, "biMu[2]"], median)
  biSigma1 <- sapply(bimodal_posterior[, "biSigma[1]"], median)
  biSigma2 <- sapply(bimodal_posterior[, "biSigma[2]"], median)
  biNu1 <- sapply(bimodal_posterior[, "biNu[1]"], median)
  biNu2 <- sapply(bimodal_posterior[, "biNu[2]"], median)

  uniMu <- sapply(unimodal_posterior[, "uniMu"], median)
  uniSigma <- sapply(unimodal_posterior[, "uniSigma"], median)
  uniNu <- sapply(unimodal_posterior[, "uniNu"], median)


  # for every subject, choose whether they come from distribution 1 or 2
  print("calculating BIC...")
  bimodal_LL <- rep(0, 3)
  unimodal_LL <- rep(0, 3)
  for (chain in 1:3) {
    bimodal_subject_mapping <- round(colMeans(bimodal_posterior[[chain]][, paste0("mBi[", seq_along(dist), "]")]))
    mus <- NULL
    mus[bimodal_subject_mapping == 1] <- biMu1[chain]
    mus[bimodal_subject_mapping == 2] <- biMu2[chain]
    sigmas <- NULL
    sigmas[bimodal_subject_mapping == 1] <- biSigma1[chain]
    sigmas[bimodal_subject_mapping == 2] <- biSigma2[chain]
    nus <- NULL
    nus[bimodal_subject_mapping == 1] <- biNu1[chain]
    nus[bimodal_subject_mapping == 2] <- biNu2[chain]

    bimodal_LL[chain] <- -sum(dstp(dist, mus, 1 / sigmas^2, nus, log = T))
    unimodal_LL[chain] <- -sum(dstp(dist, uniMu, 1 / uniSigma^2, uniNu, log = T))
  }
  unimodal_bic <- 3 * log(length(dist)) - 2 * unimodal_LL %>% mean
  bimodal_bic <- 6 * log(length(dist)) - 2 * bimodal_LL %>% mean
  print("Done.")
  return(list(unimodal_posterior = unimodal_posterior,
              bimodal_posterior = bimodal_posterior,
              unimodal_LL = unimodal_LL,
              bimodal_LL = bimodal_LL,
              unimodal_bic = unimodal_bic,
              bimodal_bic = bimodal_bic, subject_mapping = bimodal_subject_mapping))
}


GetSubjectMapping <- function(list, N) {
  list$mapping <- lapply(list, function(x) { round(colMeans(x[, paste0("mBi[", 1:N, "]")])) %>%
    as.matrix() %>%
    t() })
  list$mapping <- do.call(rbind, list$mapping)

  a <- list$mapping %>%
    as.data.frame() %>%
    gather() %>%
    table() %>%
    apply(1, function(x) { ifelse(x[1] > x[2], 1, 2) })
  list$subject.mapping <- a[paste0("mBi[", 1:N, "]")]
  list
}

BIMODAL_MODEL_FILENAME <- "bimodal_model.txt"
UNIMODAL_MODEL_FILENAME <- "unimodal_model.txt"
BIMODAL_PARAMS <- c("biMu", "biSigma", "biNu", "mBi", "mBiProb")
UNIMODAL_PARAMS <- c("uniMu", "uniSigma", "uniNu")
BIMODAL_MODEL_STRING <- "
model{
  # actually sample y from the chosen parameters
  for (i in 1:N) {
    # choose bimodal index
    mBi[i] ~ dcat(mBiProb[]) # choose one of the two distributions for this subject
    y[i] ~ dt(biMu[mBi[i]], 1/biSigma[mBi[i]]^2,biNu[mBi[i]]) T(0,)
  }

  # set probability for the bimodal options
  mBiProb[1] <- 0.5
  mBiProb[2] <- 0.5

  biMu[1] ~ dnorm(distMean,1/(10*sd)^2)
  biMu[2] ~ dnorm(distMean,1/(10*sd)^2)
  biSigma[1] ~ dgamma(sh,ra)
  biSigma[2] ~ dgamma(sh,ra)
  biNu[1] ~ dexp(30.0)
  biNu[2] ~ dexp(30.0)
}"

UNIMODAL_MODEL_STRING <- "
model{
  # actually sample y from the chosen parameters
  for (i in 1:N) {
    y[i] ~ dt(uniMu, 1/uniSigma^2,uniNu)
  }

  uniMu ~ dnorm(distMean,1/(10*sd)^2)
  uniSigma ~ dgamma(sh,ra)
  uniNu ~ dexp(30.0)

}"

# load the data
monetary_table <- read.csv2("data/processed/run1_monetary.csv", sep = ",")
shapes_table <- read.csv2("data/processed/run1_shapes.csv", sep = ",")

monetary <- list()
monetary$good_luck_dist <- data.frame(subset(monetary_table, monetary_table$pc_result == "5.0"))
monetary$mild_luck_dist <- data.frame(subset(monetary_table, monetary_table$pc_result == "1.0"))
monetary$bad_luck_dist <- data.frame(subset(monetary_table, monetary_table$pc_result == "0.0"))

shapes <- list()
shapes$data <- data.frame(shapes_table)
shapes$data$dist <- as.numeric(shapes$data$dist)

monetary$good_luck_dist$dist <- as.numeric(monetary$good_luck_dist$dist)
monetary$mild_luck_dist$dist <- as.numeric(monetary$mild_luck_dist$dist)
monetary$bad_luck_dist$dist <- as.numeric(monetary$bad_luck_dist$dist)

# write the models to text files
writeLines(BIMODAL_MODEL_STRING, BIMODAL_MODEL_FILENAME)
writeLines(UNIMODAL_MODEL_STRING, UNIMODAL_MODEL_FILENAME)

# perform the calculations
good_bayes <- GetBimodalComparison(monetary$good_luck_dist$dist)
bad_bayes <- GetBimodalComparison(monetary$bad_luck_dist$dist)
mild_bayes <- GetBimodalComparison(monetary$mild_luck_dist$dist)
shapes_bayes <- GetBimodalComparison(shapes$data$dist)