Install (if needed) and load required packages
install.packages('EnvStats')
install.packages('dixonTest')
install.packages('outliers')
install.packages('jsonlite')

library(outliers)
library(EnvStats)
library(dixonTest)
library(jsonlite)

set.seed(42)

generate_list_with_outliers <- function(n, num_outliers) {
  base_list <- rnorm(n, mean = 50, sd = 10)

  if (num_outliers > 0) {
    outliers <- rnorm(num_outliers, mean = 100, sd = 10)
    outlier_indices <- sample(1:n, num_outliers)
    base_list[outlier_indices] <- outliers
  }

  return(base_list)
}

generate_lists <- function(sizes, outliers) {
  list_of_lists <- list()

  for (size in sizes) {
    for (outliers in outliers) {
      version_name <- paste(size, "elements", outliers, "outliers", sep = "_")
      list_of_lists[[version_name]] <- generate_list_with_outliers(size, outliers)
    }
  }

  return(list_of_lists)
}
rosnor_sizes <- c(25, 40, 50)
rosnor_outliers <- c(0, 1, 2)
rosnor_lists <- generate_lists(sizes=rosnor_sizes, outliers=rosnor_outliers)

dixon_sizes <- c(3, 10, 26)
dixon_outliers <- c(0, 1)
dixon_lists <- generate_lists(sizes=dixon_sizes, outliers=dixon_outliers)

grubbs_sizes <- c(3, 15, 30)
grubbs_outliers <- c(0, 1)
grubbs_lists <- generate_lists(sizes=grubbs_sizes, outliers=grubbs_outliers)

dixon_options <- c("two.sided", "greater", "less")

results <- list()

# -- Rosner Test --
for (list_name in names(rosnor_lists)) {
  data_vector <- rosnor_lists[[list_name]]

  rosner_result <- rosnerTest(data_vector, k = 5)
  rosnor_q       <- rosner_result$statistic
  rosner_stats   <- rosner_result$all.stats

  results[[list_name]] <- list(
    data   = data_vector,
    rosner = list(
      rosnor_q  = unname(rosnor_q),
      all_stats = rosner_stats
    )
  )
}

# -- Dixon Test --
for (list_name in names(dixon_lists)) {
  data_vector <- dixon_lists[[list_name]]
  dixon_results <- list()

  for (option in dixon_options) {
    dixon_res <- dixonTest(data_vector, alternative = option, refined = FALSE)
    dixon_results[[option]] <- list(
      statistic = unname(dixon_res$statistic),
      p_value   = unname(dixon_res$p.value),
      estimate  = unname(dixon_res$estimate)
    )
  }
  results[[list_name]] <- list(
    data   = data_vector,
    dixon  = dixon_results
  )
}

# -- Grubbs Test --
for (list_name in names(grubbs_lists)) {
  data_vector <- grubbs_lists[[list_name]]

  grubbs_result <- grubbs.test(data_vector)
  grubbs_obj <- list(
    statistic = unname(grubbs_result$statistic),
    p_value   = unname(grubbs_result$p.value)
  )

  results[[list_name]] <- list(
    data   = data_vector,
    grubbs = grubbs_obj
  )
}

json_data <- toJSON(results, pretty = TRUE)
write(json_data, "test_data\\results.json")
cat(json_data)

