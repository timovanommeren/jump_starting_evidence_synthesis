descriptive_tables <- function(data) {

  #Summary of number of papers found by condition
  desc_by_cond <- data %>%
    group_by(condition) %>%
    summarise(
      papers_found_mean = mean(papers_found),
      papers_found_sd   = sd(papers_found),
      n       = n(),
      .groups = "drop"
    )

  #Summary of

  # Return results as a list
  list(by_condition = desc_by_cond)
}



library(dplyr)
library(forcats)
library(ggplot2)
library(rlang)

descriptive_barchart <- function(data, metadata, variable, y_axis = 100) {
  var_quo  <- enquo(variable)          # capture column
  var_name <- as_name(var_quo)         # e.g. "papers_found"
  
  # Step 1: mean per dataset x condition
  means <- data %>%
    group_by(dataset, condition) %>%
    summarise(
      mean_value = mean(!!var_quo),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(names_from = condition, values_from = mean_value) %>%
    mutate(contrast = .data[["llm"]] - .data[["no_initialisation"]]) %>%
    dplyr::select(dataset, contrast)
  
  # Step 2: join contrast back to full data
  data_with_contrast <- data %>% left_join(means, by = "dataset")
  
  # Step 3: compute mean and SE per dataset x condition
  plot_data <- data_with_contrast %>%
    group_by(dataset, condition, contrast) %>%
    summarise(
      n = n(),
      variable_mean = mean(!!var_quo),
      variable_se   = sd(!!var_quo) / sqrt(n),
      .groups = "drop"
    ) %>%
    tidyr::replace_na(list(variable_se = 0))
  
  plot_data <- plot_data %>%
    left_join(metadata %>% dplyr::select(dataset, percent_rel), by = "dataset") %>%
    mutate(
      dataset = fct_reorder(
        paste0(dataset, " (", percent_rel, "%)"),
        percent_rel,
        .desc = TRUE
      ),
      condition = factor(
        condition,
        levels = c("llm", "criteria", "random", "no_initialisation")
      )
    )
  
  
  plot <- ggplot(
    plot_data,
    aes(x = dataset, y = variable_mean, fill = factor(condition))
  ) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(
      aes(ymin = variable_mean - variable_se,
          ymax = variable_mean + variable_se),
      position = position_dodge(width = 0.8),
      width = 0.2
    ) +
    geom_hline(yintercept = y_axis, linetype = "dashed", color = "black", linewidth = 0.4) +
    labs(
      title = "",
      x = "Datasets <span style='color:#888888;'>(ordered by percent_rel of relevant records)</span>",
      y = "",
      fill = "Examples given before screening:"
    ) +
    scale_fill_manual(
      values = c(
        llm = "chartreuse3",
        random = "blue",
        criteria = "orange",
        no_initialisation = "red"
      ),
      breaks = c("llm", "criteria", "random", "no_initialisation"),
      labels = c("LLM", "Inclusion criteria", "True examples", "Cold Start")
    ) +
    theme_minimal() +
    theme(
      plot.title    = element_text(hjust = 0.5, face = "bold"),
      axis.title.x  = ggtext::element_markdown(),
      axis.title.y  = ggtext::element_markdown(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key       = element_rect(fill = "white", color = NA),
      plot.margin = margin(10, 20, 10, 10),
      axis.text.x = element_text(angle = 55, hjust = 1),
      legend.position = "top"
    )
  
  ggsave(
    filename = here::here(paste0("Report/results/", var_name, "_barchart.png")),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 300
  )
  
  plot
}




plots_llm_vs_no_init_conditions <- function(simulation, dataset_name) {

  library(ggdist)

  diff <- simulation %>%
    filter(dataset == dataset_name) %>%
    dplyr::select(run, condition, papers_found) %>%
    mutate(diff_llm_no = llm - no_priors)


  ggplot(diff, aes(x = diff_llm_no)) +
    geom_histogram(aes(y = ..density..), bins = 10, alpha = 0.6) +
    geom_density(linewidth = 0.8) +
    labs(
      x = "papers_found_llm − papers_found_no_priors (per run)",
      y = "Density",
      title = paste("Distribution of papers_found difference (LLM vs no priors) – ", dataset_name)
    )

  ggplot(diff, aes(x = diff_llm_no)) +
    stat_ecdf(linewidth = 0.8) +
    labs(
      x = "papers_found_llm − papers_found_no_priors",
      y = "Empirical CDF",
      title = paste("ECDF of papers_found difference – ", dataset_name)
    )

  ggplot(diff, aes(x = "", y = diff_llm_no)) +
    geom_boxplot(width = 0.2, outlier.shape = NA) +
    geom_jitter(width = 0.05, height = 0, alpha = 0.6) +
    labs(
      x = NULL,
      y = "papers_found_llm − papers_found_no_priors",
      title = paste("Run-wise differences in papers_found – ", dataset_name)
    ) +
    theme(axis.text.x = element_blank())

  ggplot(diff, aes(x = "", y = diff_llm_no)) +
    geom_violin(trim = FALSE, alpha = 0.6) +
    geom_jitter(width = 0.05, height = 0, alpha = 0.6) +
    labs(x = NULL, y = "papers_found_llm − papers_found_no_priors")


  ggplot(diff, aes(sample = diff_llm_no)) +
    stat_qq() +
    stat_qq_line() +
    labs(
      title = paste("QQ plot of papers_found difference – ", dataset_name),
      x = "Theoretical quantiles",
      y = "Sample quantiles"
    )

  ggplot(diff, aes(x = diff_llm_no, y = 1)) +
    stat_halfeye(point_interval = median_qi) +
    labs(
      x = "papers_found_llm − papers_found_no_priors",
      y = NULL,
      title = paste("Sampling distribution of papers_found difference – ", dataset_name)
    ) +
    theme(axis.text.y = element_blank())


}



# ggplot(
#   subset(simulation, !dataset %in% c("Walker_2018", "Brouwer_2019")),
#   aes(x = records, y = papers_found)) +
#   geom_point(alpha = 0.6) +
#   geom_smooth(method = "lm",
#               aes(color = "linear"),
#               se = FALSE) +
#   geom_smooth(method = "lm",
#               formula = y ~ x + I(x^2),
#               aes(color = "quadratic"),
#               se = FALSE) +
#   labs(
#     title = "Scatter plot of records vs. Number of relevant records found",
#     x = "Number of records",
#     y = "Number of relevant records found in first 100 screened"
#   ) +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold"),
#     panel.background = element_rect(fill = "white", color = NA),
#     plot.background  = element_rect(fill = "white", color = NA),
#     legend.background = element_rect(fill = "white", color = NA),
#     legend.key = element_rect(fill = "white", color = NA),
#     plot.margin = margin(10, 20, 10, 10)
#   )
