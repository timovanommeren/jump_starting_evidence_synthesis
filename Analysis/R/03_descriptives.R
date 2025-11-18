descriptive_tables <- function(data) {
  
  #Summary by condition
  desc_by_cond <- data %>%
    group_by(condition) %>%
    summarise(
      td_mean = mean(td),
      td_sd   = sd(td),
      n       = n(),
      .groups = "drop"
    )
  
  # Return results as a list
  list(by_condition = desc_by_cond)
}

descriptive_barchart <- function(data, metadata) {
  
  # Step 1: mean per dataset x condition (to compute contrast) 
  means <- data %>% 
    group_by(dataset, condition) %>% 
    summarise(mean_value = mean(td), .groups = "drop") %>% 
    pivot_wider(names_from = condition, values_from = mean_value) %>% 
    mutate(contrast = .data[["llm"]] - .data[["no_priors"]]) %>% 
    select(dataset, contrast) 
  
  # Step 2: join contrast back to full data 
  data_with_contrast <- data %>% left_join(means, by = "dataset") 
  
  # Step 3: compute mean and SE per dataset x condition 
  plot_data <- data_with_contrast %>% 
    group_by(dataset, condition, contrast) %>% 
    summarise( 
      n = n(), 
      td_mean = mean(td), 
      td_se = sd(td) / sqrt(n),
      .groups = "drop" 
    ) %>% 
    tidyr::replace_na(list(td_se = 0)) 
  
  # if only 1 obs, sd = NA → set SE to 0 # Join % to your plot_data and set the x-order by pct (descending = highest % first) 
  plot_data <- plot_data %>% 
    left_join(metadata %>% select(dataset, pct), by = c("dataset" = "dataset")) %>% 
    mutate(dataset = fct_reorder( 
      paste0(dataset, " (", pct, "%)"), 
      pct, .desc = TRUE))
  
  plot <- ggplot(
    plot_data,
    aes(x = dataset,
        y = td_mean,
        fill = factor(condition))
  ) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = td_mean - td_se, ymax = td_mean + td_se),
                  position = position_dodge(width = 0.8), width = 0.2) +
    geom_hline(yintercept = 100,
               linetype = "dashed",
               color = "black",
               linewidth = 0.4) +
    labs(
      title = "",
      x = "Datasets <span style='color:#888888;'>(ordered by percentage of relevant records)</span>", 
      y = "Number of relevant records",
      fill = "Examples given before screening:"             
    ) +
    scale_fill_manual(                      
      values = c(
        llm       = "chartreuse3",
        minimal   = "blue",
        no_priors = "red"
        
      ),
      breaks = c("llm", "minimal", "no_priors"),
      labels = c("LLM-generated", "True examples", "None")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title.x = element_markdown(),  # enables HTML styling for x label
      axis.title.y = element_markdown(),   # enables HTML styling for y label
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      plot.margin = margin(10, 20, 10, 10),
      axis.text.x = element_text(angle = 55, hjust = 1),
      legend.position = "top"               # optional
    )
  
  ggsave(
    filename = here::here("Report/results/td_barchart.png"),
    plot = plot,
    width = 10,                 # width in inches
    height = 6,                 # height in inches
    dpi = 300                   # resolution (300 is print-quality)
  )
  
  return(plot)

}