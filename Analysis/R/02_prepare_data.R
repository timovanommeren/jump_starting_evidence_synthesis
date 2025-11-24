prepare_data <- function(simulation_long, metadata) {
  
  #filter for run 1 only
  data_run1 <- simulation_long %>%
    filter(run == 1)
  
  #transform data from wide to long format
  data = pivot_wider(simulation_long,
                     id_cols    = c(dataset, condition, run, n_abstracts, length_abstracts, typicality, degree_jargon, llm_temperature, timestamp),
                     names_from = metric,
                     values_from = value)
  
  meta <- metadata
  
  #change column name metadata_datasets from Dataset to dataset to match data
  colnames(meta)[colnames(meta) == 'Dataset'] <- 'dataset'
  colnames(meta)[colnames(meta) == 'pct'] <- 'percent_rel'
  colnames(meta)[colnames(meta) == 'Records'] <- 'records'
  
  
  # add 'records' and 'percent_rel' to data from metadata based on column id 'dataset'
  data <- data %>%
    left_join(meta %>% select(dataset, records, percent_rel, Topics), by = "dataset")
  
  # return as list
  list(
    simulation = data,
    meta       = meta
  )
  
}
