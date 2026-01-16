
# Function for generic data preparation
prepare_data <- function(simulation_long, metadata) {

  #filter for run 1 only
  data_run1 <- simulation_long %>%
    filter(run == 1)

  #transform data from wide to long format
  data = tidyr::pivot_wider(simulation_long,
                     id_cols    = c(dataset, condition, run, n_trials, n_abstracts, length_abstracts, llm_temperature),
                     names_from = metric,
                     values_from = value)
  
  meta <- metadata

  #change column name metadata_datasets from Dataset to dataset to match data
  colnames(meta)[colnames(meta) == 'Dataset'] <- 'dataset'
  colnames(meta)[colnames(meta) == 'pct'] <- 'percent_rel'
  colnames(meta)[colnames(meta) == 'Records'] <- 'records'
  colnames(meta)[colnames(meta) == 'Topics'] <- 'topic'
  colnames(meta)[colnames(meta) == 'Included'] <- 'included'

  print(head(meta))
  
  # add metadata variables to data
  data <- data %>%
    left_join(meta %>% dplyr::select(dataset, records, percent_rel, topic, included), by = "dataset")

  # truncate the topic variable to only include the first topic (i.e., until the comma)
  data <- data %>%
    mutate(topic = stringr::str_extract(topic, "^[^,]+"))

  #rescale variables
  data <- data %>%
    mutate(
      length_abstracts = length_abstracts / 100,
      records = records / 1000
    )

  # return as list
  list(
    simulation = data,
    meta       = meta
  )

}

#Function for implicit dummy variable creation through zero imputation

two_part_coding <- function(data) {

  data <- data %>%
    mutate(
      
      # indicator where llm parameters are actually defined
      llm_active = (condition == "llm"),
      
      # set to 0 if not llm condition (two-part coding)
      n_abstracts_llm = ifelse(
        llm_active,
        n_abstracts - mean(n_abstracts[llm_active], na.rm = TRUE),
        0
      ),
      length_abstracts_llm = ifelse(
        llm_active,
        length_abstracts  - mean(length_abstracts[llm_active], na.rm = TRUE),
        0
      ),
      llm_temperature_llm = ifelse(
        llm_active,
        llm_temperature - mean(llm_temperature[llm_active], na.rm = TRUE),
        0
      )
    )
  
  return(data)
}


