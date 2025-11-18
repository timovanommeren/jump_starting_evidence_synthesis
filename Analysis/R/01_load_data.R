load_simulation_data <- function() {
  
  path_data      <- here::here("simulation_results/added_IVs/all_simulation_results.csv")
  path_meta_data <- here::here("Analysis/percentage_relevant.csv")
  
  simulation <- read.csv(path_data)
  meta    <- read.csv(path_meta_data)
  
  list(
    simulation = simulation,
    meta       = meta
  )
}
