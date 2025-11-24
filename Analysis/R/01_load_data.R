load_simulation_data <- function(data, metadata) {
  
  path_data      <- here::here(data)
  path_meta_data <- here::here(metadata)
  
  simulation <- read.csv(path_data)
  meta    <- read.csv(path_meta_data)
  
  list(
    simulation = simulation,
    meta       = meta
  )
}
