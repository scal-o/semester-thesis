library(dplyr)

nodes <- readr::read_csv("networks/sioux_falls/sioux_falls_nodes_updated.csv")
# nodes |> rename(Node = id, X = x, Y = y) |> readr::write_csv("networks/sioux_falls/sioux_falls_nodes_updated.csv")
net <- readr::read_csv("networks/sioux_falls/sioux_falls_net.csv")

links <- net |>
    select(LINK, A, B)

colnames(links) <- c("id", "from", "to")

wkt_links <- links  |> 
    left_join(nodes, by = join_by(from == Node)) |> 
    rename(start_x = X, start_y = Y) |> 
    left_join(nodes, by = join_by(to == Node)) |> 
    rename(end_x = X, end_y = Y)

wkt_links <- wkt_links |> 
    mutate(geom = paste0("LINESTRING (",start_x," ", start_y, ", ", end_x, " ", end_y,")")
)
    
readr::write_csv(wkt_links, "networks/sioux_falls/sioux_falls_wkt_links.csv")


    