# Export fields::ozone2 to the long CSV expected by src/run_ozone2.py.
if (!requireNamespace("fields", quietly = TRUE)) install.packages("fields")
library(fields)
data(ozone2)
out <- do.call(rbind, lapply(seq_len(ncol(ozone2$y)), function(j) {
  data.frame(
    station = j,
    day = seq_len(nrow(ozone2$y)),
    longitude = ozone2$lon.lat[j, 1],
    latitude = ozone2$lon.lat[j, 2],
    ozone = ozone2$y[, j]
  )
}))
write.csv(out, "data/ozone2_long.csv", row.names = FALSE)
