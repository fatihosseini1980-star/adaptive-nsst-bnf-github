# ozone2 data

The analysis uses `ozone2` from the R package `fields`.
The CSV is not duplicated here. Generate it from the packaged source data with:

```bash
Rscript scripts/export_ozone2.R
```

This creates:

```text
data/ozone2_long.csv
```

Expected columns are `station`, `day`, `longitude`, `latitude`, and `ozone`.
