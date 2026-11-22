# DP2 labels and periods

`pull_dp2_labels.py` downloads the current classifications and periods of Rubin Data Preview 2 (DP2) objects from
[Orcus](https://orcusgate.org) and writes them to two tables:

| File | Contents |
|---|---|
| `dp2_variables.csv` | objects in the Orcus group "Rubin DP2 Variables" |
| `dp2_transients.csv` | objects in the Orcus group "Rubin DP2 Transients" |

Only objects with at least one classification in the LSST/Rubin Taxonomy are included. The labels are updated as
scanning continues, so running the script again gives the latest version.

## Access

1. You need an Orcus account that is a member of the Rubin DP2 group(s). Ask a group admin to add you; a group you are
   not in is skipped with a note.
2. Create an API token in your Orcus profile (read access is enough).

## Run

```
python pull_dp2_labels.py --token YOUR_ORCUS_TOKEN
```

or set the token once with `export ORCUS_TOKEN=YOUR_ORCUS_TOKEN` and run `python pull_dp2_labels.py`.
Only the Python standard library is needed (`pyarrow` for `--format parquet`).

| Option | Meaning |
|---|---|
| `--groups variables transients` | which groups to pull (default: both) |
| `--min-prob 0.7` | ignore classifications with a lower probability |
| `--exclude-classes CLASS ...` | leave these classes out |
| `--format csv\|parquet` | output format (default csv) |
| `--outdir DIR` | where to write the files |

## Columns

| Column | Meaning |
|---|---|
| `id` | DP2 `diaObjectId` |
| `ra`, `dec` | position in degrees |
| `class` | majority class: each person counts once (their latest classification); classes tied at the top are all listed, separated by `; ` |
| `votes` | how many people chose the class, out of how many classified the object, e.g. `2 of 3` |
| `classifications` | every person's latest classification as `author:class:probability`, separated by `; ` |
| `period` | the set period in days (the `period` field of the object's annotation named `period` on Orcus); empty unless the class is periodic (pulsators, binary stars, the periodic branch and cyclotron-emitting magnetic white dwarfs) |
| `groups` | the Orcus DP2 group(s) the object is in |

Machine-learning classifications are not counted.

## Citation

If you use these labels or periods, please cite Atal Agrawal et al., "The Rubin Source Classification Project: I.
Period Estimation and Recognition of Variable Stars in Data Preview 2", in preparation. Classifications by other people
are credited to their authors in the `classifications` column.
