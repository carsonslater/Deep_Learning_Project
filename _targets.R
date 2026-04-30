# =============================================================================
# _targets.R
# Production-ready targets pipeline — AMI HMM Project
# Stack: targets, tarchetypes, duckdb, duckplyr, arrow, data.table
# =============================================================================

library(targets)
library(tarchetypes)
library(crew)

# Source all helper functions from R/
tar_source()

# ---------------------------------------------------------------------------
# Crew parallel controllers
# ---------------------------------------------------------------------------
# Two named controllers let us tune parallelism per target type:
#
#   "main"  — general-purpose workers for I/O-bound steps (DuckDB reads,
#              Parquet writes, Arrow conversions).  4 workers is a safe ceiling
#              on a 16 GB machine: each worker can hold ~1–2 GB for a meter
#              slice + DuckDB in-process engine without swapping.
#
#   "api"   — rate-limited workers exclusively for the meter_weather branches
#              that call the Open-Meteo Archive API.  2 workers means at most
#              2 simultaneous HTTP requests, well within the free-tier limits
#              (the API allows ~10 req/min per IP; each call covers months of
#              data so 2 concurrent workers is plenty for throughput).
#
# crew_controller_group() exposes both under one handle; targets routes each
# branch to the right controller via tar_resources(crew = tar_resources_crew()).

controllers <- crew_controller_group(
  crew_controller_local(
    name    = "main",
    workers = 4L # 4 workers — safe for 16 GB RAM
  ),
  crew_controller_local(
    name    = "api",
    workers = 2L # 2 workers — throttles Open-Meteo API calls
  )
)

# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------
tar_option_set(
  packages = c(
    "data.table",
    "duckdb",
    "duckplyr",
    "arrow",
    "httr2",
    "lubridate",
    "glue",
    "stringr",
    "crew"
  ),
  # qs is faster and more space-efficient than RDS for intermediate objects
  format = "qs",
  # Default controller — most targets run on "main"
  controller = controllers,
  resources = tar_resources(
    crew = tar_resources_crew(controller = "main")
  )
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
list(
  # 1. DuckDB path  --------------------------------------------------------
  #    Returns the path string to the persistent DuckDB file.
  #    Each function that needs DB access opens its own connection internally
  #    with on.exit() — this avoids serialising a live DBI pointer (which
  #    causes "Invalid connection" errors when targets deserialises targets).
  tar_target(
    db_path,
    {
      dir.create("data", showWarnings = FALSE, recursive = TRUE)
      "data/ami_db.duckdb"
    }
  ),

  # 2. Raw file paths (format = "file" triggers re-run on file change) ----
  tar_target(
    raw_usage_path,
    "all_data.rds",
    format = "file"
  ),
  tar_target(
    raw_metadata_path,
    "AMI_all_data_fields.csv",
    format = "file"
  ),

  # 3. One-time RDS → Parquet conversion ----------------------------------
  #    Runs exactly once; re-runs only if raw_usage_path changes.
  #    format = "file" means targets tracks the Parquet file on disk —
  #    subsequent targets depend on this path string, NOT an in-memory object.
  #
  #    WHY PARQUET:
  #    Parquet stores data column-by-column.  This means the branching target
  #    below can read only 2 columns (date_time + one meter) per branch
  #    instead of loading the full 3 GB object for every meter.
  tar_target(
    raw_usage_parquet,
    convert_rds_to_parquet(
      rds_path     = raw_usage_path,
      parquet_path = "data/all_data.parquet"
    ),
    format = "file"
  ),

  # 3b. Register Parquet as a persistent DuckDB VIEW -------------------------
  #     Creates (or replaces) 'usage_view' pointing at data/all_data.parquet.
  #     Runs exactly ONCE — before any meter_usage branch starts.
  #
  #     WHY:
  #     Without this, each branch calls FROM read_parquet('data/all_data.parquet')
  #     which forces DuckDB to re-open the file and re-parse the ~MB metadata
  #     footer on every single branch — O(N_meters × file_open_cost).
  #     With the VIEW, DuckDB caches the handle in the persistent .duckdb file.
  #     All branches resolve 'usage_view' instantly; column projection still
  #     applies so only 2 columns are read from disk per branch.
  #
  #     OUTPUT: the string "usage_view" (sentinel for downstream targets)
  tar_target(
    usage_view,
    register_usage_view(
      db_path      = db_path,
      parquet_path = raw_usage_parquet,
      view_name    = "usage_view"
    )
  ),

  # 4. Meter IDs -----------------------------------------------------------
  #    Reads the RDS once for column names, then frees memory.
  #    Upstream of the branching target so this cost is paid exactly once.
  tar_target(
    meter_ids,
    get_meter_ids(raw_usage_path)
  ),

  # 5. Meter locations → DuckDB -------------------------------------------
  #    Reads only 3 columns from the CSV via Arrow's column-projection,
  #    deduplicates, casts meter_id to character, and writes to DuckDB with
  #    an index on meter_id for fast downstream joins.
  #
  #    RETURNS: a dplyr::tbl() lazy reference — no data.frame held in memory.
  #    Downstream targets join against this tbl() and call collect() only
  #    when rows are actually needed in R.
  tar_target(
    meter_locations,
    load_meter_locations(
      metadata_path = raw_metadata_path,
      db_path       = db_path,
      table_name    = "meter_locations"
    )
  ),

  # 6. Per-meter usage extraction + location join (dynamic branching) ---------
  #
  #    pattern = map(meter_ids) → one branch per meter ID.
  #
  #    EXECUTION MODEL (parallel-safe):
  #      Each branch opens its own read-only DuckDB connection to db_path.
  #      DuckDB supports many concurrent read-only readers on the same file,
  #      so tar_make_future(workers = N) is fully safe here.
  #
  #    QUERY PLAN (inside DuckDB per branch):
  #      read_parquet(parquet_path)     ← 2-column projection from disk
  #        └─ INNER JOIN meter_locations ← already indexed on meter_id
  #             └─ WHERE usage / lat / lon IS NOT NULL
  #      → collect() into data.table    ← only the final small result hits R
  #
  #    SENTINEL: `meter_locations` is passed so targets knows this target
  #      depends on the DB table being written first.  The object is not used
  #      inside the function body.
  #
  #    OUTPUT per branch: data.table [meter_id, date_time, usage, lat, lon]
  tar_target(
    meter_usage,
    extract_meter_usage(
      parquet_path    = raw_usage_parquet,
      meter_id        = meter_ids,
      db_path         = "data/ami_db.duckdb",
      meter_locations = meter_locations, # sentinel: locations table must exist
      usage_view      = usage_view # sentinel + view name used in SQL
    ),
    pattern = map(meter_ids),
    format = "file"
  ),

  # 7. Per-meter weather fetch + write Parquet (dynamic branching) -----------
  #
  #    pattern = map(meter_usage) → one branch per meter slice.
  #
  #    WHAT CHANGED FROM format = "qs":
  #      attach_weather() now writes data/parquet/meter_<id>.parquet and returns
  #      the file path.  format = "file" tells targets to track the Parquet file
  #      on disk — the branch only re-runs if the file is deleted or modified.
  #
  #    PARALLEL SAFETY:
  #      Each branch writes to its OWN file (no shared file handle).
  #      The ASOF JOIN uses an in-memory DuckDB (no dbdir) — zero contention.
  #
  #    OUTPUT per branch: character scalar path to meter_<id>.parquet
  tar_target(
    meter_weather,
    attach_weather(meter_usage, meter_ids),
    pattern = map(meter_usage, meter_ids),
    format = "file",
    resources = tar_resources(
      crew = tar_resources_crew(controller = "api")
    )
  ),

  # 8. Register DuckDB VIEW over Parquet directory ---------------------------
  #
  #    Runs exactly ONCE after ALL meter_weather branches complete.
  #    meter_weather (the aggregated path vector) is passed as a sentinel,
  #    ensuring targets enforces the ordering.
  #
  #    WHY VIEW OVER PARQUET:
  #      • No write contention — branches wrote independent files in parallel.
  #      • No data duplication — the Parquet files ARE the store.
  #      • Predicate pushdown — DuckDB filters inside the Parquet reader.
  #      • Auto-discovery — new meter files appear in queries automatically.
  #
  #    OUTPUT: "climate_data" (the view name string)
  tar_target(
    climate_view,
    register_climate_view(
      db_path             = db_path,
      parquet_dir         = "data/parquet",
      view_name           = "climate_data",
      meter_weather_paths = meter_weather # sentinel: wait for all branches
    )
  ),

  # 9. Validation summary — per-meter aggregation via DuckDB -----------------
  #
  #    A single GROUP BY query runs entirely inside DuckDB against the
  #    climate_data VIEW (which scans data/parquet/*.parquet lazily).
  #    Only ~1 row per meter ever reaches R memory.
  #
  #    DuckDB optimisations used:
  #      • COUNT(*)  → resolved from Parquet row-group metadata (no decode)
  #      • MIN/MAX   → resolved from Parquet column statistics (no decode)
  #      • NULL counts → validity bitmaps, no value scan needed
  #
  #    OUTPUT: data/parquet/climate_summary.parquet
  #      Columns: meter_id, n_obs, pct_missing_temp, pct_missing_precip,
  #               min_usage, max_usage, date_start, date_end
  tar_target(
    climate_summary,
    summarize_climate_data(
      db_path      = db_path,
      climate_view = climate_view, # sentinel: VIEW must exist first
      view_name    = "climate_data",
      out_path     = "data/parquet/climate_summary.parquet"
    ),
    format = "file"
  )
)
