# =============================================================================
# AMI HMM Pipeline — Helper Functions
# =============================================================================


# -----------------------------------------------------------------------------
# 1. DuckDB connection
# -----------------------------------------------------------------------------

#' Initialize a persistent DuckDB connection
#'
#' @param db_path Path to the DuckDB file (created if absent).
#' @return A live DBI connection.
#' @export
init_duckdb_con <- function(db_path = "data/ami_db.duckdb") {
  dir.create(dirname(db_path), showWarnings = FALSE, recursive = TRUE)
  DBI::dbConnect(duckdb::duckdb(), dbdir = db_path)
}


# -----------------------------------------------------------------------------
# 2. Metadata helpers
# -----------------------------------------------------------------------------

#' Extract meter column names from an RDS file
#'
#' Reads the full object once solely to capture column names, then frees
#' memory immediately.  The caller (a dedicated \code{tar_target}) should be
#' upstream of every branching target so this read happens exactly once.
#'
#' @param path File path to \code{all_data.rds}.
#' @return Character vector of meter IDs (all columns except \code{date_time}).
#' @export
get_meter_ids <- function(path) {
  data <- readRDS(path)
  ids <- setdiff(names(data), "date_time")
  rm(data)
  gc()
  ids
}


# -----------------------------------------------------------------------------
# 3. One-time RDS → Parquet conversion
# -----------------------------------------------------------------------------

#' Convert an RDS file to Parquet format
#'
#' This runs exactly once (tracked by \code{format = "file"} in the target).
#' Parquet's columnar storage means downstream per-meter reads load only 2
#' columns instead of the full dataset, regardless of how many meters exist.
#'
#' @param rds_path   Path to \code{all_data.rds}.
#' @param parquet_path  Destination path for the Parquet file.
#' @return The \code{parquet_path} string (for \code{format = "file"} tracking).
#' @export
convert_rds_to_parquet <- function(rds_path, parquet_path) {
  dir.create(dirname(parquet_path), showWarnings = FALSE, recursive = TRUE)

  data <- readRDS(rds_path)
  arrow::write_parquet(data, parquet_path)
  rm(data)
  gc()

  parquet_path
}


# -----------------------------------------------------------------------------
# 3b. Register Parquet as a persistent DuckDB VIEW (run once)
# -----------------------------------------------------------------------------

#' Register the usage Parquet file as a DuckDB VIEW
#'
#' Creates a VIEW named \code{usage_view} that points at the Parquet file.
#' Called ONCE before any \code{meter_usage} branches start.
#'
#' NOTE: accepts \code{db_path} (a plain string) rather than a live connection
#' so that the target result is serialisable by \code{targets}.
#'
#' @param db_path      Path to the DuckDB file.
#' @param parquet_path Path to \code{data/all_data.parquet}.
#' @param view_name    Name of the VIEW to create. Default: \code{"usage_view"}.
#' @return The \code{view_name} string (serialisable sentinel).
#' @export
register_usage_view <- function(db_path,
                                parquet_path,
                                view_name = "usage_view") {
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = db_path)
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

  sql <- sprintf(
    "CREATE OR REPLACE VIEW %s AS SELECT * FROM read_parquet(%s);",
    DBI::dbQuoteIdentifier(con, view_name),
    DBI::dbQuoteString(con, parquet_path)
  )
  DBI::dbExecute(con, sql)

  message(sprintf(
    "[register_usage_view] VIEW '%s' registered over '%s'",
    view_name, parquet_path
  ))

  view_name
}


# -----------------------------------------------------------------------------
# 4. Per-meter extraction with location join  (mapped target)
# -----------------------------------------------------------------------------

#' Extract usage data for one meter and join with its location from DuckDB
#'
#' Called once per branch by \code{pattern = map(meter_ids)}.
#' Designed for safe parallel execution (\code{tar_make_future}):
#'
#' \itemize{
#'   \item Opens its own \strong{read-only} DuckDB connection — multiple
#'     parallel workers can read the same \code{.duckdb} file concurrently.
#'   \item Uses DuckDB's built-in \code{read_parquet()} to scan the Parquet
#'     file \emph{server-side}, projecting only 2 columns before the join.
#'   \item Joins with \code{meter_locations} (already in the DuckDB file)
#'     inside DuckDB — nothing large is materialised in R.
#'   \item All NULL filtering (usage, lat, lon) happens inside the SQL query.
#'   \item The connection is always closed via \code{on.exit()}.
#' }
#'
#' @param parquet_path    Path to \code{data/all_data.parquet}.
#' @param meter_id        A single meter ID string (one branch).
#' @param db_path         Path to the DuckDB file. Used to open a fresh
#'   read-only connection inside this worker process.
#' @param meter_locations A \code{dplyr::tbl()} sentinel reference. targets
#'   uses this to enforce that the \code{meter_locations} table is written
#'   before any branch starts. The object is not used in the function body.
#' @return A \code{data.table} with columns
#'   \code{meter_id}, \code{date_time}, \code{usage}, \code{lat}, \code{lon}.
#' @export
extract_meter_usage <- function(parquet_path, meter_id, db_path,
                                meter_locations, usage_view) {
  # Open READ-ONLY — safe for concurrent parallel branch workers.
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = db_path, read_only = TRUE)
  on.exit(DBI::dbDisconnect(con, shutdown = FALSE), add = TRUE)

  # Build SQL-safe identifiers/literals for this connection's dialect.
  #   col_id   → double-quoted column name : "0004100169"
  #   str_id   → single-quoted SQL literal : '0004100169'
  #   view_id  → double-quoted VIEW name   : "usage_view"
  col_id <- DBI::dbQuoteIdentifier(con, meter_id)
  str_id <- DBI::dbQuoteString(con, meter_id)
  view_id <- DBI::dbQuoteIdentifier(con, usage_view)

  # ---------------------------------------------------------------------------
  # Full pipeline executes inside DuckDB:
  #   1. Reads from usage_view (a pre-registered VIEW over the Parquet file).
  #      DuckDB reuses the cached file handle and row-group metadata — no
  #      per-branch file re-open cost.  Column projection still applies.
  #   2. INNER JOIN with meter_locations (indexed on meter_id).
  #   3. WHERE drops NULL usage, lat, lon before any data reaches R.
  # ---------------------------------------------------------------------------
  sql <- paste0(
    "SELECT\n",
    "  m.meter_id,\n",
    "  m.date_time,\n",
    "  m.usage,\n",
    "  l.lat,\n",
    "  l.lon\n",
    "FROM (\n",
    "  SELECT\n",
    "    date_time,\n",
    "    ", col_id, " AS usage,\n",
    "    SUBSTR(", str_id, ", 1, 10) AS meter_id\n", # Extract 10-digit ID prefix
    "  FROM ", view_id, "\n",
    "  WHERE ", col_id, " IS NOT NULL\n",
    ") m\n",
    "INNER JOIN meter_locations l\n",
    "  ON m.meter_id = l.meter_id\n",
    "WHERE\n",
    "  l.lat IS NOT NULL\n",
    "  AND l.lon IS NOT NULL"
  )

  result <- DBI::dbGetQuery(con, sql)
  data.table::setDT(result)

  # ── 4. Write to intermediate Parquet ───────────────────────────────────────
  # Returning a file path instead of a data.table prevents the main 'targets'
  # process from having to hash and serialize the full dataset for 2789 branches.
  out_dir <- "data/usage_slices"
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir, sprintf("usage_%s.parquet", meter_id))
  
  arrow::write_parquet(result, out_path)
  
  # Return path for format = "file"
  out_path
}


# -----------------------------------------------------------------------------
# 5. Metadata → DuckDB: meter_locations table
# -----------------------------------------------------------------------------

#' Load and clean meter location metadata into DuckDB
#'
#' Reads \code{AMI_all_data_fields.csv} via Arrow (zero-copy CSV parser),
#' selects and renames the three relevant columns, deduplicates, casts
#' \code{meter_id} to character, then writes the result into DuckDB as the
#' \code{meter_locations} table.  An index on \code{meter_id} is created for
#' fast lookup in downstream joins.
#'
#' NOTE: accepts \code{db_path} (a plain string) rather than a live connection
#' so that the target result is serialisable by \code{targets}.  A write
#' connection is opened internally and closed via \code{on.exit()}.
#'
#' @param metadata_path Path to \code{AMI_all_data_fields.csv}.
#' @param db_path       Path to the DuckDB file.
#' @param table_name    Target table name in DuckDB. Default: \code{"meter_locations"}.
#' @return The \code{table_name} string — a serialisable sentinel that
#'   downstream targets can depend on.
#' @export
load_meter_locations <- function(metadata_path,
                                 db_path,
                                 table_name = "meter_locations") {
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = db_path)
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

  # ---------------------------------------------------------------------------
  # 1. Read with Arrow — stays in Arrow columnar memory, no R data.frame yet
  # ---------------------------------------------------------------------------
  raw_arrow <- arrow::read_csv_arrow(
    metadata_path,
    col_select = c(
      "Meter_ID",
      "Service_Point_Latitude",
      "Service_Point_Longitude"
    ),
    as_data_frame = FALSE
  )

  # ---------------------------------------------------------------------------
  # 2. Clean inside Arrow compute engine (no R copies)
  # ---------------------------------------------------------------------------
  clean_arrow <- raw_arrow |>
    dplyr::rename(
      meter_id = Meter_ID,
      lat      = Service_Point_Latitude,
      lon      = Service_Point_Longitude
    ) |>
    dplyr::mutate(
      meter_id = stringr::str_pad(as.character(meter_id), width = 10, side = "left", pad = "0")
    ) |>
    dplyr::distinct()

  # ---------------------------------------------------------------------------
  # 3. Materialise once at write time
  # ---------------------------------------------------------------------------
  locs_df <- dplyr::collect(clean_arrow)

  # ---------------------------------------------------------------------------
  # 4. Write to DuckDB (idempotent)
  # ---------------------------------------------------------------------------
  DBI::dbWriteTable(
    conn = con, name = table_name, value = locs_df,
    overwrite = TRUE, row.names = FALSE
  )
  rm(locs_df, raw_arrow, clean_arrow)
  gc()

  # ---------------------------------------------------------------------------
  # 5. Index meter_id for fast downstream joins
  # ---------------------------------------------------------------------------
  DBI::dbExecute(
    con,
    glue::glue(
      "CREATE INDEX IF NOT EXISTS idx_{table_name}_meter_id ",
      "ON {table_name} (meter_id);"
    )
  )

  # Return the table name string — serialisable, usable as a sentinel
  table_name
}

# -----------------------------------------------------------------------------
# 6. Weather API Helpers
# -----------------------------------------------------------------------------

#' Fetch 15-minute weather data from Open-Meteo Archive API
#'
#' WHY RATE LIMITING MATTERS IN A PARALLEL PIPELINE:
#' When \code{crew} runs multiple \code{meter_weather} branches concurrently,
#' every worker calls this function at almost the same wall-clock instant.
#' Without a pre-request delay, all workers fire HTTP requests simultaneously,
#' which can trigger 429 (Too Many Requests) responses from Open-Meteo even
#' with only 2 API workers.  The \code{Sys.sleep()} stagger ensures each
#' worker pauses briefly before its request, spreading the load in time.
#' The exponential backoff then handles any 429s that do slip through.
#'
#' @param lat        Numeric. Latitude.
#' @param lon        Numeric. Longitude.
#' @param start_date Character/Date. Start date (YYYY-MM-DD).
#' @param end_date   Character/Date. End date (YYYY-MM-DD).
#' @param polite_delay_s Seconds to sleep before the request. Default 0.5 s.
#'   Small enough to not slow a single-worker run meaningfully; large enough
#'   to stagger simultaneous parallel workers.
#' @param cache_dir  Directory to store weather Parquet files. Default
#'   \code{"data/weather_cache"}.
#' @return A \code{data.table} with columns \code{date_time} (POSIXct,
#'   America/Denver), \code{temp_c}, \code{precip_mm}.  Returns a zero-row
#'   table (with correct types) if the request fails after all retries.
#' @export
fetch_weather_15min <- function(lat, lon, start_date, end_date,
                                polite_delay_s = 0.5,
                                cache_dir = "data/weather_cache") {
  # ── Disk cache — check before touching the network ─────────────────────────
  # Cache key: rounded to 4 decimal places (~11 m precision) so nearby meters
  # that differ only in float noise share the same cache file.
  # This is Fix #3 AND Fix #2 in one: once a (lat, lon) pair is fetched, every
  # other meter at that location reads the Parquet from disk instead of the API.
  lat_r <- round(lat, 4L)
  lon_r <- round(lon, 4L)
  cache_file <- file.path(
    cache_dir,
    sprintf(
      "lat_%s_lon_%s_%s_%s.parquet",
      gsub("\\.", "p", lat_r),
      gsub("\\.", "p", lon_r),
      as.character(start_date),
      as.character(end_date)
    )
  )

  if (file.exists(cache_file)) {
    # Fast path: load from Parquet — no network call, no API quota consumed
    dt <- arrow::read_parquet(cache_file)
    data.table::setDT(dt)
    # Ensure timezone is restored correctly after Parquet round-trip
    data.table::setattr(dt$date_time, "tzone", "America/Denver")
    return(dt)
  }

  # ── Polite pre-request delay ───────────────────────────────────────────────
  # Each parallel worker sleeps before firing its request.  This staggers
  # simultaneous calls from different workers so they don't land on the API
  # server within the same millisecond.  Parallel-safe: Sys.sleep() is
  # process-local and has no shared state between crew workers.
  Sys.sleep(polite_delay_s)

  base_url <- "https://archive-api.open-meteo.com/v1/archive"

  req <- httr2::request(base_url) |>
    httr2::req_url_query(
      latitude    = lat,
      longitude   = lon,
      start_date  = as.character(start_date),
      end_date    = as.character(end_date),
      hourly      = "temperature_2m,precipitation,snowfall",
      timezone    = "America/Denver",
      timeformat  = "unixtime" # integer timestamps → fast vectorised parsing
    ) |>
    httr2::req_retry(
      max_tries = 5,
      # TRUE exponential backoff: attempt 1 → 2 s, 2 → 4 s, 3 → 8 s, 4 → 16 s.
      # The previous `backoff = ~ 2` was a CONSTANT 2 s (ignored attempt number).
      backoff = ~ 2^.x,
      # Retry on network errors AND on 429 / 503 / 422 HTTP status codes.
      is_transient = \(resp) httr2::resp_status(resp) %in% c(429L, 503L, 422L)
    ) |>
    httr2::req_timeout(30)

  # Catch total failure (all retries exhausted) without killing the branch
  resp <- tryCatch(
    httr2::req_perform(req),
    error = function(e) {
      warning(sprintf(
        "[fetch_weather_15min] Failed after retries — lat=%s lon=%s %s to %s: %s",
        lat, lon, start_date, end_date, conditionMessage(e)
      ))
      NULL
    }
  )

  # ── Empty sentinel on failure ──────────────────────────────────────────────
  empty_dt <- function() {
    data.table::data.table(
      date_time = as.POSIXct(character(0), tz = "America/Denver"),
      temp_c    = numeric(0),
      precip_mm = numeric(0),
      snow_cm   = numeric(0)
    )
  }

  if (is.null(resp)) {
    return(empty_dt())
  }

  body <- httr2::resp_body_json(resp, simplifyVector = TRUE)
  hr_data <- body$hourly

  if (is.null(hr_data) || length(hr_data$time) == 0L) {
    warning(sprintf(
      "[fetch_weather_15min] API returned no hourly data — lat=%s lon=%s %s to %s",
      lat, lon, start_date, end_date
    ))
    return(empty_dt())
  }

  # Vectorised data.table construction — no row-by-row allocation
  dt <- data.table::data.table(
    date_time = as.POSIXct(hr_data$time,
      origin = "1970-01-01",
      tz     = "America/Denver"
    ),
    temp_c = as.numeric(hr_data$temperature_2m),
    precip_mm = as.numeric(hr_data$precipitation),
    snow_cm = as.numeric(hr_data$snowfall)
  )

  # ── Write to disk cache ────────────────────────────────────────────────────
  # Any future branch with the same (lat, lon, start, end) will hit the fast
  # path above.  arrow::write_parquet preserves POSIXct with timezone metadata.
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(dt, cache_file)

  dt
}


# -----------------------------------------------------------------------------
# 7. Attach weather + write Parquet  (mapped target)
# -----------------------------------------------------------------------------

#' Fetch weather, join to one meter's usage data, and write to Parquet
#'
#' Designed to be called once per branch via \code{pattern = map(meter_usage)}.
#' Guarantees exactly ONE API call per meter:
#' \enumerate{
#'   \item Computes \code{start_date}/\code{end_date} from \code{min}/\code{max}
#'     of \code{date_time} inside the already-collected \code{meter_usage} slice.
#'   \item Calls \code{fetch_weather_15min()} with those bounds.
#'   \item Joins via DuckDB ASOF JOIN — handles mismatched 15-min timestamps
#'     by matching each usage row to the nearest prior weather timestamp.
#'   \item Writes the result to \code{data/parquet/meter_<id>.parquet} with
#'     \code{arrow::write_parquet()} and frees R memory immediately.
#' }
#'
#' Parallel safety: uses an in-memory DuckDB (no \code{dbdir}) — each crew
#' worker gets its own private database, no file locks, no shared state.
#'
#' @param meter_usage A \code{data.table} from \code{extract_meter_usage}:
#'   columns \code{meter_id}, \code{date_time}, \code{usage}, \code{lat}, \code{lon}.
#' @param parquet_dir Directory to write per-meter Parquet files.
#'   Created if absent. Default: \code{"data/parquet"}.
#' @return The path to the written Parquet file (character scalar).
#'   Returning a path enables \code{format = "file"} in the targets target
#'   so targets re-runs the branch only when the file changes on disk.
#' @export
attach_weather <- function(meter_usage_path, meter_id,
                           parquet_dir = "data/parquet") {

  # ── 0. Load data from disk ────────────────────────────────────────────────
  # meter_usage_path is now a file path from the upstream target.
  # This offloads the memory cost from the main process to the worker.
  meter_usage <- arrow::read_parquet(meter_usage_path)
  data.table::setDT(meter_usage)
  # Output path is always stable: keyed on meter_id, not on row content.
  # This avoids the digest::digest() non-determinism that caused
  # "file does not exist" errors when targets tried to verify empty branches.
  out_path <- file.path(parquet_dir, sprintf("meter_%s.parquet", meter_id))
  dir.create(parquet_dir, showWarnings = FALSE, recursive = TRUE)

  # ── 0. Guard: empty branch → write zero-row Parquet ──────────────────────
  if (nrow(meter_usage) == 0L) {
    empty <- data.table::data.table(
      meter_id  = character(0),
      date_time = as.POSIXct(character(0), tz = "America/Denver"),
      usage     = numeric(0),
      temp_c    = numeric(0),
      precip_mm = numeric(0),
      snow_cm   = numeric(0),
      lat       = numeric(0),
      lon       = numeric(0)
    )
    arrow::write_parquet(empty, out_path)
    return(out_path)
  }

  # meter_id_val used only for warnings below (meter_id arg is the stable key)
  meter_id_val <- meter_id

  # ── 1. Date range — drives exactly ONE API call ────────────────────────────
  start_date <- as.Date(min(meter_usage$date_time))
  end_date <- as.Date(max(meter_usage$date_time))
  lat <- meter_usage$lat[[1L]]
  lon <- meter_usage$lon[[1L]]

  # ── 2. Fetch weather ───────────────────────────────────────────────────────
  weather_dt <- fetch_weather_15min(lat, lon, start_date, end_date)

  if (nrow(weather_dt) == 0L) {
    warning(sprintf(
      "[attach_weather] No weather data for meter %s (lat=%s, lon=%s, %s to %s)",
      meter_id_val, lat, lon, start_date, end_date
    ))
    dir.create(parquet_dir, showWarnings = FALSE, recursive = TRUE)
    empty <- data.table::data.table(
      meter_id  = character(0),
      date_time = as.POSIXct(character(0), tz = "America/Denver"),
      usage     = numeric(0),
      temp_c    = numeric(0),
      precip_mm = numeric(0),
      lat       = numeric(0),
      lon       = numeric(0)
    )
    arrow::write_parquet(empty, out_path)
    return(out_path)
  }

  # ── 3. ASOF JOIN inside ephemeral in-memory DuckDB ────────────────────────
  #    Matches each usage timestamp to the nearest PRIOR weather timestamp.
  #    In-memory DuckDB: no dbdir → private to this worker → parallel-safe.
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(
    {
      duckdb::duckdb_unregister(con, "usage_tbl")
      duckdb::duckdb_unregister(con, "weather_tbl")
      DBI::dbDisconnect(con, shutdown = TRUE)
    },
    add = TRUE
  )

  duckdb::duckdb_register(con, "usage_tbl", meter_usage)
  duckdb::duckdb_register(con, "weather_tbl", weather_dt)

  sql <- "
    SELECT
      u.meter_id,
      u.date_time,
      u.usage,
      w.temp_c,
      w.precip_mm,
      w.snow_cm,
      u.lat,
      u.lon
    FROM usage_tbl u
    ASOF JOIN weather_tbl w
      ON u.date_time >= w.date_time
    ORDER BY u.date_time
  "

  result <- DBI::dbGetQuery(con, sql)
  data.table::setDT(result)

  # Restore timezone — some DuckDB driver versions strip it on collect
  if (!inherits(result$date_time, "POSIXct")) {
    result[, date_time := as.POSIXct(date_time, tz = "America/Denver")]
  } else {
    data.table::setattr(result$date_time, "tzone", "America/Denver")
  }

  # ── 4. Write Parquet — then free memory ───────────────────────────────────
  #    arrow::write_parquet() accepts a data.frame / data.table directly.
  #    Snappy compression is the default; fast and broadly compatible.
  dir.create(parquet_dir, showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(result, out_path)
  rm(result, weather_dt, meter_usage)
  gc(verbose = FALSE)

  # ── 5. Return path — targets tracks this file on disk ─────────────────────
  out_path
}


# -----------------------------------------------------------------------------
# 8. Register DuckDB VIEW over Parquet directory
# -----------------------------------------------------------------------------

#' Create (or replace) a DuckDB VIEW over the per-meter Parquet directory
#'
#' Called once after all \code{meter_weather} branches complete.  The VIEW
#' definition uses DuckDB's direct glob-path syntax:
#' \preformatted{
#'   SELECT * FROM 'data/parquet/*.parquet'
#' }
#' DuckDB resolves the glob at query time — files are read lazily, nothing
#' is loaded into DuckDB storage or R memory until a downstream target calls
#' \code{dplyr::collect()} or executes SQL against the view.
#'
#' WHY A VIEW OVER PARQUET (vs. appending to a DuckDB table):
#' \itemize{
#'   \item \strong{Lazy reads}: DuckDB scans only the row groups needed to
#'     satisfy a query's filters (predicate pushdown into Parquet).
#'   \item \strong{No duplication}: the Parquet files ARE the storage;
#'     the VIEW is a zero-cost SQL alias.
#'   \item \strong{Auto-discovery}: new meter files land in the directory
#'     and are visible to the next query without re-registering.
#'   \item \strong{No write contention}: parallel branches wrote independent
#'     files; this target simply names them collectively.
#' }
#'
#' @param con          A live DuckDB DBI connection (the shared \code{con} target).
#' @param parquet_dir  Directory containing per-meter Parquet files.
#'   Default: \code{"data/parquet"}.
#' @param view_name    Name of the DuckDB VIEW to create / replace.
#'   Default: \code{"climate_data"}.
#' @param meter_weather_paths Character vector of Parquet paths from the
#'   \code{meter_weather} branching target — used as a sentinel so targets
#'   enforces that all branches finish before this target runs.
#' @return A \code{dplyr::tbl()} lazy reference to the \code{climate_data}
#'   view.  Downstream targets can pipe dplyr verbs and call
#'   \code{dplyr::collect()} only when they actually need rows in R.
#' @export
register_climate_view <- function(db_path,
                                  parquet_dir = "data/parquet",
                                  view_name = "climate_data",
                                  meter_weather_paths = NULL) {
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = db_path)
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

  glob <- file.path(parquet_dir, "*.parquet")

  sql <- sprintf(
    "CREATE OR REPLACE VIEW %s AS SELECT * FROM '%s';",
    DBI::dbQuoteIdentifier(con, view_name),
    glob
  )
  DBI::dbExecute(con, sql)

  message(sprintf(
    "[register_climate_view] VIEW '%s' \u2192 lazy scan of '%s'",
    view_name, glob
  ))

  # Return the view name string — serialisable
  view_name
}


# -----------------------------------------------------------------------------
# 9. Validation summary across all meters
# -----------------------------------------------------------------------------

#' Compute a per-meter summary table from the Parquet-backed DuckDB VIEW
#'
#' Runs a single \code{GROUP BY meter_id} query entirely inside DuckDB against
#' the \code{climate_data} VIEW (which itself scans \code{data/parquet/*.parquet}
#' lazily).  Only the resulting summary — one row per meter — is ever
#' materialised into R memory.
#'
#' Metrics computed per meter:
#' \itemize{
#'   \item \code{n_obs}              — total row count
#'   \item \code{pct_missing_temp}   — percentage of rows where \code{temp_c} is NULL
#'   \item \code{pct_missing_precip} — percentage of rows where \code{precip_mm} is NULL
#'   \item \code{min_usage} / \code{max_usage} — range of the usage column
#'   \item \code{date_start} / \code{date_end}  — observation window per meter
#' }
#'
#' The summary is written to \code{data/parquet/climate_summary.parquet} so it
#' is persisted on disk and targets can track it with \code{format = "file"}.
#'
#' @param con         A live DuckDB DBI connection (the shared \code{con} target).
#' @param climate_view A \code{dplyr::tbl()} lazy reference — passed as a
#'   sentinel so targets enforces that the VIEW exists before this runs.
#'   The object is not used in the function body; DuckDB resolves the view
#'   name directly in SQL.
#' @param view_name   Name of the registered DuckDB VIEW.
#'   Default: \code{"climate_data"}.
#' @param out_path    Destination for the summary Parquet file.
#'   Default: \code{"data/parquet/climate_summary.parquet"}.
#' @return The \code{out_path} string — enables \code{format = "file"} in the
#'   targets target so targets re-runs only when the file changes.
#' @export
summarize_climate_data <- function(db_path,
                                   climate_view = NULL,
                                   view_name = "climate_data",
                                   out_path = "data/parquet/climate_summary.parquet") {
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = db_path, read_only = TRUE)
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

  sql <- sprintf(
    "
    SELECT
      meter_id,
      COUNT(*)                                                  AS n_obs,
      ROUND(
        100.0 * SUM(CASE WHEN temp_c    IS NULL THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0), 2)                         AS pct_missing_temp,
      ROUND(
        100.0 * SUM(CASE WHEN precip_mm IS NULL THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0), 2)                         AS pct_missing_precip,
      ROUND(
        100.0 * SUM(CASE WHEN snow_cm IS NULL THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0), 2)                         AS pct_missing_snow,
      MIN(usage)                                                AS min_usage,
      MAX(usage)                                                AS max_usage,
      MIN(date_time)::DATE                                      AS date_start,
      MAX(date_time)::DATE                                      AS date_end
    FROM %s
    GROUP BY meter_id
    ORDER BY meter_id
    ",
    DBI::dbQuoteIdentifier(con, view_name)
  )

  summary_dt <- DBI::dbGetQuery(con, sql)
  data.table::setDT(summary_dt)

  dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(summary_dt, out_path)

  message(sprintf(
    "[summarize_climate_data] %d meters summarised \u2192 %s",
    nrow(summary_dt), out_path
  ))

  rm(summary_dt)
  out_path
}


# =============================================================================
# 10. Query helpers for downstream modeling
# =============================================================================
# All three functions follow the same contract:
#
#   • They accept `con` (the shared DuckDB connection) and query against the
#     `climate_data` VIEW registered by register_climate_view().
#
#   • They return a LAZY dplyr::tbl() by default — no data leaves DuckDB
#     until the caller explicitly calls dplyr::collect().
#
#   • Set `.collect = TRUE` only when you need a data.frame/data.table in R.
#     For most modelling workflows, keep the result lazy and pipe further
#     dplyr verbs (filter, mutate, summarise) before collecting.
#
#   • All filtering, ordering, and projection happen inside DuckDB's C++
#     engine via dplyr-to-SQL translation — only matching rows hit R memory.
#
# WHY LAZY BY DEFAULT:
#   Downstream targets often chain additional filters before materialising.
#   Returning a tbl() lets callers compose queries without loading intermediate
#   results. This is especially important when the full dataset is large:
#   get_panel_subset() might be called with 500 meters × 2 years but the
#   downstream model needs only 10 columns and 3 months — all resolved in SQL.


#' Retrieve all data for a single meter
#'
#' @param con       A live DuckDB DBI connection.
#' @param meter_id  Character scalar. The meter ID to retrieve.
#' @param view_name Name of the DuckDB VIEW. Default: \code{"climate_data"}.
#' @param .collect  Logical. If \code{TRUE}, call \code{dplyr::collect()} and
#'   return a \code{data.table}. Default: \code{FALSE} (lazy tbl).
#' @return A lazy \code{dplyr::tbl()} or a collected \code{data.table}.
#' @examples
#' \dontrun{
#' # Lazy — compose further before pulling into R:
#' tbl <- get_meter_data(con, "0004100169")
#' tbl |>
#'   dplyr::filter(temp_c > 30) |>
#'   dplyr::collect()
#'
#' # Eager — pull immediately:
#' dt <- get_meter_data(con, "0004100169", .collect = TRUE)
#' }
#' @export
get_meter_data <- function(con,
                           meter_id,
                           view_name = "climate_data",
                           .collect = FALSE) {
  tbl <- dplyr::tbl(con, view_name) |>
    dplyr::filter(meter_id == !!meter_id) |>
    dplyr::arrange(date_time)

  if (.collect) {
    result <- dplyr::collect(tbl)
    data.table::setDT(result)
    return(result)
  }

  tbl
}


#' Retrieve all meters within a datetime window
#'
#' Returns every meter's data where \code{date_time} falls within
#' [\code{start_datetime}, \code{end_datetime}], inclusive.
#'
#' @param con            A live DuckDB DBI connection.
#' @param start_datetime POSIXct or character (ISO 8601). Window start.
#' @param end_datetime   POSIXct or character (ISO 8601). Window end.
#' @param view_name      Name of the DuckDB VIEW. Default: \code{"climate_data"}.
#' @param .collect       Logical. Collect into \code{data.table}? Default \code{FALSE}.
#' @return A lazy \code{dplyr::tbl()} or a collected \code{data.table}.
#' @examples
#' \dontrun{
#' tbl <- get_time_range(con,
#'   start_datetime = as.POSIXct("2023-06-01", tz = "America/Denver"),
#'   end_datetime   = as.POSIXct("2023-08-31", tz = "America/Denver")
#' )
#' tbl |>
#'   dplyr::count(meter_id) |>
#'   dplyr::collect()
#' }
#' @export
get_time_range <- function(con,
                           start_datetime,
                           end_datetime,
                           view_name = "climate_data",
                           .collect = FALSE) {
  # Coerce character input to POSIXct so DuckDB receives a typed timestamp
  if (is.character(start_datetime)) {
    start_datetime <- as.POSIXct(start_datetime, tz = "America/Denver")
  }
  if (is.character(end_datetime)) {
    end_datetime <- as.POSIXct(end_datetime, tz = "America/Denver")
  }

  tbl <- dplyr::tbl(con, view_name) |>
    dplyr::filter(
      date_time >= !!start_datetime,
      date_time <= !!end_datetime
    ) |>
    dplyr::arrange(meter_id, date_time)

  if (.collect) {
    result <- dplyr::collect(tbl)
    data.table::setDT(result)
    return(result)
  }

  tbl
}


#' Retrieve a panel subset: specific meters within a datetime window
#'
#' Combines a meter whitelist with a time filter.  Useful for preparing
#' modelling datasets where only a cohort of meters is needed.
#'
#' @param con            A live DuckDB DBI connection.
#' @param meter_ids      Character vector of meter IDs to include.
#' @param start_datetime POSIXct or character (ISO 8601). Window start.
#' @param end_datetime   POSIXct or character (ISO 8601). Window end.
#' @param view_name      Name of the DuckDB VIEW. Default: \code{"climate_data"}.
#' @param .collect       Logical. Collect into \code{data.table}? Default \code{FALSE}.
#' @return A lazy \code{dplyr::tbl()} or a collected \code{data.table}.
#' @examples
#' \dontrun{
#' ids <- c("0004100169", "0004100170", "0004100205")
#' panel <- get_panel_subset(
#'   con,
#'   meter_ids      = ids,
#'   start_datetime = "2023-01-01",
#'   end_datetime   = "2023-12-31",
#'   .collect       = TRUE
#' )
#' }
#' @export
get_panel_subset <- function(con,
                             meter_ids,
                             start_datetime,
                             end_datetime,
                             view_name = "climate_data",
                             .collect = FALSE) {
  if (is.character(start_datetime)) {
    start_datetime <- as.POSIXct(start_datetime, tz = "America/Denver")
  }
  if (is.character(end_datetime)) {
    end_datetime <- as.POSIXct(end_datetime, tz = "America/Denver")
  }

  tbl <- dplyr::tbl(con, view_name) |>
    # DuckDB translates %in% to SQL IN (...) — efficient for short lists.
    # For very large meter_id vectors (>10k), consider writing them to a
    # temp table and joining instead.
    dplyr::filter(
      meter_id %in% !!meter_ids,
      date_time >= !!start_datetime,
      date_time <= !!end_datetime
    ) |>
    dplyr::arrange(meter_id, date_time)

  if (.collect) {
    result <- dplyr::collect(tbl)
    data.table::setDT(result)
    return(result)
  }

  tbl
}
