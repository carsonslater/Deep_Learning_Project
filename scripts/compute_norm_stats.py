import numpy as np
import json
import duckdb
from pathlib import Path

def compute_normalisation_stats(parquet_dir: str = "data/features", output_path: str = "norm_stats.json"):
    """
    Compute log1p mean/std of NON-ZERO usage values only.
    Zeros are structural (no flow) and must not influence the scale.
    """
    print(f"Connecting to DuckDB and analyzing {parquet_dir}/*.parquet...")
    con = duckdb.connect()

    # Pull only non-zero usage values and raw weather values
    weather_cols = ["temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h", "precip_3d", "snow_24h", "gdd_7d"]
    
    weather_select = ", ".join([f"AVG({c}) AS {c}_mean, STDDEV({c}) AS {c}_std" for c in weather_cols])
    
    query = f"""
        SELECT
            AVG(log_usage)  AS log_mean,
            STDDEV(log_usage) AS log_std,
            {weather_select}
        FROM (
            SELECT usage, LN(usage + 1.0) AS log_usage, {", ".join(weather_cols)}
            FROM read_parquet('{parquet_dir}/*.parquet')
        )
        WHERE usage > 0.0 OR usage = 0.0 -- Include all for weather stats
    """
    try:
        result = con.execute(query).fetchone()
        
        if result[0] is None:
            print("Warning: No data found or parquet files missing.")
            return
        
        stats = {
            "log_mean": float(result[0]),
            "log_std": float(result[1])
        }
        
        # Add weather stats
        res_idx = 2
        for col in weather_cols:
            stats[f"{col}_mean"] = float(result[res_idx]) if result[res_idx] is not None else 0.0
            stats[f"{col}_std"] = float(result[res_idx+1]) if result[res_idx+1] is not None else 1.0
            res_idx += 2
            
        Path(output_path).write_text(json.dumps(stats, indent=2))
        print(f"✅ Normalisation Stats (Usage + {len(weather_cols)} Weather features) saved to {output_path}")
        return stats
    except Exception as e:
        print(f"❌ Error computing stats: {e}")

if __name__ == "__main__":
    compute_normalisation_stats()
