# Guide: Scenario-Based Sampling Implementation

This guide provides instructions for updating `scripts/05_generate_samples.py` to move beyond dummy data and implement realistic scenario-based synthetic data generation.

## 1. Feature Mapping Reference
The model was trained on conditioning vectors created in `scripts/02_make_windows.R`. Any generated samples must match these indices exactly.

### Indoor Conditioning (`c_in`) - Dimension: 9
| Index | Feature | Description |
|---|---|---|
| 0 | `month_sin` | `sin(2 * pi * month / 12)` |
| 1 | `month_cos` | `cos(2 * pi * month / 12)` |
| 2 | `dow_sin` | `sin(2 * pi * day_of_week / 7)` |
| 3 | `dow_cos` | `cos(2 * pi * day_of_week / 7)` |
| 4 | `hour_sin` | `sin(2 * pi * hour / 24)` |
| 5 | `hour_cos` | `cos(2 * pi * hour / 24)` |
| 6 | `usage_1h` | Lagged usage (normalized) |
| 7 | `usage_24h` | Lagged usage (normalized) |
| 8 | `usage_48h` | Lagged usage (normalized) |

### Outdoor Conditioning (`c_out`) - Dimension: 10
| Index | Feature | Description |
|---|---|---|
| 0 | `temp_c` | Current Temperature in Celsius |
| 1 | `precip_mm` | Current Precipitation in mm |
| 2 | `snow_cm` | Current Snowfall in cm |
| 3 | `temp_1h` | Lagged Temperature (1h) |
| 4 | `temp_24h` | Lagged Temperature (24h) |
| 5 | `temp_48h` | Lagged Temperature (48h) |
| 6 | `snow_flag` | Binary: 1 if snowing, else 0 |
| 7 | `precip_3d` | Rolling 3-day precipitation sum |
| 8 | `snow_24h` | Rolling 24h snowfall sum |
| 9 | `gdd_7d` | Rolling 7-day Growing Degree Days (base 10C) |

## 2. Implementation Steps for Coding Assistant

1. **Create Parameter Helpers**: Add a helper function to `05_generate_samples.py` that takes a `datetime` and a `weather_dict` and returns the properly encoded `c_in` and `c_out` tensors.
2. **Handle Time-Varying Columns**: Note that `hour_sin/cos` varies across the 96 sequence steps, whereas `month` and `dow` typically stay constant for a daily window. The tensors are `[Batch, Feature, 96]`.
3. **Scaler Integration**: Ensure that the generated features are scaled using the same means/standard deviations used during training.
4. **Sample Call**:
   ```python
   # Example: Simulate a Monday in January at 0 degrees Celsius
   c_in, c_out = encode_scenario(month=1, dow=0, temp=0.0)
   samples = generate_samples_with_condition(c_in, c_out)
   ```
