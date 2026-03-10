# Sweep Results

## By Scene

_Which scenes are most challenging? Sorted by scene, then course, then best rotation error._

### Scene → Course → Performance

| Scene              | Course             | Model       |  N | Rot Med (°) | Trans Med (°) | AUC@5 | AUC@15 |
| ------------------ | ------------------ | ----------- | -: | ----------: | ------------: | ----: | -----: |
| button_drywall_on  | circle_along_track | da3_chunked | 70 |        1.77 |          3.18 |  0.38 |   0.70 |
| button_drywall_on  | circle_along_track | streamvggt  | 70 |        3.96 |         16.75 |  0.08 |   0.25 |
| button_drywall_on  | square_along_track | da3_chunked | 70 |        1.93 |          2.42 |  0.41 |   0.71 |
| button_drywall_on  | square_along_track | streamvggt  | 70 |        3.89 |          6.43 |  0.13 |   0.46 |
| flightroom_ssv_exp | circle_along_track | da3_chunked | 70 |        1.64 |          2.56 |  0.49 |   0.79 |
| flightroom_ssv_exp | circle_along_track | streamvggt  | 70 |       24.36 |         28.57 |  0.04 |   0.14 |
| flightroom_ssv_exp | square_along_track | da3_chunked | 70 |        3.50 |          2.50 |  0.33 |   0.65 |
| flightroom_ssv_exp | square_along_track | streamvggt  | 70 |        9.23 |         13.72 |  0.06 |   0.18 |
| packardpark        | circle_along_track | streamvggt  | 70 |        3.40 |         10.02 |  0.11 |   0.35 |
| packardpark        | circle_along_track | da3_chunked | 70 |        4.45 |          4.75 |  0.17 |   0.45 |
| packardpark        | square_along_track | streamvggt  | 70 |        3.34 |         10.35 |  0.10 |   0.32 |
| packardpark        | square_along_track | da3_chunked | 70 |        3.48 |          4.23 |  0.20 |   0.56 |

## By Course

_Which flight courses are most challenging? Sorted by course, then scene, then best rotation error._

### Course → Scene → Performance

| Scene              | Course             | Model       |  N | Rot Med (°) | Trans Med (°) | AUC@5 | AUC@15 |
| ------------------ | ------------------ | ----------- | -: | ----------: | ------------: | ----: | -----: |
| button_drywall_on  | circle_along_track | da3_chunked | 70 |        1.77 |          3.18 |  0.38 |   0.70 |
| button_drywall_on  | circle_along_track | streamvggt  | 70 |        3.96 |         16.75 |  0.08 |   0.25 |
| flightroom_ssv_exp | circle_along_track | da3_chunked | 70 |        1.64 |          2.56 |  0.49 |   0.79 |
| flightroom_ssv_exp | circle_along_track | streamvggt  | 70 |       24.36 |         28.57 |  0.04 |   0.14 |
| packardpark        | circle_along_track | streamvggt  | 70 |        3.40 |         10.02 |  0.11 |   0.35 |
| packardpark        | circle_along_track | da3_chunked | 70 |        4.45 |          4.75 |  0.17 |   0.45 |
| button_drywall_on  | square_along_track | da3_chunked | 70 |        1.93 |          2.42 |  0.41 |   0.71 |
| button_drywall_on  | square_along_track | streamvggt  | 70 |        3.89 |          6.43 |  0.13 |   0.46 |
| flightroom_ssv_exp | square_along_track | da3_chunked | 70 |        3.50 |          2.50 |  0.33 |   0.65 |
| flightroom_ssv_exp | square_along_track | streamvggt  | 70 |        9.23 |         13.72 |  0.06 |   0.18 |
| packardpark        | square_along_track | streamvggt  | 70 |        3.34 |         10.35 |  0.10 |   0.32 |
| packardpark        | square_along_track | da3_chunked | 70 |        3.48 |          4.23 |  0.20 |   0.56 |

## By Model

_Which model performs best overall? Sorted by model, then best rotation error._

### Model → Performance

| Scene              | Course             | Model       |  N | Rot Med (°) | Trans Med (°) | AUC@5 | AUC@15 |
| ------------------ | ------------------ | ----------- | -: | ----------: | ------------: | ----: | -----: |
| flightroom_ssv_exp | circle_along_track | da3_chunked | 70 |        1.64 |          2.56 |  0.49 |   0.79 |
| button_drywall_on  | circle_along_track | da3_chunked | 70 |        1.77 |          3.18 |  0.38 |   0.70 |
| button_drywall_on  | square_along_track | da3_chunked | 70 |        1.93 |          2.42 |  0.41 |   0.71 |
| packardpark        | square_along_track | da3_chunked | 70 |        3.48 |          4.23 |  0.20 |   0.56 |
| flightroom_ssv_exp | square_along_track | da3_chunked | 70 |        3.50 |          2.50 |  0.33 |   0.65 |
| packardpark        | circle_along_track | da3_chunked | 70 |        4.45 |          4.75 |  0.17 |   0.45 |
| packardpark        | square_along_track | streamvggt  | 70 |        3.34 |         10.35 |  0.10 |   0.32 |
| packardpark        | circle_along_track | streamvggt  | 70 |        3.40 |         10.02 |  0.11 |   0.35 |
| button_drywall_on  | square_along_track | streamvggt  | 70 |        3.89 |          6.43 |  0.13 |   0.46 |
| button_drywall_on  | circle_along_track | streamvggt  | 70 |        3.96 |         16.75 |  0.08 |   0.25 |
| flightroom_ssv_exp | square_along_track | streamvggt  | 70 |        9.23 |         13.72 |  0.06 |   0.18 |
| flightroom_ssv_exp | circle_along_track | streamvggt  | 70 |       24.36 |         28.57 |  0.04 |   0.14 |

## Comparison Tables

### Rotation Error — Median (°)

| Scene              | Course             |  N | da3_chunked | streamvggt |
| ------------------ | ------------------ | -: | ----------: | ---------: |
| button_drywall_on  | circle_along_track | 70 |    **1.77** |       3.96 |
| button_drywall_on  | square_along_track | 70 |    **1.93** |       3.89 |
| flightroom_ssv_exp | circle_along_track | 70 |    **1.64** |      24.36 |
| flightroom_ssv_exp | square_along_track | 70 |    **3.50** |       9.23 |
| packardpark        | circle_along_track | 70 |        4.45 |   **3.40** |
| packardpark        | square_along_track | 70 |        3.48 |   **3.34** |

### Translation Error — Median (°)

| Scene              | Course             |  N | da3_chunked | streamvggt |
| ------------------ | ------------------ | -: | ----------: | ---------: |
| button_drywall_on  | circle_along_track | 70 |    **3.18** |      16.75 |
| button_drywall_on  | square_along_track | 70 |    **2.42** |       6.43 |
| flightroom_ssv_exp | circle_along_track | 70 |    **2.56** |      28.57 |
| flightroom_ssv_exp | square_along_track | 70 |    **2.50** |      13.72 |
| packardpark        | circle_along_track | 70 |    **4.75** |      10.02 |
| packardpark        | square_along_track | 70 |    **4.23** |      10.35 |

### AUC @ 5°

| Scene              | Course             |  N | da3_chunked | streamvggt |
| ------------------ | ------------------ | -: | ----------: | ---------: |
| button_drywall_on  | circle_along_track | 70 |    **0.38** |       0.08 |
| button_drywall_on  | square_along_track | 70 |    **0.41** |       0.13 |
| flightroom_ssv_exp | circle_along_track | 70 |    **0.49** |       0.04 |
| flightroom_ssv_exp | square_along_track | 70 |    **0.33** |       0.06 |
| packardpark        | circle_along_track | 70 |    **0.17** |       0.11 |
| packardpark        | square_along_track | 70 |    **0.20** |       0.10 |

## Model Summary

### Model Summary (averaged across scenes/courses)

| Model       | N | Rot Med (°) | Rot Mean (°) | Trans Med (°) | Trans Mean (°) | AUC@5 | AUC@15 |
| ----------- | : | ----------: | -----------: | ------------: | -------------: | ----: | -----: |
| da3_chunked | 6 |        2.80 |         3.42 |          3.27 |           7.17 |  0.33 |   0.64 |
| streamvggt  | 6 |        8.03 |        13.21 |         14.31 |          19.80 |  0.09 |   0.28 |
