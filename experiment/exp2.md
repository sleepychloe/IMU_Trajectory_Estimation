
 * [Experiment 2](#exp-2) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-2-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Method](#exp-2-method) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Accelerometer Correction](#exp-2-method-acc-correction) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Detect Quasi-static](#exp-2-method-quasi-static) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Suggest Tau and K](#exp-2-method-tau-k) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Suggest Gate Sigma](#exp-2-method-sigma) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Results](#exp-2-res) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 01 — 5 min](#exp-2-res-data-01) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 02 — 9 min](#exp-2-res-data-02) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 03 — 13 min](#exp-2-res-data-03) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 04 — 96 min](#exp-2-res-data-04) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Cross-dataset Summary](#exp-2-data-sum) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Conclusion](#exp-2-conclusion) <br>


<br>
<br>

## Experiment 2  — Gyro + Accelerometer <a name="exp-2"></a>

### Goal <a name="exp-2-goal"></a>

This experiment evaluates how accelerometer-based gravity correction improves gyro-only orientation,<br>
and when gating is necessary to avoid injecting wrong tilt updates during dynamic motion.<br>

<br>

Four runs are compared (same dataset, same trimmed start):<br>

- [exp 1-2] Gyro-only (baseline, after stabilization trimming)
- [exp 2-1] Gyro + Acc, without gating
- [exp 2-2] Gyro + Acc with Acc gating
- [exp 2-3] Gyro + Acc with Gyro/Acc gating

<br>

Key hypothesis:<br>
The correction term counteracts accumulated gyro bias<br>
so the gap `|err(gyro-only) − err(gyro+acc)|` tends to increase over time.<br>

<br>
<br>
<br>
<br>

### Method <a name="exp-2-method"></a>

### Accelerometer Correction <a name="exp-2-method-acc-correction"></a>

At each step:<br>

- propagate with gyro  
- predict gravity direction in body frame  
- compute error axis `e = g_pred × a_unit`  
- apply a small-angle correction quaternion with effective gain `K_eff = K * weight_acc * weight_gyro`, where `K = dt_median / tau`

** `weight_acc`: Accel magnitude residual-based confidence (approaches to 1 when `| ||a|| - g0 |` is smaller)<br>
** `weight_gyro`: Gyro norm-based confidence (approaches to 1 when `||w||` is smaller)<br>

<br>
<br>
<br>

### Detect Quasi-static <a name="exp-2-method-quasi-static"></a>

A quasi-static segment is automatically detected to evaluate gravity stability without being polluted by motion.<br>

<br>

Quasi-static here means low angular rate and near-gravity accel magnitude, not necessarily perfectly motionless.<br>

<br>

- Compute `||w||` and the accelerometer magnitude residual `| ||a|| - g0 |`
- `quasi_static = (||w|| < w_thr) & (| ||a|| - g0 | < a_thr)`, if thresholds are not provided, they are set to a low percentile
- Apply a majority-vote smoothing window to debounce short spikes
- Keep the longest continuous True run, and reject if it is shorter than `min_duration_s`

<br>

##### [Implementation]

```py
# autotune.py

def smooth_bool(mask: BoolBatch, win: int = 5) -> BoolBatch:
        . . .
        m: Int8Batch = as_int8_batch(mask.astype(np.int8, copy=False))
        kernel: Int32Batch = np.ones(win, dtype=np.int32)

        s: Int64Batch = np.convolve(m, kernel, mode="same")
        res: BoolBatch = (s >= (win//2 + 1))
        return as_bool_batch(res)

def largest_true_run(mask: BoolBatch) -> tuple[int, int, int] | None:
        . . .
        m: Int8Batch = as_int8_batch(mask.astype(np.int8, copy=False))
        diff: Int8Batch = as_int8_batch(np.diff(m).astype(np.int8, copy=False))
        start: Int64Batch = np.where(diff == 1)[0] + 1
        end: Int64Batch = np.where(diff == -1)[0] + 1

        if bool(mask[0]):
                start = np.r_[np.array([0], dtype=start.dtype), start]
        if bool(mask[-1]):
                end = np.r_[end, np.array([mask.size], dtype=end.dtype)]
        . . .
        return int(start[i]), int(end[i]), int(length[i])

def quasi_static_detector(w: Vec3Batch, a: Vec3Batch, dt: ScalarBatch, g0: float,
                          w_thr: float, a_thr: float,
                          min_duration_s: float, smooth_win: int,
                          ) -> tuple[int, int, int]:
	. . .
        raw_mask: BoolBatch = (w_norm < w_thr) & (acc_resid < a_thr)
        quasi_static_mask: BoolBatch = smooth_bool(raw_mask, win=smooth_win)
	. . .
        best_quasi_static: tuple[int, int, int] = largest_true_run(quasi_static_mask)
	. . .
        return best_quasi_static
```

<br>
<br>
<br>

### Suggest Tau and K <a name="exp-2-method-tau-k"></a>

The `tau` that makes the estimated gravity direction most stable during the quasi-static segment is selected, then converted to the discrete gain `K = median(dt) / tau`.<br>

For each `tau` candidate:<br>

- Run the pipeline and collect the estimated gravity direction in the body frame
- Inside the quasi-static segment, normalize `g_body_est` and measure stability by the mean angular deviation from the mean direction
- Select the `tau` with the lowest mean angular deviation score

##### [Implementation]

```py
# autotune.py

def choose_tau_from_quasi_static(dt: ScalarBatch, runner_func: Callable[[float], tuple[Any, ...]],
                                 best_quasi_static: tuple[int, int, int] | None = None,
                                 tau_candidates: tuple[float, ...] = (0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3),
                                 runner_kwargs: dict[str, Any] = None,
                                 ) -> tuple[list[dict[str, Any]], float, float]:
        . . .
        for tau in tau_candidates:
                K = float(dt_median / tau)
                _, extra = runner_func(K=K, **runner_kwargs)
                g_body_est, _, _, _ = extra

                gb = g_body_est[s:e]
                . . .
                mean_dir: Vec3 = as_vec3(np.mean(gb_unit, axis=0))
                . . .
                dot: ScalarBatch = np.clip(gb_unit @ mean_dir, -1, 1)
                ang: ScalarBatch = np.arccos(dot)
                score: float = float(np.mean(ang))
		. . .
                tau_table.append({ . . . })
        tau_table.sort(key=lambda d: d["quasi_static_score_mean_angle(rad)"])
        . . .
        return tau_table, best_tau, best_K
```

<br>
<br>
<br>

### Suggest Gate Sigma <a name="exp-2-method-sigma"></a>

Gate sigmas are suggested from robust percentiles of the data.<br>
If quasi-static exists, statistics are computed on that segment, otherwise fallback to the full sequence.<br>

<br>

```
For gyro gating
	gyro_sigma = max(sigma_floor, percentile(||w||, p_gyro)) (or inf if disabled)

For accel gating:
	acc_sigma = max(sigma_floor, percentile(| ||a|| - g0 |, p_acc)) (or inf if disabled)
```
<br>

These sigma values set the scale for how aggressively measurement can be trusted/ignored when motion or acceleration residuals are high.<br>

<br>
<br>
<br>
<br>

### Results <a name="exp-2-res"></a>

Each plot compares:<br>

- blue: exp 1-2
- orange: exp 2-1
- green: exp 2-2
- red: exp 2-3

<br>
<br>

### Dataset 01 — 5 min <a name="exp-2-res-data-01"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data01_exp2_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data01_exp2_02.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data01_exp2_03.png" width="952" height="471">

#### [Chosen parameters]

- quasi-static: (2523, 3540, 1017)
- suggested σ_gyro: 0.4479197
- suggested σ_acc : 2.9085688

<br>

|         |  tau  |         K         |       σ_gyro      |       σ_acc       |
|:-------:|------:|------------------:|------------------:|------------------:|
| exp 2-1 |  3.0  | 0.003332926432292 | inf (not applied) | inf (not applied) |
| exp 2-2 |  3.0  | 0.003332926432292 | inf (not applied) |         2.9085688 |
| exp 2-3 |  3.0  | 0.003332926432292 |         0.4479197 |         2.9085688 |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-01 01:43:43.253

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301

[END] 2026-03-01 01:43:43.920
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |
| exp 2-1 | <ul><li>0.34141 rad</li><li>19.56163 deg</li></ul> | <ul><li>0.51488 rad</li><li>29.50019 deg</li></ul> |
| exp 2-2 | <ul><li>0.34020 rad</li><li>19.49187 deg</li></ul> | <ul><li>0.51364 rad</li><li>29.42940 deg</li></ul> |
| exp 2-3 | <ul><li>0.33909 rad</li><li>19.42834 deg</li></ul> | <ul><li>0.51240 rad</li><li>29.35820 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-01 01:43:44.746
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0028253656890090757 1.4041137225284805 0.3414149857047015 0.5148755033035524

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.16188152956129193 80.44979025728507 19.56163774339873 29.500193313967625

[END] 2026-03-01 01:44:23.195



[START] 2026-03-01 01:44:23.208
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) angle error in rad — min/max/mean/p90
0.0028007213045730187 1.4022353056038215 0.3401972619552348 0.5136400057527237

[exp 2-2] Gyro+Acc+Gating(Acc) angle error in deg — min/max/mean/p90
0.16046951034440796 80.34216489533617 19.491867311941444 29.429404518706395

[END] 2026-03-01 01:45:17.135



[START] 2026-03-01 01:45:17.149
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in rad — min/max/mean/p90
0.002782597463724189 1.4018972772745735 0.33908843712552167 0.5123973023219879

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in deg — min/max/mean/p90
0.15943109075520323 80.32279729871439 19.42833632897957 29.358202856938803

[END] 2026-03-01 01:46:33.207
```

<br>

#### [Observation]

- Gating provides a small but consistent improvement
- The error curve stays below gyro-only more clearly as time progresses,
  indicating continuous correction against drift/bias

<br>
<br>
<br>

### Dataset 02 — 9 min <a name="exp-2-res-data-02"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data02_exp2_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data02_exp2_02.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data02_exp2_03.png" width="952" height="471">

#### [Chosen parameters]

- quasi-static: (0, 1635, 1635)
- suggested σ_gyro: 0.6304230
- suggested σ_acc : 2.5176967

<br>

|         |  tau  |         K         |       σ_gyro      |       σ_acc       |
|:-------:|------:|------------------:|------------------:|------------------:|
| exp 2-1 |  0.7  | 0.014283970424133 | inf (not applied) | inf (not applied) |
| exp 2-2 |  0.5  | 0.019997558593786 | inf (not applied) | inf (not applied) |
| exp 2-3 |  0.3  | 0.033329264322977 | inf (not applied) | inf (not applied) |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-02-28 21:10:05.307

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874

[END] 2026-02-28 21:10:05.940
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |
| exp 2-1 | <ul><li>0.23149 rad</li><li>13.26360 deg</li></ul> | <ul><li>0.37960 rad</li><li>21.74953 deg</li></ul> |
| exp 2-2 | <ul><li>0.21107 rad</li><li>12.09349 deg</li></ul> | <ul><li>0.36347 rad</li><li>20.82510 deg</li></ul> |
| exp 2-3 | <ul><li>0.16233 rad</li><li>9.30065 deg</li></ul> | <ul><li>0.32539 rad</li><li>18.64351 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-02-28 21:10:07.209
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0065255237080641786 0.6783031599782409 0.231493439286664 0.3796008242820292

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.3738849675846366 38.86390829714029 13.26359705609381 21.74952513104745

[END] 2026-02-28 21:11:11.830



[START] 2026-02-28 21:11:11.841
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) angle error in rad — min/max/mean/p90
0.005011736195100535 0.6670413009297944 0.21107127007672222 0.3634665404316075

[exp 2-2] Gyro+Acc+Gating(Acc) angle error in deg — min/max/mean/p90
0.2871513320122144 38.218651304193095 12.093492951862126 20.825098760952205

[END] 2026-02-28 21:12:47.383



[START] 2026-02-28 21:12:47.397
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in rad — min/max/mean/p90
0.0026744437984058097 0.6576030554091132 0.16232690143638706 0.32539059191857556

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in deg — min/max/mean/p90
0.15323434219358967 37.67787966984981 9.30064635374108 18.643507610198053

[END] 2026-02-28 21:14:59.631
```

<br>

#### [Observation]

- Large gain from adding accel correction compare to exp 1-2(gyro-only)
- In this dataset, the optimizer effectively chooses `no gating`, because accel looks reliable enough

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-2-res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data03_exp2_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data03_exp2_02.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data03_exp2_03.png" width="952" height="471">

#### [Chosen parameters]

- quasi-static: (41487, 43669, 2182)
- suggested σ_gyro: 0.4822681
- suggested σ_acc : 0.6755996

<br>

|         |  tau  |         K         |       σ_gyro      |       σ_acc       |
|:-------:|------:|------------------:|------------------:|------------------:|
| exp 2-1 |  0.3  | 0.033329264322977 | inf (not applied) | inf (not applied) |
| exp 2-2 |  0.2  | 0.049993896484466 | inf (not applied) |         0.6755996 |
| exp 2-3 |  0.2  | 0.049993896484466 | inf (not applied) |         0.6755996 |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-02-28 21:02:42.102

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786

[END] 2026-02-28 21:02:42.680
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| exp 2-1 | <ul><li>0.40981 rad</li><li>23.48033 deg</li></ul> | <ul><li>0.75477 rad</li><li>43.24533 deg</li></ul> |
| exp 2-2 | <ul><li>0.37464 rad</li><li>21.46521 deg</li></ul> | <ul><li>0.56815 rad</li><li>32.55285 deg</li></ul> |
| exp 2-3 | <ul><li>0.37464 rad</li><li>21.46521 deg</li></ul> | <ul><li>0.56815 rad</li><li>32.55285 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-02-28 21:02:44.215
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.005370781710236194 1.1276601890042688 0.4098091184475541 0.7547733170837478

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.30772312468258817 64.61016955486933 23.480332893021696 43.24532555798819

[END] 2026-02-28 21:04:14.148



[START] 2026-02-28 21:04:14.158
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) angle error in rad — min/max/mean/p90
0.006049463731097126 1.1110652039347422 0.37463867609401474 0.5681543377919447

[exp 2-2] Gyro+Acc+Gating(Acc) angle error in deg — min/max/mean/p90
0.3466087401093293 63.65934694930284 21.465214982555736 32.55284566752856

[END] 2026-02-28 21:06:25.183



[START] 2026-02-28 21:06:25.195
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in rad — min/max/mean/p90
0.006049463731097126 1.1110652039347422 0.37463867609401474 0.5681543377919447

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in deg — min/max/mean/p90
0.3466087401093293 63.65934694930284 21.465214982555736 32.55284566752856

[END] 2026-02-28 21:09:32.388
```

<br>

#### [Observation]

- This is a clean gating matters case — p90 drops significantly because gating suppresses bad accel updates during dynamic segments
- The reduced-error region aligns with periods where `||a||` deviates from `g0`

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-2-res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data04_exp2_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data04_exp2_02.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data04_exp2_03.png" width="952" height="471">

#### [Chosen parameters]

- quasi-static: (252162, 310194, 58032)
- suggested σ_gyro: 0.2752885
- suggested σ_acc : 0.5070689

<br>

|         |  tau  |         K         |       σ_gyro      |       σ_acc       |
|:-------:|------:|------------------:|------------------:|------------------:|
| exp 2-1 |  3.0  | 0.003352945963646 | inf (not applied) | inf (not applied) |
| exp 2-2 |  3.0  | 0.003352945963646 | inf (not applied) | inf (not applied) |
| exp 2-3 |  3.0  | 0.003352945963646 | inf (not applied) | inf (not applied) |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-02-28 20:10:29.708

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965

[END] 2026-02-28 20:10:30.331
```

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| exp 2-1 | <ul><li>0.85937 rad</li><li>49.23849 deg</li></ul> | <ul><li>2.00162 rad</li><li>114.68462 deg</li></ul> |
| exp 2-2 | <ul><li>0.85937 rad</li><li>49.23849 deg</li></ul> | <ul><li>2.00162 rad</li><li>114.68462 deg</li></ul> |
| exp 2-3 | <ul><li>0.85937 rad</li><li>49.23849 deg</li></ul> | <ul><li>2.00162 rad</li><li>114.68462 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-02-28 20:10:31.958
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.002969310113133571 3.139880414590167 0.8593738181216991 2.001624262321689

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.1701289375480666 179.90189593180372 49.23849280241658 114.68462240201954

[END] 2026-02-28 20:21:32.331



[START] 2026-02-28 20:21:32.344
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) angle error in rad — min/max/mean/p90
0.002969310113133571 3.139880414590167 0.8593738181216991 2.001624262321689

[exp 2-2] Gyro+Acc+Gating(Acc) angle error in deg — min/max/mean/p90
0.1701289375480666 179.90189593180372 49.23849280241658 114.68462240201954

[END] 2026-02-28 20:37:12.914



[START] 2026-02-28 20:37:12.930
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in rad — min/max/mean/p90
0.002969310113133571 3.139880414590167 0.8593738181216991 2.001624262321689

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) angle error in deg — min/max/mean/p90
0.1701289375480666 179.90189593180372 49.23849280241658 114.68462240201954

[END] 2026-02-28 20:59:52.226
```

<br>

#### [Observation]

- Very long, uncontrolled motion introduces large error plateaus/spikes
- Gating is rejected, indicating that `| ||a|| - g0 |` is not a useful reliability proxy for this motion pattern
- Remaining error likely comes from yaw/heading drift or mounting/frame mismatch rather than pure roll/pitch correction quality

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-2-data-sum"></a>

| Dataset | exp 1-2 Mean   | best exp2 Mean | exp 1-2 p90    | best exp2 p90  |
|:--------|---------------:|---------------:|---------------:|---------------:|
| data 01 | 22.40684 deg   | 19.42834 deg   | 32.44667 deg   | 29.35820 deg   |
| data 02 | 21.99183 deg   | 9.30065 deg    | 31.17450 deg   | 18.64351 deg   |
| data 03 | 30.81266 deg   | 21.46521 deg   | 46.56837 deg   | 32.55285 deg   |
| data 04 | 50.88316 deg   | 49.23849 deg   | 116.70718 deg  | 114.68462 deg  |

** best exp2 = minimum mean error among exp 2-1, 2-2, 2-3 (per dataset)<br>

<br>

Across all datasets:<br>

- Adding accelerometer correction consistently reduces drift relative to gyro-only,
  and the improvement gap tends to grow over time
- Gating is dataset-dependent — it helps when linear acceleration frequently violates the gravity assumption, but may be marginal or rejected when accel measurements are already consistent

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-2-conclusion"></a>

Experiment 2 confirms:<br>

1. Accelerometer correction stabilizes roll/pitch by continuously correcting gyro drift
2. Gating acts as a reliability controller that prevents incorrect tilt injections during dynamic motion

Next steps:<br>
- Experiment 3: add magnetometer correction to constrain yaw drift

<br>
<br>
<br>
<br>
