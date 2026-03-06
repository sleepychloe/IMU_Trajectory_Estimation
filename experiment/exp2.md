
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
- [exp 2-2] Gyro + Acc with Acc gating — fixed sigma
- [exp 2-3] Gyro + Acc with Gyro/Acc gating — fixed sigma
- [exp 2-4] Gyro + Acc with Gyro/Acc gating — time-varying sigma

<br>

Key hypothesis:<br>
The correction term counteracts accumulated gyro bias<br>
so the correction is expected to counteract accumulated gyro drift, making improvements more visible over longer sequences.<br>

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
        tau_table.sort(key=lambda d: d["score"])
        . . .
        return tau_table, best_tau, best_K
```

<br>
<br>
<br>

### Suggest Gate Sigma <a name="exp-2-method-sigma"></a>

Gate sigmas control how aggressively measurements are trusted via soft gating weights.<br>
A larger sigma makes gating weaker (more permissive), while a smaller sigma makes gating stricter (more selective).<br>

<br>

#### [Fixed sigma (baseline)]

A fixed sigma_base is suggested from robust percentiles of the data.<br>
If a quasi-static segment exists, statistics are computed on that segment, otherwise, the full sequence is used.<br>

<br>

```py
# autotune.py

def suggest_fixed_gate_sigma(w: Vec3Batch, a: Vec3Batch, m: Vec3Batch, g0: float,
                       p_gyro: int, p_acc: int, p_mag: int, sigma_floor: float,
                       best_quasi_static: tuple[int, int, int] = None
                       ) -> tuple[float, float, float]:
	. . .
        gyro_sigma: float = max(sigma_floor, float(np.percentile(w_norm, p_gyro)))
	. . .
        acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))
	. . .
        return gyro_sigma, acc_sigma, mag_sigma
```
<br>
<br>

#### [Sigma scale sweep]

To compensate for dataset-dependent behavior, a discrete sweep is performed around the suggested base sigma.<br>

<br>

The best scale is selected by minimizing the mean angle error on the evaluation set.<br>
Setting `scale = inf` disables gating (`sigma = inf`).<br>

<br>

```py
# autotune.py

def choose_best_by_sigma_scale(scales: tuple[float, ...],
                               K: float, sigma_base: float, q_ref: QuatBatch,
                               runner_func: Callable[[float], tuple[Any, ...]],
                               sigma_kw: str,
                               fixed_kwargs: dict[str, Any] = None
                               ) -> SweepBest:
        best: SweepBest = None
        for s in scales:
                sigma: float = calc_sigma(sigma_base, s)
		. . .
                if best is None or mean_err < best.mean_err:
                        best = SweepBest(s, sigma, angle_err, mean_err, q_est, extra)
        return best
```

<br>
<br>

#### [Time-varying sigma]

Fixed sigmas can underfit non-stationary motion patterns.<br>
A time-varying sigma schedule is therefore constructed by recomputing percentile-based sigmas over a sliding window.<br>

<br>

At regular update steps, statistics are computed on the recent window and smoothed with EMA.<br>

```py
# autotune.py

def suggest_timevarying_gate_sigma(w: Vec3Batch, a: Vec3Batch, m: Vec3Batch,
                                   dt: ScalarBatch, g0: float,
                                   p_gyro: int, p_acc: int, p_mag: int, sigma_floor: float,
                                   win_s: float, update_s: float, ema_alpha: float
                                   ) -> tuple[ScalarBatch, ScalarBatch, ScalarBatch]:
        . . .
        window_size: int = max(1, int(np.ceil(win_s / max(dt_median, EPS))))
        update_period: int = max(1, int(np.ceil(update_s / max(dt_median, EPS))))
        . . .
        gyro_sigma = max(sigma_floor, float(np.percentile(w_norm[:min(len(dt), window_size)], p_gyro)))
	. . .
        acc_sigma = max(sigma_floor, float(np.percentile(acc_resid[:min(len(dt), window_size)], p_acc)))
        . . .
        for i in range(len(dt)):
                if i % update_period == 0:
                        low: int = max(0, i - window_size)
                        high: int = i + 1

                        . . .
			gyro_sigma = (1 - ema_alpha) * gyro_sigma + ema_alpha * gyro_tmp
                        . . .
			acc_sigma = (1 - ema_alpha) * acc_sigma + ema_alpha * acc_tmp
                        . . .
                batch_gyro_sigma[i] = gyro_sigma
                batch_acc_sigma[i] = acc_sigma
                . . .
        return batch_gyro_sigma, batch_acc_sigma, batch_mag_sigma
```

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
- purple: exp 2-4

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

|         |  tau  |         K         |       σ_gyro       |        σ_acc       |
|:-------:|------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  3.0  | 0.003332926432292 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  3.0  | 0.003332926432292 |  inf (not applied) |          2.9085688 |
| exp 2-3 |  3.0  | 0.003332926432292 |          0.4479197 |          2.9085688 |
| exp 2-4 |  3.0  | 0.003332926432292 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-03 09:24:46.022

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301

[END] 2026-03-03 09:24:46.672
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |
| exp 2-1 | <ul><li>0.34141 rad</li><li>19.56164 deg</li></ul> | <ul><li>0.51488 rad</li><li>29.50019 deg</li></ul> |
| exp 2-2 | <ul><li>0.34020 rad</li><li>19.49187 deg</li></ul> | <ul><li>0.51364 rad</li><li>29.42940 deg</li></ul> |
| exp 2-3 | <ul><li>0.33909 rad</li><li>19.42834 deg</li></ul> | <ul><li>0.51240 rad</li><li>29.35820 deg</li></ul> |
| exp 2-4 | <ul><li>0.33846 rad</li><li>19.39258 deg</li></ul> | <ul><li>0.51209 rad</li><li>29.34040 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-03 09:25:18.174
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0028253656890090757 1.4041137225284805 0.3414149857047015 0.5148755033035524

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.16188152956129193 80.44979025728507 19.56163774339873 29.500193313967625

[END] 2026-03-03 09:25:23.091



[START] 2026-03-03 09:25:57.228
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0028007213045730187 1.4022353056038215 0.3401972619552348 0.5136400057527237

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.16046951034440796 80.34216489533617 19.491867311941444 29.429404518706395

[END] 2026-03-03 09:26:15.764



[START] 2026-03-03 09:26:15.775
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.002782597463724189 1.4018972772745735 0.33908843712552167 0.5123973023219879

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.15943109075520323 80.32279729871439 19.42833632897957 29.358202856938803

[END] 2026-03-03 09:26:35.554



[START] 2026-03-03 09:26:35.565
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.0027410136353430113 1.4015576507916865 0.3384643527738604 0.5120866094986346

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.1570485128929654 80.3033381346341 19.39257892956922 29.340401469435655

[END] 2026-03-03 09:26:40.807



[START] 2026-03-03 09:26:42.126

best: exp2-4

[END] 2026-03-03 09:26:42.127
```

<br>

#### [Secondary validation — Gravity & Linear Accel]

Estimated gravity and linear acceleration are compared against reference signals as a sanity check.<br>

<br>

Gravity direction error remains low (mean/p90 ≈ 1.05° / 1.59°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 8.22° / 18.08°).<br>

<br>

These checks support that Experiment 2 is physically consistent and the orientation error comparisons are meaningful.<br>

<br>

```
[START] 2026-03-03 09:26:47.193

[Gravity]
RMSE norm: 0.1989648675156791

Gravity est/ref angle error in rad — min/max/mean/p90
0.000791771032443189 0.08743225325717106 0.018255781216350436 0.027724722306785583

Gravity est/ref angle error in deg — min/max/mean/p90
0.04536513849971051 5.009499104954847 1.0459792154110845 1.588509576351022


[Linear accel]
RMSE norm: 0.7348191607050267

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00029173436024331384 2.9508899078869413 0.1434996181112735 0.31555470830185856

Linear accel est/ref angle error in deg — min/max/mean/p90
0.01671514758089104 169.07353752967 8.221922479515042 18.079952991178295
. . .
[END] 2026-03-03 09:26:47.626
```
<br>

#### [Observation]

- Adding accelerometer correction reduces drift relative to gyro-only, and the gap stays visible as time progresses
- Fixed gating (exp 2-2/2-3) provides a small improvement over ungated correction, indicating that `| ||a|| - g0 |` is a meaningful reliability proxy in this dataset
- Time-varying sigma (exp 2-4) achieves the best overall score, suggesting that a single global sigma underfits changing motion regimes
- The improvement is small but consistent across mean and p90 metrics, suggesting slightly better suppression of occasional bad accel updates

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

|         |  tau  |         K         |       σ_gyro       |        σ_acc       |
|:-------:|------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  0.7  | 0.014283970424133 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  0.3  | 0.033329264322977 |  inf (not applied) |  inf (not applied) |
| exp 2-3 |  0.3  | 0.033329264322977 |  inf (not applied) |  inf (not applied) |
| exp 2-4 |  0.3  | 0.033329264322977 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-03 09:27:09.867

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874

[END] 2026-03-03 09:27:10.491
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |
| exp 2-1 | <ul><li>0.23149 rad</li><li>13.26360 deg</li></ul> | <ul><li>0.37960 rad</li><li>21.74953 deg</li></ul> |
| exp 2-2 | <ul><li>0.16233 rad</li><li>9.30065 deg</li></ul> | <ul><li>0.32539 rad</li><li>18.64351 deg</li></ul> |
| exp 2-3 | <ul><li>0.16233 rad</li><li>9.30065 deg</li></ul> | <ul><li>0.32539 rad</li><li>18.64351 deg</li></ul> |
| exp 2-4 | <ul><li>0.30322 rad</li><li>17.37319 deg</li></ul> | <ul><li>0.44436 rad</li><li>25.46013 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-03 09:28:08.995
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0065255237080641786 0.6783031599782409 0.231493439286664 0.3796008242820292

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.3738849675846366 38.86390829714029 13.26359705609381 21.74952513104745

[END] 2026-03-03 09:28:17.642



[START] 2026-03-03 09:29:19.031
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0026744437984058097 0.6576030554091132 0.16232690143638706 0.32539059191857556

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.15323434219358967 37.67787966984981 9.30064635374108 18.643507610198053

[END] 2026-03-03 09:29:53.400



[START] 2026-03-03 09:29:53.410
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0026744437984058097 0.6576030554091132 0.16232690143638706 0.32539059191857556

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.15323434219358967 37.67787966984981 9.30064635374108 18.643507610198053

[END] 2026-03-03 09:30:27.976



[START] 2026-03-03 09:30:27.988
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.006666712206210218 0.6288374468231976 0.30321945241744436 0.444363049530556

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.38197447264419526 36.02973170275156 17.37319488978745 25.460127309663616

[END] 2026-03-03 09:30:37.126



[START] 2026-03-03 09:30:40.315

best: exp2-3

[END] 2026-03-03 09:30:40.315
```

<br>

#### [Secondary validation — Gravity & Linear Accel]

Gravity direction error remains low (mean/p90 ≈ 3.20° / 5.43°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 15.15° / 30.52°).<br>

<br>

```
[START] 2026-03-03 09:30:48.508

[Gravity]
RMSE norm: 0.6766599654644616

Gravity est/ref angle error in rad — min/max/mean/p90
3.268142294931132e-05 0.4057637100354742 0.05589164572705595 0.09476578786202543

Gravity est/ref angle error in deg — min/max/mean/p90
0.00187250760347753 23.248548064602797 3.2023554102007075 5.429679686726142


[Linear accel]
RMSE norm: 0.8551364875651073

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0008710787681218445 3.098396866952071 0.2643509816669709 0.5326364688596705

Linear accel est/ref angle error in deg — min/max/mean/p90
0.04990913703683657 177.52506373291092 15.146195559657633 30.51782168041042
. . .
[END] 2026-03-03 09:30:49.118
```
<br>

#### [Observation]

- Accelerometer correction yields a large gain over gyro-only, dominating all other effects
- The best run in exp 2-3 corresponds to disabled gating (`σ_acc = σ_gyro = inf`), indicating that gating does not provide benefit under this motion pattern (either because accel is already reliable enough, or because the proxy does not separate good/bad updates well)
- Time-varying sigma (exp 2-4) degrades performance, consistent with over-gating (down-weighting useful accel corrections) under this motion pattern
- This dataset acts as a negative control: adaptive gating is not universally beneficial, and “no gating” is the best configuration when the accel reliability proxy does not need to reject updates

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

|         |  tau  |         K         |       σ_gyro       |        σ_acc       |
|:-------:|------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  0.3  | 0.033329264322977 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  0.2  | 0.049993896484466 |  inf (not applied) |          0.6755996 |
| exp 2-3 |  0.2  | 0.049993896484466 |  inf (not applied) |          0.6755996 |
| exp 2-3 |  0.2  | 0.049993896484466 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-03 09:19:08.857

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786

[END] 2026-03-03 09:19:09.557
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| exp 2-1 | <ul><li>0.40981 rad</li><li>23.48033 deg</li></ul> | <ul><li>0.75477 rad</li><li>43.24533 deg</li></ul> |
| exp 2-2 | <ul><li>0.37464 rad</li><li>21.46521 deg</li></ul> | <ul><li>0.56815 rad</li><li>32.55285 deg</li></ul> |
| exp 2-3 | <ul><li>0.37464 rad</li><li>21.46521 deg</li></ul> | <ul><li>0.56815 rad</li><li>32.55285 deg</li></ul> |
| exp 2-4 | <ul><li>0.20511 rad</li><li>11.75182 deg</li></ul> | <ul><li>0.35550 rad</li><li>20.36846 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-03 09:20:30.696
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.005370781710236194 1.1276601890042688 0.4098091184475541 0.7547733170837478

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.30772312468258817 64.61016955486933 23.480332893021696 43.24532555798819

[END] 2026-03-03 09:20:43.054



[START] 2026-03-03 09:22:12.537
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.006049463731097126 1.1110652039347422 0.37463867609401474 0.5681543377919447

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.3466087401093293 63.65934694930284 21.465214982555736 32.55284566752856

[END] 2026-03-03 09:23:00.306



[START] 2026-03-03 09:23:00.315
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.006049463731097126 1.1110652039347422 0.37463867609401474 0.5681543377919447

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.3466087401093293 63.65934694930284 21.465214982555736 32.55284566752856

[END] 2026-03-03 09:23:51.305



[START] 2026-03-03 09:23:51.315
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.0029110099044786742 0.7230102613066783 0.20510794235564486 0.35549670098991354

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.16678858164740895 41.42543651752348 11.751819441591026 20.36846059754624

[END] 2026-03-03 09:24:04.330



[START] 2026-03-03 09:24:08.318

best: exp2-4

[END] 2026-03-03 09:24:08.319
```

<br>

#### [Secondary validation — Gravity & Linear Accel]

Gravity direction error remains low (mean/p90 ≈ 2.81° / 4.71°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 11.89° / 23.83°).<br>

<br>

```
[START] 2026-03-03 09:24:20.040

[Gravity]
RMSE norm: 0.5512051303746985

Gravity est/ref angle error in rad — min/max/mean/p90
0.00011130987185848577 0.26905061237099187 0.04904475787599305 0.08212381211599154

Gravity est/ref angle error in deg — min/max/mean/p90
0.006377585875633248 15.41546456426813 2.8100576335354055 4.7053478317716495


[Linear accel]
RMSE norm: 0.7822488748515818

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0005318317560533587 2.8115581878599625 0.20743818058754332 0.41598265398237994

Linear accel est/ref angle error in deg — min/max/mean/p90
0.030471715032888622 161.0904180198257 11.885332257538836 23.83405042384126
. . .
[END] 2026-03-03 09:24:20.975
```
<br>

#### [Observation]

- This dataset is a clean “gating matters” case — fixed gating reduces large error regions and significantly tightens the error distribution
- The best result is obtained with time-varying sigma (exp 2-4), indicating that reliability conditions change over time and a fixed sigma cannot balance all segments equally well
- The reduced-error region aligns with periods where `||a||` deviates from g0, supporting the interpretation that gating suppresses incorrect tilt injections during dynamic motion
- Compared with fixed gating, adaptive sigma improves both average error and the tail behavior, suggesting better segment-wise trade-offs (strictness during dynamic motion, permissiveness when stable)

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

|         |  tau  |         K         |       σ_gyro       |        σ_acc       |
|:-------:|------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  3.0  | 0.003352945963646 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  3.0  | 0.003352945963646 |  inf (not applied) |  inf (not applied) |
| exp 2-3 |  3.0  | 0.003352945963646 |  inf (not applied) |  inf (not applied) |
| exp 2-4 |  3.0  | 0.003352945963646 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-03 09:31:23.414

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965

[END] 2026-03-03 09:31:23.999
```

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| exp 2-1 | <ul><li>0.85937 rad</li><li>49.23849 deg</li></ul> | <ul><li>2.00162 rad</li><li>114.68462 deg</li></ul> |
| exp 2-2 | <ul><li>0.85937 rad</li><li>49.23849 deg</li></ul> | <ul><li>2.00162 rad</li><li>114.68462 deg</li></ul> |
| exp 2-3 | <ul><li>0.85937 rad</li><li>49.23849 deg</li></ul> | <ul><li>2.00162 rad</li><li>114.68462 deg</li></ul> |
| exp 2-4 | <ul><li>0.87787 rad</li><li>50.29802 deg</li></ul> | <ul><li>2.04125 rad</li><li>116.95505 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-03 09:41:24.170
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.002969310113133571 3.139880414590167 0.8593738181216991 2.001624262321689

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.1701289375480666 179.90189593180372 49.23849280241658 114.68462240201954

[END] 2026-03-03 09:42:51.162



[START] 2026-03-03 09:53:30.916
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.002969310113133571 3.139880414590167 0.8593738181216991 2.001624262321689

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.1701289375480666 179.90189593180372 49.23849280241658 114.68462240201954

[END] 2026-03-03 09:59:25.790



[START] 2026-03-03 09:59:25.801
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.002969310113133571 3.139880414590167 0.8593738181216991 2.001624262321689

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.1701289375480666 179.90189593180372 49.23849280241658 114.68462240201954

[END] 2026-03-03 10:05:18.977



[START] 2026-03-03 10:05:18.987
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.00218375174558757 3.1410985491597296 0.8778661053503135 2.0412506145658096

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.12511975852649404 179.97168990151866 50.29802281415986 116.95504514310642

[END] 2026-03-03 10:06:51.075



[START] 2026-03-03 10:07:12.455

best: exp2-3

[END] 2026-03-03 10:07:12.458
```

<br>

#### [Secondary validation — Gravity & Linear Accel]

Gravity direction error remains low (mean/p90 ≈ 1.45° / 3.42°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 20.04° / 49.35°).<br>

<br>

```
[START] 2026-03-03 10:08:18.854

[Gravity]
RMSE norm: 0.34626176727215563

Gravity est/ref angle error in rad — min/max/mean/p90
1.6384502658134697e-06 0.21117532162835242 0.02528871049584768 0.059667497541057046

Gravity est/ref angle error in deg — min/max/mean/p90
9.387628517319968e-05 12.099454666622325 1.4489363807402595 3.4186957832097864


[Linear accel]
RMSE norm: 0.6593208369308915

Linear accel est/ref angle error in rad — min/max/mean/p90
2.007417052311769e-05 3.1381061892000726 0.3496816823772473 0.8613714614092618

Linear accel est/ref angle error in deg — min/max/mean/p90
0.0011501652482005676 179.80024030504634 20.035284573250443 49.35294933176656
. . .
[END] 2026-03-03 10:08:24.996
```
<br>

#### [Observation]

- Very long, uncontrolled motion introduces large error plateaus/spikes that dominate both mean and tail metrics
- The best exp2 configuration again corresponds to disabled gating `(σ_acc = σ_gyro = inf)`, indicating that the current accel/gyro reliability proxies do not explain the dominant failure mode in this dataset
- Time-varying sigma (exp 2-4) slightly worsens the metrics, consistent with unnecessary suppression of helpful correction without effectively rejecting the true failure segments
- Remaining error may involve yaw/heading drift and unmodeled effects (e.g. mounting/frame mismatch), in addition to limitations of the current accel-magnitude proxy

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-2-data-sum"></a>

| Dataset | best exp2 | exp 1-2 Mean   | best exp2 Mean | exp 1-2 p90    | best exp2 p90  |
|:--------|----------:|---------------:|---------------:|---------------:|---------------:|
| data 01 | exp2-4    | 22.40684 deg   | 19.39258 deg   | 32.44667 deg   | 29.34040 deg   |
| data 02 | exp2-3    | 21.99183 deg   | 9.30065 deg    | 31.17450 deg   | 18.64351 deg   |
| data 03 | exp2-4    | 30.81266 deg   | 11.75182 deg   | 46.56837 deg   | 20.36846 deg   |
| data 04 | exp2-3    | 50.88316 deg   | 49.23849 deg   | 116.70718 deg  | 114.68462 deg  |

** best exp2 = minimum mean error among exp 2-3, 2-4 (per dataset)<br>

<br>

Across all datasets:<br>

- Adding accelerometer correction consistently reduces drift relative to gyro-only,
  and the improvement gap tends to grow over time
- Gating is dataset-dependent. It helps when linear acceleration frequently violates the gravity assumption, but may be marginal or not selected by the sigma sweep when accel measurements are already consistent

<br>

Datasets where exp2-4 wins (data 01, data 03):<br>

- Fixed gating already shows a meaningful reduction, indicating that the reliability proxy `| ||a|| - g0 |` is informative in these datasets
- Time-varying sigma further improves performance by adapting strictness over time (more selective during dynamic motion, more permissive when stable), producing additional error reduction beyond the best fixed-sigma setting

<br>

Datasets where exp2-3 wins (data 02, data 04):<br>

- The best exp 2-3 configuration corresponds to disabled gating (`σ_acc = σ_gyro = inf`)
- This indicates that gating is either unnecessary (data 02: accel is already consistent) or ineffective against the dominant failure mode (data 04: long uncontrolled motion where the magnitude proxy is insufficient)
- Running exp 2-4 in these cases can increase error by over-gating, but it provides a useful confirmation that accel/gyro gating does not add value for these datasets under the current proxy design

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-2-conclusion"></a>

Experiment 2 confirms:<br>

1. Accelerometer correction stabilizes roll/pitch by continuously correcting gyro drift
2. Gating acts as a reliability controller that prevents incorrect tilt injections during dynamic motion
3. Time-varying gating tends to help when fixed gating already demonstrates a clear advantage, suggesting that the proxy captures meaningful reliability structure in the data
4. When fixed gating is not selected by the sigma sweep (`best sigma = inf`), adaptive gating often provides limited benefit and can degrade performance by suppressing valid corrections
<br>

Next steps:<br>
- Experiment 3: add magnetometer correction to constrain yaw drift

<br>
<br>
<br>
<br>
