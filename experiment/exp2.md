
 * [Experiment 2](#exp-2) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-2-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Method](#exp-2-method) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Accelerometer Correction](#exp-2-method-acc-correction) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Detect Quasi-static](#exp-2-method-quasi-static) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Suggest Tau and K](#exp-2-method-tau-k) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Suggest Gate Sigma](#exp-2-method-sigma) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Hyperparameter Search with Optuna](#exp-2-method-optuna) <br>
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
and when gating is necessary to avoid injecting incorrect tilt updates during dynamic motion.<br>

<br>

Four runs are compared (same dataset, same trimmed start):<br>

- [exp 1-2] Gyro-only (baseline, after stabilization trimming)
- [exp 2-1] Gyro + Acc, without gating
- [exp 2-2] Gyro + Acc with Acc gating — fixed sigma
- [exp 2-3] Gyro + Acc with Gyro/Acc gating — fixed sigma
- [exp 2-4] Gyro + Acc with Gyro/Acc gating — time-varying sigma

<br>

Key hypothesis:<br>

Accelerometer-based correction is expected to counteract accumulated gyro drift, which may make its benefits more visible.<br>

<br>
<br>
<br>
<br>

### Method <a name="exp-2-method"></a>

### Accelerometer Correction <a name="exp-2-method-acc-correction"></a>

At each step:<br>

- propagate orientation using gyro integration
- predict gravity direction in the body frame  
- compute error axis `e = g_pred × a_unit`  
- apply a small-angle correction quaternion with effective gain `K_eff = K * weight_acc * weight_gyro`, where `K = dt_median / tau`

** `weight_acc`: Accel magnitude residual-based confidence (approaches to 1 when `| ||a|| - g0 |` is small)<br>
** `weight_gyro`: Gyro norm-based confidence (approaches to 1 when `||w||` is small)<br>

<br>

##### [Implementation]

```py
# pipeline.py

def predict_gravity_body_frame(q_pred: Quat, g_world_unit: Vec3) -> Vec3:
       g_pred: Vec3 = libq.rotate_world_to_body(q_pred, g_world_unit)
       return safe_unit_vec3(g_pred)
. . .

def calc_acc_gating(g0: float, acc_sigma: float, a_meas: Vec3, g_pred: Vec3) -> float:
        . . .
        theta: float = np.arccos(np.clip(np.dot(a_unit, g_unit), -1, 1))
        weight_acc : float = np.exp(-0.5 * (theta / acc_sigma) ** 2)
        return weight_acc

def calc_gyro_gating(gyro_sigma: float, w: Vec3) -> float:
        . . .
        weight_gyro: float = np.exp(-0.5 * (w_norm / gyro_sigma) ** 2)
        return weight_gyro
. . .

def integrate_gyro_acc(. . .) -> tuple[. . .]:
	. . .
        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])
                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                . . .
                weight_acc[i] = calc_acc_gating(g0, acc_gate_sigma[i], a_meas, g_pred)
                weight_gyro[i] = calc_gyro_gating(gyro_gate_sigma[i], w_avg[i])

                e_axis: Vec3 = np.cross(g_pred, a_unit)
                dq_corr: Quat = small_angle_correction_quat(K*(weight_acc[i]*weight_gyro[i]), e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q
		. . .
        return res, g_body_est, a_lin_est, weight_acc, weight_gyro
```

<br>
<br>
<br>

### Detect Quasi-static <a name="exp-2-method-quasi-static"></a>

A quasi-static segment is automatically detected to evaluate gravity stability without being contaminated by motion.<br>

<br>

Quasi-static here means low angular rate and near-gravity accelerometer magnitude, not necessarily perfectly motionless.<br>

<br>

- Compute `||w||` and the accelerometer magnitude residual `| ||a|| - g0 |`
- `quasi_static = (||w|| < w_thr) & (| ||a|| - g0 | < a_thr)`, if thresholds are not provided, they are set to a low percentile
- Apply a majority-vote smoothing window to debounce short spikes
- Keep the longest continuous True run, and reject the segment if its duration is shorter than `min_duration_s`

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

def quasi_static_detector(. . .) -> tuple[int, int, int]:
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

The `tau` that produces the most stable gravity direction during the quasi-static segment is selected and converted to the discrete gain `K = median(dt) / tau`.<br>

The quasi-static score is computed as follows:<br>

- Run the pipeline and collect the estimated gravity direction in the body frame
- Within the quasi-static segment, normalize `g_body_est`
- Measure stability by the mean angular deviation from the mean direction

<br>

A lower score indicates lower orientation error (better quasi-static gravity consistency).<br>

##### [Implementation]

```py
# autotune.py

def calc_score_quasi(. . .) -> float:
        dt_median: float = float(np.median(dt))
        . . .
        K: float = float(dt_median / tau)
        _, extra = runner_func(K=K, **runner_kwargs)
        g_body_est, _, _, _, _ = extra

        gb = g_body_est[s:e]
        . . .
        mean_dir: Vec3 = as_vec3(np.mean(gb_unit, axis=0))
        . . .
        dot: ScalarBatch = np.clip(gb_unit @ mean_dir, -1, 1)
        ang: ScalarBatch = np.arccos(dot)
        quasi_score: float = score_angle_err(ang)
        return quasi_score
```

<br>
<br>
<br>

### Suggest Gate Sigma <a name="exp-2-method-sigma"></a>

Gate sigmas control how aggressively measurements are trusted via soft gating weights.<br>
A larger sigma makes gating weaker (more permissive), while a smaller sigma makes gating stricter (more selective).<br>

<br>

#### [Fixed sigma]

A fixed sigma_base is suggested from robust percentiles of the data.<br>
If a quasi-static segment exists, statistics are computed on that segment, otherwise, the full sequence is used.<br>

<br>

```py
# autotune.py

def suggest_fixed_gyro_gate_sigma(. . .) -> float:
        . . .
        w_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(w_use, axis=1))
        gyro_sigma: float = max(sigma_floor, float(np.percentile(w_norm, p_gyro)))
        return gyro_sigma

def suggest_fixed_acc_gate_sigma(. . .) -> float:
        . . .
        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a_use, axis=1))
        acc_resid: ScalarBatch = np.abs(a_norm - g0)
        acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))
        return acc_sigma

def suggest_fixed_gate_sigma(w. . .) -> tuple[float, float, float]:
        gyro_sigma: float = suggest_fixed_gyro_gate_sigma(. . .)
        . . .
        acc_sigma: float = suggest_fixed_acc_gate_sigma(. . .)
        . . .
        return gyro_sigma, acc_sigma, mag_sigma
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

def suggest_timevarying_gate_sigma(. . .) -> tuple[ScalarBatch, ScalarBatch, ScalarBatch]:
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

### Hyperparameter Search with Optuna <a name="exp-2-method-optuna"></a>

#### [Discrete search vs Optuna]

A discrete search is simple and interpretable but it has two limitations:<br>

- It searches only a small predefined grid
- It becomes inefficient when several parameters interact

<br>

Optuna is a hyperparameter optimization framework that automatically explores the search space and concentrates trials around promising regions.<br>
Its default optimization strategy is particularly efficient when gradients are unavailable.<br>

<br>

The Optuna-based approach:

- Searches a much richer continuous space
- Better captures parameter interactions
- Reduces dependence on hand-picked scale multipliers

<br>

#### [How Optuna works]

In this experiment, Optuna is used with `TPESampler(seed=42)`.<br>

<br>

TPE(Tree-structured Parzen Estimator) is a Bayesian optimization-style sampler.<br>

<br>

Instead of testing parameters on a fixed grid, it uses previous trial results to model which parameter regions are associated with good objective values ​​and which are associated with worse objective values.<br>

<br>

TPE fits one probability density to promising trials and another to the remaining trials,<br>
then samples new parameters that maximize the ratio between them:<br>

- Early trials explore the range
- Later trials increasingly sample near regions that already produced low error

<br>

#### [Optimization target]

| exp | trial |             Target ∈ Range (min, max)             |
|:---:|------:|:--------------------------------------------------|
| 2-1 |    15 | <ul><li>tau ∈ (0.1, 4)</li></ul> |
| 2-2 |    20 | <ul><li>tau ∈ 2-1_best*(0.9, 1.1)</li><li>acc_gate_sigma ∈ suggested_acc*(0.01, 10)</li></ul> |
| 2-3 |    20 | <ul><li>tau ∈ 2-2_best*(0.9, 1.1)</li><li>acc_gate_sigma ∈ 2-2_best*(0.7, 1.3)</li><li>gyro_gate_sigma ∈ suggested_gyro*(0.1, 10)</li></ul> |
| 2-4 |    40 | <ul><li>tau ∈ 2-3_best*(0.9, 1.1)</li><li>percentile `p` ∈ (50. 80)</li><li>sliding window size `win_s` ∈ (5, 10)</li><li>update ratio `update_ratio` ∈ (0.1, 0.5)</li><li>EMA factor `ema_alpha` ∈ (0.02, 0.2)</li></ul> |

<br>

** 2-x_best: Optimal value selected from experiment 2-x<br>
** x_gate_sigma: fixed gate sigma estimated from robust percentiles of the data<br>
** For exp2-4, Optuna optimizes the parameters of the function `suggest_timevarying_gate_sigma(...)`, rather than the sigma values directly<br>

<br>

#### [Implementation]

The pipeline is wrapped inside an Optuna objective function.<br>

<br>

For each trial:<br>

1. Sample candidate parameters from the defined search space
2. Run the orientation pipeline using those parameters
3. Compute the evaluation score
4. Return the score to Optuna

<br>

```py
# optuna_exp_2.py

def exp_2_1(. . .) -> tuple[float, float]:
	. . .

def exp_2_2(. . .) -> tuple[float, float]:
	. . .

def exp_2_3(. . .) -> tuple[float, float]:
        def objective(trial):
                tau: float = trial.suggest_float("tau", . . .)
                acc_gate_sigma: float = trial.suggest_float("acc_gate_sigma", . . .)
                gyro_gate_sigma: float = trial.suggest_float("gyro_gate_sigma", . . .)

                score: float = calc_score_quasi_ori(. . .)
                return score
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
	. . .
        return best_tau, best_K, best_acc_sigma, best_gyro_sigma

def exp_2_4(. . .) -> tuple[Any, ...]:
	. . .
        def objective_sigma(trial):
                tau: float = trial.suggest_float("tau", . . .)
                p: int = trial.suggest_int("p", . . .)
                win: float = trial.suggest_float("win_s", . . .)
                update_ratio: float = trial.suggest_float("update_ratio", . . .)
                update: float = win * update_ratio
                ema: float = trial.suggest_float("ema_alpha", . . .)

                [. . .] = suggest_timevarying_gate_sigma(. . .)
                score: float = calc_score_quasi_ori(. . .)
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective_sigma, n_trials=n_trials)
        best = study.best_params
        . . .
        [timevarying_gyro_sigma, timevarying_acc_sigma, _] = suggest_timevarying_gate_sigma(. . .)
        return best_tau, best_K, timevarying_acc_sigma, timevarying_gyro_sigma
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

| exp |  tau   |      K      |    σ_gyro    |     σ_acc    |
|:---:|-------:|------------:|-------------:|-------------:|
| 2-1 |  3.96  | 0.002521892 |          inf |          inf |
| 2-2 |  4.03  | 0.002483076 |          inf |    2.9133442 |
| 2-3 |  4.41  | 0.002266319 |    0.1969152 |    2.9791675 |
| 2-4 |  4.50  | 0.002220946 | time-varying | time-varying |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-17 04:05:48.975

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301

[END] 2026-03-17 04:05:49.806
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |
| 2-1 | <ul><li>0.33986 rad</li><li>19.47266 deg</li></ul> | <ul><li>0.51315 rad</li><li>29.40161 deg</li></ul> |
| 2-2 | <ul><li>0.33849 rad</li><li>19.39417 deg</li></ul> | <ul><li>0.51207 rad</li><li>29.33957 deg</li></ul> |
| 2-3 | <ul><li>0.33782 rad</li><li>19.35529 deg</li></ul> | <ul><li>0.51093 rad</li><li>29.27439 deg</li></ul> |
| 2-4 | <ul><li>0.33899 rad</li><li>19.42277 deg</li></ul> | <ul><li>0.51110 rad</li><li>29.28387 deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>

<br>

```
[START] 2026-03-17 04:05:50.987
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.002792569731405555 1.4026426778532652 0.3398619909190511 0.51315497786888

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.1600024596055202 80.36550560592005 19.47265769657514 29.401614468015985

[END] 2026-03-17 04:07:15.421




[START] 2026-03-17 04:07:15.434
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.002748354142392041 1.401548675497198 0.33849208040378065 0.5120721420212044

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.15746909296636083 80.30282388814003 19.39416760573955 29.339572544038706

[END] 2026-03-17 04:09:43.773




[START] 2026-03-17 04:09:43.784
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.00163324271634456 1.4005973749091025 0.3378135110770164 0.510934527798625

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.09357791456702556 80.24831837939384 19.355288447208924 29.27439205037085

[END] 2026-03-17 04:12:31.277




[START] 2026-03-17 04:12:31.327
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.0027016091819519134 1.400982210449621 0.33899127245759747 0.5110999765456887

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.15479080401963552 80.27036783167219 19.42276920358972 29.28387156530333

[END] 2026-03-17 04:17:37.159




[START] 2026-03-17 04:17:38.907

best: exp2-3

[END] 2026-03-17 04:17:38.916
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Estimated gravity and linear acceleration are compared against reference signals as a sanity check.<br>

<br>

Gravity direction error remains low (mean/p90 ≈ 0.98° / 1.52°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 8.92° / 19.42°).<br>

<br>

These checks support that Experiment 2 is physically consistent and the orientation error comparisons are meaningful.<br>

<br>

|      |  Mean error  |  p90 error   |
|:----:|-------------:|-------------:|
| grav | <ul><li>0.01704 rad</li><li>0.97623 deg</li></ul> | <ul><li>0.02655 rad</li><li>1.52132 deg</li></ul> |
| acc  | <ul><li>0.15561 rad</li><li>8.91570 deg</li></ul> | <ul><li>0.33894 rad</li><li>19.42011 deg</li></ul> |

<br>

```
[START] 2026-03-17 04:17:45.687

[Gravity]
RMSE norm: 0.20127494414237856

Gravity est/ref angle error in rad — min/max/mean/p90
4.258330309511835e-05 0.09699583475086833 0.01703841175553693 0.026552023496872593

Gravity est/ref angle error in deg — min/max/mean/p90
0.002439843545076657 5.55745196157312 0.9762290831983539 1.5213188839029932


[Linear accel]
RMSE norm: 0.7422099471787518

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0009246217951388085 3.073500429721502 0.15560836401671843 0.3389448841717979

Linear accel est/ref angle error in deg — min/max/mean/p90
0.052976926507263544 176.09860295468695 8.915702515093352 19.42011135059456
. . .
[END] 2026-03-17 04:17:46.155
```
<br>

#### [Observation]

- Accelerometer correction improves over the gyro-only baseline, confirming that gravity-based tilt correction is beneficial on this sequence
- The best result is exp 2-3, indicating that jointly tuned fixed gyro/acc gating provides the most effective trade-off for this dataset
- The gain over exp 2-1 is modest but consistent in both mean and p90 error, suggesting that soft gating helps suppress occasional incorrect tilt corrections without removing too many useful ones
- The time-varying schedule remains competitive, but it does not outperform the simpler fixed-gating configuration on this sequence

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

| exp |  tau   |      K      |    σ_gyro    |     σ_acc    |
|:---:|-------:|------------:|-------------:|-------------:|
| 2-1 |  0.32  | 0.030621686 |          inf |          inf |
| 2-2 |  0.29  | 0.033967126 |          inf |   24.0230485 |
| 2-3 |  0.27  | 0.037574026 |    6.2974666 |   22.9724577 |
| 2-4 |  0.27  | 0.037549933 | time-varying | time-varying |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-17 04:21:30.057

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874

[END] 2026-03-17 04:21:30.973
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |
| 2-1 | <ul><li>0.16965 rad</li><li>9.72036 deg</li></ul>  | <ul><li>0.32722 rad</li><li>18.74848 deg</li></ul> |
| 2-2 | <ul><li>0.16139 rad</li><li>9.24721 deg</li></ul>  | <ul><li>0.32466 rad</li><li>18.60155 deg</li></ul> |
| 2-3 | <ul><li>0.15914 rad</li><li>9.11787 deg</li></ul>  | <ul><li>0.32830 rad</li><li>18.81032 deg</li></ul> |
| 2-4 | <ul><li>0.25474 rad</li><li>14.59551 deg</li></ul> | <ul><li>0.39538 rad</li><li>22.65358 deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>

<br>

```
[START] 2026-03-17 04:21:32.349
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0014567112071040497 0.6559884332313202 0.16965223911367894 0.32722263712136174

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.08346340413646963 37.58536863355405 9.72035728615807 18.74847606819489

[END] 2026-03-17 04:24:01.134




[START] 2026-03-17 04:24:01.145
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.004575584167113969 0.6583342412761097 0.16139428046193366 0.3246582173822904

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.2621616615825124 37.719773534068324 9.24721110801952 18.601545640246062

[END] 2026-03-17 04:28:46.238




[START] 2026-03-17 04:28:46.252
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.004691134603283964 0.6479224948123612 0.1591369156292557 0.3283019619432845

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.2687822138959489 37.123224404335275 9.117873630285818 18.81031682521477

[END] 2026-03-17 04:33:34.495




[START] 2026-03-17 04:33:34.507
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.004524775138459319 0.604989026170517 0.25473977323201047 0.39537957747500835

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.25925051867944165 34.663317851300334 14.595513880313861 22.65358109498373

[END] 2026-03-17 04:43:17.978




[START] 2026-03-17 04:43:22.166

best: exp2-3

[END] 2026-03-17 04:43:22.177
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Gravity direction error remains low (mean/p90 ≈ 3.38° / 5.74°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 15.85° / 31.78°).<br>

<br>

|      |  Mean error  |  p90 error   |
|:----:|-------------:|-------------:|
| grav | <ul><li>0.05904 rad</li><li>3.38289 deg</li></ul> | <ul><li>0.10017 rad</li><li>5.73918 deg</li></ul> |
| acc  | <ul><li>0.27664 rad</li><li>15.85026 deg</li></ul> | <ul><li>0.55462 rad</li><li>31.77723 deg</li></ul> |

<br>

```
[START] 2026-03-17 04:43:32.825

[Gravity]
RMSE norm: 0.7095678954417411

Gravity est/ref angle error in rad — min/max/mean/p90
0.00011644698410429857 0.421200010323766 0.05904257526139471 0.10016762190613224

Gravity est/ref angle error in deg — min/max/mean/p90
0.006671920726203293 24.132982922418496 3.38289037406144 5.739181979083548


[Linear accel]
RMSE norm: 0.884723577813496

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0012662548761560905 3.100442485184134 0.2766393045845288 0.5546173107550053

Linear accel est/ref angle error in deg — min/max/mean/p90
0.07255106019160472 177.64226902410314 15.850264600127586 31.777231151157444
. . .
[END] 2026-03-17 04:43:33.664
```
<br>

#### [Observation]

- Accelerometer correction provides a large improvement over the gyro-only baseline
- The best result is exp 2-3, although exp 2-1 and exp 2-2 remain very close
- This suggests that, for this dataset, mild jointly tuned fixed gating improves the balance between average error and tail error better than ungated correction
- In contrast, the time-varying schedule clearly underperforms, indicating that additional adaptivity is not necessarily beneficial when the fixed-gating solution is already well matched to the motion pattern

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

| exp |  tau   |      K      |    σ_gyro    |     σ_acc    |
|:---:|-------:|------------:|-------------:|-------------:|
| 2-1 |  0.33  | 0.030621686 |          inf |          inf |
| 2-2 |  0.33  | 0.030128090 |          inf |    4.9552873 |
| 2-3 |  0.30  | 0.033410594 |    1.7842051 |    5.5064391 |
| 2-4 |  0.27  | 0.036953844 | time-varying | time-varying |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-17 04:46:01.090

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786

[END] 2026-03-17 04:46:01.897
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| 2-1 | <ul><li>0.33687 rad</li><li>19.30136 deg</li></ul> | <ul><li>0.56227 rad</li><li>32.21544 deg</li></ul> |
| 2-2 | <ul><li>0.23173 rad</li><li>13.27729 deg</li></ul> | <ul><li>0.39782 rad</li><li>22.79322 deg</li></ul> |
| 2-3 | <ul><li>0.24289 rad</li><li>13.91633 deg</li></ul> | <ul><li>0.38963 rad</li><li>22.32396 deg</li></ul> |
| 2-4 | <ul><li>0.23669 rad</li><li>13.56111 deg</li></ul> | <ul><li>0.45578 rad</li><li>26.11399 deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>

<br>

```
[START] 2026-03-17 04:46:03.781
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.007372901990831988 0.9060120450830597 0.3368722815697167 0.5622654170401866

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.4224361668382753 51.91066637127579 19.301359968887475 32.21543536256581

[END] 2026-03-17 04:49:23.512




[START] 2026-03-17 04:49:23.525
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.006504140308894591 0.7830372741420375 0.231732513184545 0.3978166616786644

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.3726597890605756 44.864731009767176 13.277294981434132 22.793215734171223

[END] 2026-03-17 04:55:51.550




[START] 2026-03-17 04:55:51.565
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.006762815923363894 0.7632415810002321 0.2428857537054454 0.3896265363969847

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.3874808100326199 43.73052134020566 13.916328591176018 22.32395612184758

[END] 2026-03-17 05:02:29.047




[START] 2026-03-17 05:02:29.063
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.0019336906480635027 0.8863477754775086 0.23668594042411384 0.45577514740030156

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.11079231301795571 50.78398671567033 13.561105456386565 26.113992352990273

[END] 2026-03-17 05:15:29.988




[START] 2026-03-17 05:15:34.348

best: exp2-3

[END] 2026-03-17 05:15:34.362
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Gravity direction error remains low (mean/p90 ≈ 2.79° / 4.58°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 11.99° / 24.07°).<br>

<br>

|      |  Mean error  |  p90 error   |
|:----:|-------------:|-------------:|
| grav | <ul><li>0.04878 rad</li><li>2.79477 deg</li></ul> | <ul><li>0.07998 rad</li><li>4.58227 deg</li></ul> |
| acc  | <ul><li>0.20927 rad</li><li>11.99003 deg</li></ul> | <ul><li>0.42003 rad</li><li>24.06610 deg</li></ul> |

<br>

```
[START] 2026-03-17 05:15:48.577

[Gravity]
RMSE norm: 0.552626287335186

Gravity est/ref angle error in rad — min/max/mean/p90
0.0002262118650697147 0.28012067223628173 0.04877787155003603 0.0799756400746747

Gravity est/ref angle error in deg — min/max/mean/p90
0.012960985144277503 16.0497322735064 2.7947661734483153 4.582266640136192


[Linear accel]
RMSE norm: 0.7964169595942013

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0004397065133648307 2.893358227492407 0.2092655255132763 0.42003276708629594

Linear accel est/ref angle error in deg — min/max/mean/p90
0.025193327440217524 165.77721505476762 11.990031409497982 24.066104811246273


[Consistency ratio]
rmse_norm of (a_lin_est / g_est):  1.44114925012814

[END] 2026-03-17 05:15:49.671
```

<br>

#### [Observation]

- This dataset is another clear case where gating helps substantially relative to ungated gyro+acc correction
- Both accel-only and joint fixed gating reduce error strongly compared with exp 2-1
- The best result is exp 2-3 under the selected ranking criterion, indicating that jointly tuned fixed gyro/acc gating provides the best overall trade-off between mean and tail error on this sequence
- Although exp 2-2 achieves a slightly lower mean error, exp 2-3 yields the best combined result once p90 error is also taken into account
- The time-varying schedule remains competitive, but it does not outperform the best fixed-gating configuration here

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

| exp |  tau   |      K      |    σ_gyro    |     σ_acc    |
|:---:|-------:|------------:|-------------:|-------------:|
| 2-1 |  3.96  | 0.002537040 |          inf |          inf |
| 2-2 |  3.98  | 0.002527921 |          inf |    1.7620850 |
| 2-3 |  4.38  | 0.002299083 |    2.6397807 |    1.8511999 |
| 2-4 |  4.77  | 0.002108354 | time-varying | time-varying |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-17 05:16:08.763

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965

[END] 2026-03-17 05:16:09.695
```

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| 2-1 | <ul><li>0.86428 rad</li><li>49.51985 deg</li></ul> | <ul><li>2.01278 rad</li><li>115.32398 deg</li></ul> |
| 2-2 | <ul><li>0.87043 rad</li><li>49.87187 deg</li></ul> | <ul><li>2.02039 rad</li><li>115.75971 deg</li></ul> |
| 2-3 | <ul><li>0.87335 rad</li><li>50.03905 deg</li></ul> | <ul><li>2.02857 rad</li><li>116.22858 deg</li></ul> |
| 2-4 | <ul><li>0.87640 rad</li><li>50.21389 deg</li></ul> | <ul><li>2.03473 rad</li><li>116.58132 deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>

<br>

```
[START] 2026-03-17 05:16:11.902
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0018930921712486044 3.1415478697090844 0.8642844509787638 2.012783220058988

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.10846619164180232 179.99743407264518 49.51985133986466 115.32398358413164

[END] 2026-03-17 05:41:25.950




[START] 2026-03-17 05:41:25.961
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.002687737167054129 3.1370720491834305 0.8704284129764128 2.020388085045019

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.15399599611264989 179.7409884466672 49.87187443181871 115.75971025159804

[END] 2026-03-17 06:32:20.184




[START] 2026-03-17 06:32:20.205
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0024923534722615195 3.14092011061767 0.8733462451214835 2.0285713937500924

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.14280133501536116 179.96146612615615 50.03905389905887 116.2285793028514

[END] 2026-03-17 07:13:07.014




[START] 2026-03-17 07:13:07.025
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.004479150162939574 3.140442692180671 0.8763976984097652 2.0347278032129728

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.2566364001417726 179.93411206465439 50.21388929385873 116.58131558202885

[END] 2026-03-17 08:50:28.899




[START] 2026-03-17 08:50:56.366

best: exp2-1

[END] 2026-03-17 08:50:56.445
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Gravity direction error remains low (mean/p90 ≈ 1.30° / 3.01°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 17.86° / 43.24°).<br>

<br>

|      |  Mean error  |  p90 error   |
|:----:|-------------:|-------------:|
| grav | <ul><li>0.02261 rad</li><li>1.29562 deg</li></ul> | <ul><li>0.05259 rad</li><li>3.01341 deg</li></ul> |
| acc  | <ul><li>0.31165 rad</li><li>17.85600 deg</li></ul> | <ul><li>0.75470 rad</li><li>43.24097 deg</li></ul> |

<br>

```
[START] 2026-03-17 08:52:16.589

[Gravity]
RMSE norm: 0.311917118411453

Gravity est/ref angle error in rad — min/max/mean/p90
1.2384971736170922e-05 0.19014754863324393 0.02261286401008407 0.05259384322202774

Gravity est/ref angle error in deg — min/max/mean/p90
0.0007096066098714055 10.894652021443441 1.2956216704810914 3.0134052449949205


[Linear accel]
RMSE norm: 0.6409108584616177

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0001003705495784379 3.124034313429527 0.31164593517791683 0.7546972116886191

Linear accel est/ref angle error in deg — min/max/mean/p90
0.005750808878253076 178.9939812135617 17.855996788102267 43.24096504004914
. . .
[END] 2026-03-17 08:52:24.211
```

<br>

#### [Observation]

- This is the most difficult sequence, with large long-duration error plateaus and heavy tail behavior
- Accelerometer correction still improves slightly over the gyro-only baseline, but the gain is much smaller than in the shorter datasets
- The best result is exp 2-1, meaning that no gating performs best on this sequence
- Both fixed and time-varying gating slightly degrade performance, suggesting that the current gating proxies do not align well with the dominant failure modes in this dataset
- The remaining error likely includes effects that are not well handled by the current roll/pitch correction logic alone, such as long-horizon drift, heading-related effects, or dataset-specific frame/mounting mismatch

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-2-data-sum"></a>

| Dataset | best | exp 1-2 Mean  | best Mean     | exp 1-2 p90   | best p90      | best grav Mean | best acc Mean  |
|:--------|-----:|--------------:|--------------:|--------------:|--------------:|---------------:|---------------:|
| data 01 | 2-3  | 22.40684 deg  | 19.35529 deg  |  32.44667 deg |  29.27439 deg | 0.97623 deg    |  8.91570 deg   |
| data 02 | 2-3  | 21.99183 deg  |  9.11787 deg  |  31.17450 deg |  18.81032 deg | 3.38289 deg    | 15.85026 deg   |
| data 03 | 2-3  | 30.81266 deg  | 13.91633 deg  |  46.56837 deg |  22.32396 deg | 2.79477 deg    | 11.99003 deg   |
| data 04 | 2-1  | 50.88316 deg  | 49.51986 deg  | 116.70718 deg | 115.32398 deg | 1.29562 deg    | 17.85600 deg   |

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift) per dataset<br>

<br>

Across all datasets:<br>

- Adding accelerometer correction reduces error relative to the gyro-only baseline on all evaluated datasets, although the magnitude of improvement varies significantly
- The magnitude of the gain is strongly dataset-dependent. (large in data 02, data 03 / modest in data 01 / small in data 04)
- The effectiveness of gating is strongly dataset-dependent. In some sequences it improves both average and tail behavior, while in others it suppresses useful corrections and slightly hurts performance

<br>
<br>

Datasets where gating helps (data 01, data 02, data 03):<br>

- A gated configuration outperforms ungated gyro+acc correction on these datasets
- In all three of these datasets, the best overall result is exp 2-3 (jointly tuned fixed gyro/acc gating)
- This suggests that combining accel-based and gyro-based confidence can improve robustness under certain motion patterns
- The improvement is strong in data 02 and data 03, while data 01 shows a smaller but still consistent gain

<br>

Dataset where gating is unnecessary or harmful (data 04):<br>

- The best configuration is exp 2-1 ( gyro+acc without gating)
- This indicates that, for this sequence, the current gating design does not align well with the dominant error sources
- Additional gating appears to attenuate useful corrections more than it filters harmful ones

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-2-conclusion"></a>

Experiment 2 shows that:<br>

1. Accelerometer correction improves roll/pitch estimation across the evaluated datasets
2. Gating can provide an additional benefit in certain motion regimes, but its usefulness depends strongly on the dataset and the motion regime
3. The benefit of gating is not universal. On the longest and most difficult sequence evaluated, ungated gyro+acc correction still performs best.
4. The best configuration is therefore dataset-dependent, which reinforces the need to evaluate robustness across multiple motion patterns rather than drawing conclusions from a single sequence

<br>

Next steps:<br>
- Experiment 3: add magnetometer correction to constrain yaw drift

<br>
<br>
<br>
<br>
