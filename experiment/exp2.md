
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
The correction is expected to counteract accumulated gyro drift, which may make its benefits more visible over longer sequences.<br>

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

#### [Fixed sigma (baseline)]

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

| exp |             Target             |
|:---:|:-------------------------------|
| 2-1 | <ul><li>tau</li></ul> |
| 2-2 | <ul><li>tau</li><li>acc_gate_sigma</li></ul> |
| 2-3 | <ul><li>tau</li><li>acc_gate_sigma</li><li>gyro_gate_sigma</li></ul> |
| 2-4 | <ul><li>tau</li><li>percentile (`p`)</li><li>sliding window size (`win_s`)</li><li>update ratio (`update_ratio`)</li><li>EMA factor (`ema_alpha`)</li></ul> |

** for exp2-4, Optuna optimizes the parameters of the function `suggest_timevarying_gate_sigma(...)`, rather than the sigma values directly.<br>

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

def exp_2_3(. . .) -> tuple[float, float]:
        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                acc_gate_sigma: float = trial.suggest_float("acc_gate_sigma", acc_gate_sigma_candidate[0], acc_gate_sigma_candidate[1])
                gyro_gate_sigma: float = trial.suggest_float("gyro_gate_sigma", gyro_gate_sigma_candidate[0], gyro_gate_sigma_candidate[1])

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
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                p: int = trial.suggest_int("p", p_candidate[0], p_candidate[1])
                win: float = trial.suggest_float("win_s", win_s_candidate[0], win_s_candidate[1], log=True)
                update_ratio: float = trial.suggest_float("update_ratio", update_ratio_candidate[0], update_ratio_candidate[1])
                update: float = win * update_ratio
                ema: float = trial.suggest_float("ema_alpha", ema_candidate[0], ema_candidate[1])

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

|         |  tau   |         K         |       σ_gyro       |        σ_acc       |
|:-------:|-------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  3.96  | 0.002521892164599 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  3.98  | 0.002511609757989 |  inf (not applied) |          1.4138655 |
| exp 2-3 |  3.99  | 0.002504218546585 |          0.1114052 |         20.0266970 |
| exp 2-4 |  2.85  | 0.003503313069360 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-13 20:36:20.699

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301

[END] 2026-03-13 20:36:21.799
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |
| exp 2-1 | <ul><li>0.33986 rad</li><li>19.47266 deg</li></ul> | <ul><li>0.51315 rad</li><li>29.40161 deg</li></ul> |
| exp 2-2 | <ul><li>0.33856 rad</li><li>19.39819 deg</li></ul> | <ul><li>0.51119 rad</li><li>29.28925 deg</li></ul> |
| exp 2-3 | <ul><li>0.33841 rad</li><li>19.38932 deg</li></ul> | <ul><li>0.51083 rad</li><li>29.26866 deg</li></ul> |
| exp 2-4 | <ul><li>0.33878 rad</li><li>19.41076 deg</li></ul> | <ul><li>0.51106 rad</li><li>29.28171 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-13 20:37:53.656
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0020530457796725846 1.401103619494298 0.3385623576830154 0.5111937635910532

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.1176308583223846 80.2773240575269 19.398194197235362 29.289245167175714

[END] 2026-03-13 20:41:21.662




[START] 2026-03-13 20:41:21.683
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0026687915359229707 1.4004505302520731 0.3384075244465869 0.5108344547068617

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.15291049140862284 80.239904800302 19.389322906259657 29.268658284569987

[END] 2026-03-13 20:45:35.625




[START] 2026-03-13 20:45:35.648
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.002701988907404848 1.4008955376132286 0.3387816686997262 0.5110622356955252

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.1548125606854624 80.26540184394847 19.410759792893614 29.281709173873722

[END] 2026-03-13 20:51:17.763




[START] 2026-03-13 20:51:19.490

best: exp2-3

[END] 2026-03-13 20:51:19.495
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Estimated gravity and linear acceleration are compared against reference signals as a sanity check.<br>

<br>

Gravity direction error remains low (mean/p90 ≈ 1.34° / 2.26°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 10.23° / 21.31°).<br>

<br>

These checks support that Experiment 2 is physically consistent and the orientation error comparisons are meaningful.<br>

<br>

```
[START] 2026-03-13 20:51:26.176

[Gravity]
RMSE norm: 0.26587589801291317

Gravity est/ref angle error in rad — min/max/mean/p90
9.496351035476002e-05 0.09487717993307279 0.023467824095431702 0.039529867883221555

Gravity est/ref angle error in deg — min/max/mean/p90
0.0054410083510746405 5.436061982268377 1.3446072750236555 2.2648945944183363


[Linear accel]
RMSE norm: 0.7652721423726998

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0006574962648661179 3.078695623870781 0.1784670038909429 0.3719501610105884

Linear accel est/ref angle error in deg — min/max/mean/p90
0.03767176102244427 176.3962656531917 10.22540610529587 21.311174415118142
. . .
[END] 2026-03-13 20:51:26.672
```
<br>

#### [Observation]

- Accelerometer correction improves over gyro-only, confirming that gravity-based tilt correction is beneficial
- With Optuna-based joint tuning, the best result is exp 2-3 rather than exp 2-4
- This suggests that for this dataset, fixed jointly tuned gyro/acc gating is slightly better than a more adaptive time-varying schedule
- The gain is modest but consistent in both mean and p90 error, suggesting better suppression of occasional bad corrections without over-attenuating useful ones

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

|         |  tau   |         K         |       σ_gyro       |        σ_acc       |
|:-------:|-------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  0.32  | 0.030621685926523 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  0.33  | 0.030621685926523 |  inf (not applied) |         21.8110573 |
| exp 2-3 |  0.33  | 0.030621685926523 |          3.8147143 |         21.8110573 |
| exp 2-4 |  2.38  | 0.004192651300694 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-13 20:53:31.115

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874

[END] 2026-03-13 20:53:32.105
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |
| exp 2-1 | <ul><li>0.16965 rad</li><li>9.72036 deg</li></ul> | <ul><li>0.32722 rad</li><li>18.74848 deg</li></ul> |
| exp 2-2 | <ul><li>0.17033 rad</li><li>9.75919 deg</li></ul> | <ul><li>0.32728 rad</li><li>18.75152 deg</li></ul> |
| exp 2-3 | <ul><li>0.18372 rad</li><li>10.52661 deg</li></ul> | <ul><li>0.33525 rad</li><li>19.20822 deg</li></ul> |
| exp 2-4 | <ul><li>0.27834 rad</li><li>15.94769 deg</li></ul> | <ul><li>0.43657 rad</li><li>25.01365 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-13 20:53:33.824
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0014567112071040497 0.6559884332313202 0.16965223911367894 0.32722263712136174

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.08346340413646963 37.58536863355405 9.72035728615807 18.74847606819489

[END] 2026-03-13 20:56:18.437




[START] 2026-03-13 20:56:18.458
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0025501833910737417 0.6562536601170682 0.17033002368757355 0.32727579437088

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.1461147452928857 37.60056501472081 9.759191481661304 18.75152175424281

[END] 2026-03-13 21:02:47.675




[START] 2026-03-13 21:02:47.696
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0048145114886251165 0.6310042690493115 0.1837240882578165 0.33524673913803477

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.27585118871546643 36.15388147126303 10.52661485206193 19.208223248132665

[END] 2026-03-13 21:10:53.528




[START] 2026-03-13 21:10:53.547
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.004083158345116812 0.6598162171077984 0.2783396728942163 0.4365705029083013

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.23394774025881496 37.80468449456447 15.947688527890474 25.013647276549495

[END] 2026-03-13 21:21:22.463




[START] 2026-03-13 21:21:26.665

best: exp2-1

[END] 2026-03-13 21:21:26.671
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Gravity direction error remains low (mean/p90 ≈ 3.07° / 5.18°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 14.61° / 29.54°).<br>

<br>

```
[START] 2026-03-13 21:21:37.889

[Gravity]
RMSE norm: 0.6514719385309286

Gravity est/ref angle error in rad — min/max/mean/p90
0.0003908888676714563 0.38395432284377373 0.05349559243639937 0.09033607145073447

Gravity est/ref angle error in deg — min/max/mean/p90
0.022396282376222175 21.998962224751686 3.0650716691576525 5.175875631919332


[Linear accel]
RMSE norm: 0.8326281426956755

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0010549179286515685 3.0958808687986137 0.2550339480877074 0.5155392527416404

Linear accel est/ref angle error in deg — min/max/mean/p90
0.060442345044417777 177.3809076574551 14.612368857984167 29.538223355424247
. . .
[END] 2026-03-13 21:21:38.720
```
<br>

#### [Observation]

- Accelerometer correction alone yields the dominant improvement over gyro-only
- The best result is exp 2-1, meaning no gating is needed for this dataset
- Both fixed and time-varying gating slightly degrade performance, consistent with over-suppressing valid accel corrections
- This dataset acts as a negative control: adaptive gating is not universally beneficial, and for this sequence, “no gating” is the best configuration when the accel reliability proxy does not need to reject updates

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

|         |  tau   |         K         |       σ_gyro       |        σ_acc       |
|:-------:|-------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  0.33  | 0.030621685926523 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  0.33  | 0.030621685926523 |  inf (not applied) |          5.8527864 |
| exp 2-3 |  0.33  | 0.030621685926523 |          2.9182227 |          5.8527864 |
| exp 2-4 |  0.22  | 0.044943586381469 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-13 21:22:58.527

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786

[END] 2026-03-13 21:22:59.416
```

<br>

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| exp 2-1 | <ul><li>0.33687 rad</li><li>19.30136 deg</li></ul> | <ul><li>0.56227 rad</li><li>32.21544 deg</li></ul> |
| exp 2-2 | <ul><li>0.26177 rad</li><li>14.99855 deg</li></ul> | <ul><li>0.40604 rad</li><li>23.26442 deg</li></ul> |
| exp 2-3 | <ul><li>0.24070 rad</li><li>13.79096 deg</li></ul> | <ul><li>0.39439 rad</li><li>22.59707 deg</li></ul> |
| exp 2-4 | <ul><li>0.22967 rad</li><li>13.15924 deg</li></ul> | <ul><li>0.45485 rad</li><li>26.06074 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-13 21:23:01.472
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.007372901990831988 0.9060120450830597 0.3368722815697167 0.5622654170401866

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.4224361668382753 51.91066637127579 19.301359968887475 32.21543536256581

[END] 2026-03-13 21:26:50.789




[START] 2026-03-13 21:26:50.805
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.004190423138418297 0.7491750581956197 0.2617740843012642 0.4060407873669706

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.2400935602053332 42.92456895107684 14.998550216364258 23.26442342629629

[END] 2026-03-13 21:35:47.849




[START] 2026-03-13 21:35:47.866
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.006448084704273436 0.7709568035636727 0.24069772632614023 0.3943932826850462

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.3694480394977294 44.17257103109491 13.790963856882762 22.597070566163154

[END] 2026-03-13 21:46:41.704




[START] 2026-03-13 21:46:41.723
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.0030553382057283093 0.8848460029354784 0.22967207907273088 0.45484578146167504

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.17505798417330576 50.69794148722337 13.159240802862398 26.06074360708376

[END] 2026-03-13 22:01:21.613




[START] 2026-03-13 22:01:26.982

best: exp2-3

[END] 2026-03-13 22:01:26.990
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Gravity direction error remains low (mean/p90 ≈ 2.75° / 4.50°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 11.88° / 23.93°).<br>

<br>

```
[START] 2026-03-13 22:01:42.403

[Gravity]
RMSE norm: 0.5460307231145585

Gravity est/ref angle error in rad — min/max/mean/p90
0.00020554934389784165 0.2799272083382073 0.04805769903184925 0.07862645299527786

Gravity est/ref angle error in deg — min/max/mean/p90
0.011777109887029468 16.038647608658582 2.7535033276349044 4.504963914713171


[Linear accel]
RMSE norm: 0.7917494979296027

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0007656885086786862 2.904607563155329 0.20734425512456875 0.41770524267912995

Linear accel est/ref angle error in deg — min/max/mean/p90
0.04387071996895483 166.42175451057906 11.879950724921581 23.932747486001976
. . .
[END] 2026-03-13 22:01:43.637

```
<br>

#### [Observation]

- This dataset is a strong “gating helps” case
- Both accel-only and joint gating reduce error substantially relative to ungated correction
- The best result is exp 2-3, suggesting that fixed jointly optimized gyro/acc gating provides the best trade-off for this dataset
- Time-varying sigma still performs well, but the Optuna result shows that a carefully tuned fixed-gating configuration can outperform the adaptive schedule on this sequence

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

|         |  tau   |         K         |       σ_gyro       |        σ_acc       |
|:-------:|-------:|------------------:|-------------------:|-------------------:|
| exp 2-1 |  3.96  | 0.002537040143494 |  inf (not applied) |  inf (not applied) |
| exp 2-2 |  0.54  | 0.018787320908048 |  inf (not applied) |          4.3770050 |
| exp 2-3 |  0.33  | 0.030805617919561 |          1.6657816 |          4.3927886 |
| exp 2-4 |  1.56  | 0.006459709761109 | time-varying sigma | time-varying sigma |

** Note: suggested sigmas are reported for reference. the selected run may disable gating (σ=inf) when it does not improve the score<br>

<br>

```
[START] 2026-03-13 22:03:04.826

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965

[END] 2026-03-13 22:03:05.740
```

#### [Metrics]

|         |  Mean error  |  p90 error   |
|:-------:|-------------:|-------------:|
| exp 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| exp 2-1 | <ul><li>0.86428 rad</li><li>49.51985 deg</li></ul> | <ul><li>2.01278 rad</li><li>115.32398 deg</li></ul> |
| exp 2-2 | <ul><li>0.75918 rad</li><li>43.49773 deg</li></ul> | <ul><li>1.77312 rad</li><li>101.59235 deg</li></ul> |
| exp 2-3 | <ul><li>0.77234 rad</li><li>44.25192 deg</li></ul> | <ul><li>1.80136 rad</li><li>103.21053 deg</li></ul> |
| exp 2-4 | <ul><li>0.87627 rad</li><li>50.20653 deg</li></ul> | <ul><li>2.02915 rad</li><li>116.26161 deg</li></ul> |

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-13 22:03:08.274
. . .
[exp 2-1] Gyro+Acc angle error in rad — min/max/mean/p90
0.0018930921712486044 3.1415478697090844 0.8642844509787638 2.012783220058988

[exp 2-1] Gyro+Acc angle error in deg — min/max/mean/p90
0.10846619164180232 179.99743407264518 49.51985133986466 115.32398358413164

[END] 2026-03-13 22:30:15.767




[START] 2026-03-13 22:30:15.785
. . .
[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0029478255158607186 3.139350949771503 0.759178684274718 1.7731209809369155

[exp 2-2] Gyro+Acc+Gating(Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.16889796079979388 179.87155983229363 43.49773450523618 101.59234877378175

[END] 2026-03-13 23:35:36.918




[START] 2026-03-13 23:35:36.939
. . .
[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in rad — min/max/mean/p90
0.0011360053884325936 3.1359210156364 0.7723417233711358 1.801363633780796

[exp 2-3] Gyro+Acc+Gating(Gyro/Acc) — fixed sigma angle error in deg — min/max/mean/p90
0.06508831426130733 179.67503908234434 44.25192109102662 103.21053358398926

[END] 2026-03-14 00:55:23.840




[START] 2026-03-14 00:55:23.859
. . .
[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in rad — min/max/mean/p90
0.002744527635666205 3.1412745144228587 0.8762691695093098 2.0291479350369532

[exp 2-4] Gyro+Acc+Gating(Gyro/Acc) — time varying sigma angle error in deg — min/max/mean/p90
0.15724985028069202 179.98177196843685 50.206525130317175 116.26161268530356

[END] 2026-03-14 02:41:47.986




[START] 2026-03-14 02:42:18.391

best: exp2-2

[END] 2026-03-14 02:42:18.427
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp2]

Gravity direction error remains low (mean/p90 ≈ 2.17° / 4.97°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 28.50° / 71.04°).<br>

<br>

```
[START] 2026-03-14 02:43:47.914

[Gravity]
RMSE norm: 0.5362058049520813

Gravity est/ref angle error in rad — min/max/mean/p90
8.36705176921966e-06 0.5051207364322517 0.03780843846066841 0.08682412742298334

Gravity est/ref angle error in deg — min/max/mean/p90
0.000479396753343755 28.941286342108064 2.166263953776399 4.974656061243018


[Linear accel]
RMSE norm: 0.7885668587209126

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00024263020163022123 3.139339013847217 0.49734893556753823 1.2399683704252757

Linear accel est/ref angle error in deg — min/max/mean/p90
0.013901686535819862 179.87087595420743 28.495994953343857 71.04495435508258
. . .
[END] 2026-03-14 02:43:55.710

```
<br>

#### [Observation]

- Very long, uncontrolled motion introduces large error plateaus/spikes that dominate both mean and tail metrics
- Even on this long and difficult sequence, gated accel correction provides a meaningful gain over gyro-only and ungated gyro+acc
- The best result is exp 2-2, meaning accel-only fixed gating performs best
- Adding gyro gating or switching to time-varying sigma does not help further, suggesting that, in this dataset, accel reliability is a more informative proxy than gyro-norm confidence for the dominant failure segments
- Remaining error may involve yaw/heading drift and unmodeled effects (e.g. mounting/frame mismatch), in addition to limitations of the current accel-magnitude proxy

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-2-data-sum"></a>

| Dataset | best   | exp 1-2 Mean  | best Mean     | exp 1-2 p90   | best p90      | best grav Mean | best acc Mean  |
|:--------|-------:|--------------:|--------------:|--------------:|--------------:|---------------:|---------------:|
| data 01 | exp2-3 | 22.40684 deg  | 19.38932 deg  | 32.44667 deg  | 29.26866 deg  | 1.34461 deg    | 10.22541 deg   |
| data 02 | exp2-1 | 21.99183 deg  | 9.72036 deg   | 31.17450 deg  | 18.74848 deg  | 3.06507 deg    | 14.61237 deg   |
| data 03 | exp2-3 | 30.81266 deg  | 13.79096 deg  | 46.56837 deg  | 22.59707 deg  | 2.75350 deg    | 11.87995 deg   |
| data 04 | exp2-2 | 50.88316 deg  | 43.49773 deg  | 116.70718 deg | 101.59235 deg | 2.16626 deg    | 28.49599 deg   |

** best = minimum error (0.8 * mean + 0.2 * p90) across exp 2 (per dataset)<br>

<br>

Across all datasets:<br>

- Adding accelerometer correction consistently reduces error relative to gyro-only across these datasets
- In the longer sequences, the benefit often becomes more visible, although the magnitude of the gain remains dataset-dependent
- Gating is strongly dataset-dependent. It helps when linear acceleration frequently violates the gravity assumption, but may be marginal or unnecessary when accelerometer measurements are already sufficiently consistent

<br>

Datasets where gating helps (data 01, data 03, data 04):<br>

- A gated configuration outperforms ungated gyro+acc correction
- This suggests that accel reliability is not uniform over time and that soft gating can suppress incorrect tilt updates during dynamic motion
- In data 01 and data 03, joint gyro/acc gating performs best
- In data 04, accel-only fixed gating performs best, suggesting that the gyro gate is not always useful even when accel gating is

<br>

Dataset where gating is unnecessary or harmful (data 02):<br>

- The best configuration is exp 2-1, i.e. gyro+acc without gating
- This indicates that accelerometer correction is already reliable enough in this motion pattern
- Additional gating appears to suppress useful corrections more than it filters harmful ones

<br>

Overall interpretation:<br>

- Across these datasets, accelerometer correction provides the main gain in roll/pitch stabilization
- Gating provides a secondary gain only when the chosen proxy meaningfully separates reliable from unreliable updates
- The fact that different datasets prefer different gating structures is itself an important experimental result

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-2-conclusion"></a>

Experiment 2 confirms:<br>

1. Accelerometer correction improves roll/pitch by continuously correcting gyro drift
2. Gating can further improve performance, but only when the reliability proxy aligns with the actual failure modes of the dataset
3. The usefulness of gating is not universal: some datasets benefit from fixed or joint gating, while others perform best with no gating at all
4. The best configuration is dataset-dependent, which suggests that robustness should be evaluated across multiple motion regimes rather than inferred from a single sequence

<br>

Next steps:<br>
- Experiment 3: add magnetometer correction to constrain yaw drift

<br>
<br>
<br>
<br>
