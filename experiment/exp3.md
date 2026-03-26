
 * [Experiment 3](#exp-3) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-3-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Method](#exp-3-method) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Magnetometer Correction](#exp-3-method-mag-correction) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Generate Horizontal Magnetic Reference](#exp-3-method-mag-ref) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Suggest Gate Sigma](#exp-3-method-sigma) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Hyperparameter Search with Optuna](#exp-3-method-optuna) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Results](#exp-3-res) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 01 — 5 min](#exp-3-res-data-01) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 02 — 9 min](#exp-3-res-data-02) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 03 — 13 min](#exp-3-res-data-03) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 04 — 96 min](#exp-3-res-data-04) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Cross-dataset Summary](#exp-3-data-sum) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Conclusion](#exp-3-conclusion) <br>


<br>
<br>

## Experiment 3  — Gyro + Accelerometer + Magnetometer <a name="exp-3"></a>

### Goal <a name="exp-3-goal"></a>

This experiment evaluates whether magnetometer-based heading correction further improves orientation accuracy beyond the best gyro+accelerometer configuration from experiment 2.<br>

<br>

Unlike accelerometer correction, which mainly stabilizes roll/pitch through gravity alignment,<br>
magnetometer correction can constrain heading (yaw) drift by providing a horizontal reference direction.<br>

<br>

Six runs are compared (same dataset, same trimmed start):<br>

- [best exp 2] Gyro + Acc (+gating)
- [exp 3-1] Gyro + Acc + Mag, without gating
- [exp 3-2] Gyro + Acc + Mag with Magnetometer innovation gating
- [exp 3-3] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating — fixed sigma
- [exp 3-4] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating + Magnetometer innovation gating — fixed sigma
- [exp 3-5] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating — time-varying sigma
- [exp 3-6] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating + Magnetometer innovation gating — time-varying sigma

<br>

** `best exp 2` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift), evaluated on the same trimmed segment<br>

<br>

Key hypothesis:<br>

Adding magnetometer correction is expected to reduce heading drift and may improve total orientation accuracy.<br>

<br>
<br>
<br>
<br>

### Method <a name="exp-3-method"></a>

### Magnetometer Correction <a name="exp-3-method-mag-correction"></a>

At each step:<br>

- propagate orientation using gyro integration
- compute gravity-based tilt correction from the accelerometer
- project the measured magnetic vector onto the horizontal plane orthogonal to gravity
- predict the horizontal magnetic direction in the body frame
- compute a heading-related (yaw) correction axis from the mismatch between measured and predicted horizontal magnetic directions
- combine accelerometer and magnetometer correction terms into a single small-angle quaternion update

<br>

The total correction axis is defined as:<br>

```py
# pipeline.py, integrate_gyro_acc_mag(. . .)

	e_axis: Vec3 = (weight_acc[i] * weight_gyro[i] * e_axis_acc
			+ mag_gain * weight_mag[i] * e_axis_mag)
	dq_corr: Quat = small_angle_correction_quat(K, e_axis)
```

<br>

** `weight_acc`: Accel magnitude residual-based confidence (approaches to 1 when `| ||a|| - g0 |` is small)<br>
** `weight_gyro`: Gyro norm-based confidence (approaches to 1 when `||w||` is small)<br>
** `e_axis_acc`: Tilt correction axis from gravity alignment<br>

<br>

** `mag_gain`: Relative weight of the magnetometer correction term<br>
** `weight_mag`: Magnetometer confidence term based on magnitude consistency and further modulated by innovation gating (approaches to 1 when `| ||m|| - m0 |` is small and the magnetometer correction remains within the accepted range)<br>
** `e_axis_mag`: Heading correction axis from horizontal magnetic alignment<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# pipeline.py

def generate_m_ref_world_h_unit(. . .) -> Vec3:
        . . .

def calc_mag_gating(m0: float, mag_sigma: float, m_meas: Vec3) -> float:
        . . .

def calc_mag_innovation_gating(e_axis_mag: Vec3, mag_err_sigma: float) -> float:
        . . .

def calc_mag_err_axis(. . .) -> Vec3:
        m_body_h: Vec3 = m_unit - np.dot(m_unit, g_pred) * g_pred
        . . .
        m_pred: Vec3 = libq.rotate_world_to_body(q_pred, m_world_h_unit)
        m_pred_h: Vec3 = m_pred - np.dot(m_pred, g_pred) * g_pred
        . . .
        mag_err_axis: Vec3 = np.cross(m_body_h_unit, m_pred_h_unit)
        mag_err_axis = np.dot(mag_err_axis, g_pred) * g_pred
        return mag_err_axis

def integrate_gyro_acc_mag(. . .) -> tuple[. . .]:
	. . .
        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])
                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                . . .
                weight_acc[i] = calc_acc_gating(g0, acc_gate_sigma[i], a_meas, g_pred)
                weight_gyro[i] = calc_gyro_gating(gyro_gate_sigma[i], w_avg[i])
                weight_mag[i] = calc_mag_gating(m0, mag_gate_sigma[i], m_meas)

                e_axis_acc: Vec3 = np.cross(g_pred, a_unit)
                e_axis_mag: Vec3 = calc_mag_err_axis(q_pred, g_pred, m_unit, m_world_h_unit)

                weight_mag_innov: float = calc_mag_innovation_gating(e_axis_mag, mag_err_sigma)
                weight_mag[i] = weight_mag[i] * weight_mag_innov

                e_axis: Vec3 = (weight_acc[i] * weight_gyro[i] * e_axis_acc
                                + mag_gain * weight_mag[i] * e_axis_mag)
                dq_corr: Quat = small_angle_correction_quat(K, e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q
		. . .
        return res, g_body_est, a_lin_est, weight_acc, weight_gyro, weight_mag
```

</details>

<br>
<br>
<br>

### Generate Horizontal Magnetic Reference <a name="exp-3-method-mag-ref"></a>

To compute a stable magnetometer correction, a global horizontal magnetic reference direction is first estimated.<br>

<br>

- Rotate measured magnetic vectors from body frame to world frame using the reference orientation
- Remove the vertical component along gravity
- Normalize the remaining horizontal component
- Average valid samples over an initial window
- Normalize the final result to obtain `m_ref_world_h_unit`

<br>

Only samples with near-gravity accelerometer magnitude and a valid magnetometer norm are used.<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# pipeline.py

def generate_m_ref_world_h_unit(. . .) -> Vec3:
        a_norm: Vec3Batch = np.linalg.norm(a_src, axis=1)
        m_norm: Vec3Batch = np.linalg.norm(m_src, axis=1)

        mask = (np.abs(a_norm - g0) < 0.5) & (m_norm > 0.1)
        mask = mask[:sample_length]
        . . .
        for i in range(min(sample_length, len(mask))):
                . . .
                m_unit: Vec3 = safe_unit_vec3(m_src[i])
                m_world: Vec3 = libq.rotate_body_to_world(q_ref[i], m_unit)
                m_world_h: Vec3 = m_world - np.dot(m_world, g_world_unit) * g_world_unit
                n: float = np.linalg.norm(m_world_h)
                if n > EPS:
                        res += (m_world_h / n)
                        cnt += 1
        . . .
        return safe_unit_vec3(res)
```

</details>

<br>
<br>
<br>

### Magnetometer gating <a name="exp-3-method-mag-gating"></a>

#### [Norm-based gating]

Norm-based magnetometer gating reduces trust when the measured field magnitude deviates from its expected nominal value.<br>

<br>

A Gaussian-like confidence weight is used.<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# pipeline.py

def calc_mag_gating(m0: float, mag_sigma: float, m_meas: Vec3) -> float:
        . . .
        dev: float = abs(float(np.linalg.norm(m_meas)) - m0)
        weight_mag : float = np.exp(-0.5 * (dev / mag_sigma) ** 2)
        return weight_mag
```

</details>

<br>
<br>

#### [Innovation-based gating]

Innovation-based gating evaluates whether the correction implied by the magnetometer is consistent with the current estimate.<br>

<br>

Instead of a purely binary decision, a hybrid strategy is used that combines hard rejection with soft attenuation.<br>

<br>

This gating suppresses unreliable heading updates when the measured magnetic direction is inconsistent with the predicted one,<br>
while still allowing partial correction when the inconsistency is moderate:<br>

- If the innovation is very large, the update is fully rejected (hard gating)
- If the innovation is moderately large, the update is down-weighted using a smooth exponential decay (soft gating)
- If the innovation is small, the update is fully trusted

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# pipeline.py

def calc_mag_innovation_gating(e_axis_mag: Vec3, mag_err_sigma: float) -> float:
        . . .
        e_axis_norm: float = float(np.linalg.norm(e_axis_mag))
        if e_axis_norm > 2:
                return 0
        elif e_axis_norm > 1.2 * mag_err_sigma:
                return np.exp(-0.06 * (e_axis_norm / mag_err_sigma) ** 2)
        return 1
```

</details>

<br>

#### [Difference]

- Norm-based gating evaluates magnitude consistency (sensor-level check)
- Innovation-based gating evaluates directional consistency of the correction (filter-level check)

<br>
<br>
<br>

### Suggest Gate Sigma <a name="exp-3-method-sigma"></a>

As in Experiment 2, fixed sigma values are first suggested from robust data statistics,<br>
and time-varying sigma schedules are optionally generated using percentile estimates over a sliding window with EMA smoothing.<br>

<br>

For the magnetometer, the residual is computed as the deviation from the global magnetometer norm baseline.<br>

<br>

#### [Fixed sigma]

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# autotune.py

def suggest_fixed_mag_gate_sigma(. . .) -> float:
        . . .
        m_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(m, axis=1))
        mag_resid: ScalarBatch = np.abs(m_norm - np.median(m_norm))
        mag_sigma: float = max(sigma_floor, float(np.percentile(mag_resid, p_mag)))
        return mag_sigma

def suggest_fixed_gate_sigma(. . .) -> tuple[float, float, float]:
        . . .
        mag_sigma: float = suggest_fixed_mag_gate_sigma(m, m0, p_mag, sigma_floor)
        . . .
        return gyro_sigma, acc_sigma, mag_sigma
```

</details>

<br>
<br>

#### [Time-varying sigma]

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# autotune.py

def suggest_timevarying_gate_sigma(. . .) -> tuple[ScalarBatch, ScalarBatch, ScalarBatch]:
	. . .
        window_size: int = max(1, int(np.ceil(win_s / max(dt_median, EPS))))
        update_period: int = max(1, int(np.ceil(update_s / max(dt_median, EPS))))
	. . .
        mag_sigma = max(sigma_floor, float(np.percentile(mag_resid_init, p_mag)))

        for i in range(len(dt)):
                if i % update_period == 0:
                        low: int = max(0, i - window_size)
                        high: int = i + 1

                        . . .
			mag_sigma = (1 - ema_alpha) * mag_sigma + ema_alpha * mag_tmp
                . . .
                batch_mag_sigma[i] = mag_sigma
        return batch_gyro_sigma, batch_acc_sigma, batch_mag_sigma
```

</details>

<br>
<br>
<br>

### Hyperparameter Search with Optuna <a name="exp-3-method-optuna"></a>

#### [Optimization target]

| exp | trial |             Target ∈ Range (min, max)             |
|:---:|------:|:--------------------------------------------------|
| 3-1 |    20 | <ul><li>tau ∈ (0.1, 4)</li><li>mag_gain ∈ (0.01, 10)</li></ul> |
| 3-2 |    20 | <ul><li>tau ∈ 3-1_best*(0.9, 1.1)</li><li>mag_gain ∈ 3-1_best*(0.7, 1.3)</li><li>mag_err_sigma ∈ (0.01, 2)</li></ul> |
| 3-3 |    30 | <ul><li>tau ∈ 3-2_best*(0.9, 1.1)</li><li>mag_gain ∈ 3-2_best*(0.7, 1.3)</li><li>acc_gate_sigma ∈ suggested_acc*(0.01, 10)</li><li>gyro_gate_sigma ∈ suggested_gyro*(0.1, 10)</li><li>mag_gate_sigma ∈ suggested_mag*(0.01, 10)</li></ul> |
| 3-4 |    20 | <ul><li>tau ∈ 3-3_best*(0.9, 1.1)</li><li>mag_gain ∈ 3-3_best*(0.7, 1.3)</li><li>acc_gate_sigma ∈ 3-3_best*(0.7, 1.3)</li><li>gyro_gate_sigma ∈ 3-3_best*(0.7, 1.3)</li><li>mag_gate_sigma ∈ 3-3_best*(0.7, 1.3)</li><li>mag_err_sigma ∈ 3-2_best*(0.5, 1.5)</li></ul> |
| 3-5 |    40 | <ul><li>tau ∈ 3-4_best*(0.9, 1.1)</li><li>mag_gain ∈ 3-4_best*(0.7, 1.3)</li><li>percentile `p` ∈ (50, 80)</li><li>sliding window size `win_s` ∈ (5, 10)</li><li>update ratio `update_ratio` ∈ (0.1, 0.5)</li><li>EMA factor `ema_alpha` ∈ (0.02, 0.2)</li></ul> |
| 3-6 |    40 | <ul><li>tau ∈ 3-5_best*(0.9, 1.1)</li><li>mag_gain ∈ 3-5_best*(0.7, 1.3)</li><li>percentile `p` ∈ (50, 80)</li><li>sliding window size `win_s` ∈ (5, 10)</li><li>update ratio `update_ratio` ∈ (0.1, 0.5)</li><li>EMA factor `ema_alpha` ∈ (0.02, 0.2)</li><li>mag_err_sigma ∈ (0.01, 2)</li></ul> |

<br>

** 3-x_best: Optimal value selected from experiment 3-x<br>
** x_gate_sigma: fixed gate sigma estimated from robust percentiles of the data<br>
** For exp 3-5 and exp 3-6, Optuna optimizes the parameters of the function `suggest_timevarying_gate_sigma(...)`, rather than the sigma values directly<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# optuna_exp_3.py

def exp_3_1(. . .) -> tuple[float, float, float]:
	. . .

def exp_3_2(. . .) -> tuple[float, ...]:
	. . .

def exp_3_3(. . .) -> tuple[float, ...]:
	. . .

def exp_3_4(. . .) -> tuple[float, ...]:
        def objective(trial):
                tau: float = trial.suggest_float("tau", . . .)
                mag_gain: float = trial.suggest_float("mag_gain", . . .)
                acc_gate_sigma: float = trial.suggest_float("acc_gate_sigma", . . .)
                gyro_gate_sigma: float = trial.suggest_float("gyro_gate_sigma", . . .)
                mag_gate_sigma: float = trial.suggest_float("mag_gate_sigma", . . .)
                mag_err_sigma: float = trial.suggest_float("mag_err_sigma", . . .)

                score: float = calc_score_quasi_ori(. . .)
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
	. . .
        return [. . .]

def exp_3_5(. . .) -> tuple[Any, ...]:
	. . .

def exp_3_6(. . .) -> tuple[Any, ...]:
	. . .
        def objective(trial):
                tau: float = trial.suggest_float("tau", . . .)
                mag_gain: float = trial.suggest_float("mag_gain", . . .)
                mag_err_sigma: float = trial.suggest_float("mag_err_sigma", . . .)
                p: int = trial.suggest_int("p", . . .)
                win: float = trial.suggest_float("win_s", . . .)
                update_ratio: float = trial.suggest_float("update_ratio", . . .)
                update: float = win * update_ratio
                ema: float = trial.suggest_float("ema_alpha", . . .)

                [. . .] = suggest_timevarying_gate_sigma(. . .)
                score: float = calc_score_quasi_ori(. . .)
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
	. . .
        [timevarying_gyro_sigma,
         timevarying_acc_sigma,
         timevarying_mag_sigma] = suggest_timevarying_gate_sigma(. . .)
        . . .
        return [. . .]
```

</details>

<br>
<br>
<br>
<br>

### Results <a name="exp-3-res"></a>

Each plot compares:<br>

- blue: exp 1-2
- orange: best exp 2
- green: exp 3-1
- red: exp 3-2
- purple: exp 3-3
- brown: exp 3-4
- pink: exp 3-5
- gray: exp 3-6

<br>
<br>

### Dataset 01 — 5 min <a name="exp-3-res-data-01"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data01_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data01_exp3_02.png" width="952" height="631">

#### [Chosen parameters]

- quasi-static: (2523, 3540, 1017)
- suggested σ_gyro: 0.4479197
- suggested σ_acc : 2.9085688
- suggested σ_mag : 6.8504558

<br>

| exp |  tau   |      K      | mag_gain |     σ_acc    |    σ_gyro    |     σ_mag    | σ_mag_err |
|:---:|-------:|------------:|---------:|-------------:|-------------:|-------------:|----------:|
| 3-1 |  3.93  | 0.002542621 | 3.577178 |          inf |          inf |          inf |       inf |
| 3-2 |  4.28  | 0.002334764 | 3.519930 |          inf |          inf |          inf | 1.6224342 |
| 3-3 |  4.57  | 0.002186687 | 4.397713 |   20.4765482 |    1.1233809 |   25.3407412 |       inf |
| 3-4 |  4.77  | 0.002096190 | 5.678210 |   15.8903128 |    0.9994370 |   18.0364040 | 1.5374059 |
| 3-5 |  5.17  | 0.001932617 | 4.930732 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  5.16  | 0.001937734 | 5.600152 | time-varying | time-varying | time-varying | 1.9951554 |

<br>

** `σ = inf` means gating not applied<br>

<br>

| exp |    p    |    win_s     | update_ratio |  ema_alpha   |
|:---:|--------:|-------------:|-------------:|-------------:|
| 3-5 |   76    | 5.777591     | 0.395468     | 0.050896     |
| 3-6 |   80    | 5.046138     | 0.265996     | 0.025141     |

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 16:47:10.126

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301
Suggested mag_sigma:  6.850455808768257

[END] 2026-03-26 16:47:11.086
```

</details>

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |
| b2  | <ul><li>0.33782 rad</li><li>19.35529 deg</li></ul> | <ul><li>0.51093 rad</li><li>29.27439 deg</li></ul> |
| 3-1 | <ul><li>0.04217 rad</li><li>2.41633 deg</li></ul>  | <ul><li>0.08175 rad</li><li>4.68411 deg</li></ul>  |
| 3-2 | <ul><li>0.04137 rad</li><li>2.37029 deg</li></ul>  | <ul><li>0.07968 rad</li><li>4.56528 deg</li></ul>  |
| 3-3 | <ul><li>0.03815 rad</li><li>2.18586 deg</li></ul>  | <ul><li>0.07351 rad</li><li>4.21208 deg</li></ul>  |
| 3-4 | <ul><li>0.03794 rad</li><li>2.17366 deg</li></ul>  | <ul><li>0.07607 rad</li><li>4.35842 deg</li></ul>  |
| 3-5 | <ul><li>0.05958 rad</li><li>3.41390 deg</li></ul>  | <ul><li>0.10931 rad</li><li>6.26327 deg</li></ul>  |
| 3-6 | <ul><li>0.05703 rad</li><li>3.26791 deg</li></ul>  | <ul><li>0.10546 rad</li><li>6.04262 deg</li></ul>  |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 16:47:11.139
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.002818835027033219 0.7172417482362367 0.042172969711376365 0.08175313998327868

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.16150735019264875 41.09492506452112 2.416333173994919 4.68410988298409

[END] 2026-03-26 16:50:30.032




[START] 2026-03-26 16:50:30.047
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.00280806873989446 0.6962182150580377 0.04136929699102485 0.07967920247718345

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.1608904873785719 39.89036534295706 2.37028611900898 4.565282016910945

[END] 2026-03-26 16:53:51.713




[START] 2026-03-26 16:53:51.728
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0022300678365621764 0.5431561440288503 0.038150425089703295 0.07351459839444256

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.12777347506288297 31.12055466945299 2.1858583442700037 4.2120762206007765

[END] 2026-03-26 16:59:42.931




[START] 2026-03-26 16:59:42.948
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0017333672375821516 0.5303552091541307 0.037937464957498106 0.07606885323359422

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.09931462705970755 30.387115127309734 2.1736566274900984 4.358424242685033

[END] 2026-03-26 17:03:45.382




[START] 2026-03-26 17:03:45.400
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002409636697114314 0.7816545685336467 0.05958371638396706 0.1093147442364582

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.13806201290449366 44.785507814097315 3.4138954765058074 6.263273483301096

[END] 2026-03-26 17:11:37.212




[START] 2026-03-26 17:11:37.225
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0017723107976843774 0.7672592415278287 0.05703574444543082 0.10546355981071441

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.10154592869277913 43.96071633195324 3.267907438109914 6.042616869579463

[END] 2026-03-26 17:19:36.843




[START] 2026-03-26 17:19:40.614

best: exp3-4

[END] 2026-03-26 17:19:40.631
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 2</th>
      <th colspan="4">best exp 3</th>
    </tr>
    <tr>
      <th>grav mean</th>
      <th>acc mean</th>
      <th>grav mean</th>
      <th>grav p90</th>
      <th>acc mean</th>
      <th>acc p90</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.98</td>
      <td>8.92</td>
      <td>1.02</td>
      <td>1.93</td>
      <td>8.15</td>
      <td>18.57</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 17:19:49.213

[Gravity]
RMSE norm: 0.20526433541541741

Gravity est/ref angle error in rad — min/max/mean/p90
0.00016459727972550922 0.09293993934479382 0.017872906262809903 0.033646807603579174

Gravity est/ref angle error in deg — min/max/mean/p90
0.009430729447605911 5.325066272658551 1.0240420964919443 1.9278200697737742


[Linear accel]
RMSE norm: 0.7382335700551836

Linear accel est/ref angle error in rad — min/max/mean/p90
0.000719256919690961 2.824066823648598 0.14232560147445011 0.3241060575267027

Linear accel est/ref angle error in deg — min/max/mean/p90
0.04121038588387206 161.80711005798082 8.154656281146918 18.56990921090433
. . .
[END] 2026-03-26 17:19:49.653
```

</details>

#### [Observation]

- Adding magnetometer correction substantially improves orientation accuracy relative to the best experiment 2 result on this dataset
- The best result is exp 3-4, indicating that fixed norm-based gating combined with innovation-based gating provides the best overall trade-off under the selected ranking criterion
- Both fixed-gating variants (exp 3-3, exp 3-4) outperform ungated and innovation-only magnetometer correction
- The additional gain from innovation gating is modest, but on this dataset it improves the final result when combined with norm-based gating
- Time-varying sigma variants remain clearly worse than the best fixed-gating configurations on this sequence

<br>
<br>
<br>

### Dataset 02 — 9 min <a name="exp-3-res-data-02"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data02_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data02_exp3_02.png" width="952" height="631">

#### [Chosen parameters]

- quasi-static: (0, 1635, 1635)
- suggested σ_gyro: 0.6304231
- suggested σ_acc : 2.5176967
- suggested σ_mag : 5.0510618

<br>

| exp |  tau   |      K      | mag_gain |     σ_acc    |    σ_gyro    |     σ_mag    | σ_mag_err |
|:---:|-------:|------------:|---------:|-------------:|-------------:|-------------:|----------:|
| 3-1 |  3.97  | 0.002518346 | 3.724649 |          inf |          inf |          inf |       inf |
| 3-2 |  4.37  | 0.002290369 | 4.124038 |          inf |          inf |          inf | 0.7518797 |
| 3-3 |  4.54  | 0.002202910 | 3.892036 |    1.7562440 |    6.2834979 |   16.2250324 |       inf |
| 3-4 |  4.19  | 0.002387545 | 4.519486 |    1.8465124 |    6.8600581 |   13.1831565 | 0.8853287 |
| 3-5 |  3.79  | 0.002635225 | 5.460352 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  3.56  | 0.002812494 | 6.998810 | time-varying | time-varying | time-varying | 1.5525143 |

<br>

** `σ = inf` means gating not applied<br>

<br>

| exp |    p    |    win_s     | update_ratio |  ema_alpha   |
|:---:|--------:|-------------:|-------------:|-------------:|
| 3-5 |   80    | 9.996655     | 0.194599     | 0.096770     |
| 3-6 |   79    | 9.474137     | 0.339160     | 0.185937     |

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-25 21:23:45.598

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874
Suggested mag_sigma:  5.051061836028509

[END] 2026-03-25 21:23:46.274
```

</details>

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |
| b2  | <ul><li>0.15914 rad</li><li>9.11787 deg</li></ul>  | <ul><li>0.32830 rad</li><li>18.81032 deg</li></ul> |
| 3-1 | <ul><li>0.04564 rad</li><li>2.61517 deg</li></ul>  | <ul><li>0.08512 rad</li><li>4.87730 deg</li></ul>  |
| 3-2 | <ul><li>0.04447 rad</li><li>2.54800 deg</li></ul>  | <ul><li>0.08333 rad</li><li>4.77430 deg</li></ul>  |
| 3-3 | <ul><li>0.03320 rad</li><li>1.90235 deg</li></ul>  | <ul><li>0.08865 rad</li><li>5.07946 deg</li></ul>  |
| 3-4 | <ul><li>0.03020 rad</li><li>1.73052 deg</li></ul>  | <ul><li>0.07193 rad</li><li>4.12132 deg</li></ul>  |
| 3-5 | <ul><li>0.04852 rad</li><li>2.78041 deg</li></ul>  | <ul><li>0.09915 rad</li><li>5.68096 deg</li></ul>  |
| 3-6 | <ul><li>0.04861 rad</li><li>2.78499 deg</li></ul>  | <ul><li>0.10116 rad</li><li>5.79594 deg</li></ul>  |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-25 21:23:46.312
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0014937441061020549 0.3703654566537466 0.04564329581096676 0.08512497175927315

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.08558523295218959 21.220377543695115 2.6151682130355454 4.877301612976674

[END] 2026-03-25 21:28:40.496




[START] 2026-03-25 21:28:40.507
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0009744398208139917 0.3921464219975504 0.044471033910069374 0.08332721312806343

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.05583128912212591 22.468334931615782 2.548002553630142 4.774297630825141

[END] 2026-03-25 21:33:33.667




[START] 2026-03-25 21:33:33.679
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0006318141187635655 0.4940754860093368 0.03320228894818098 0.08865336899765028

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.03620028244192965 28.308440109209954 1.9023510269046275 5.079463883181298

[END] 2026-03-25 21:44:20.840




[START] 2026-03-25 21:44:20.855
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0003528782380093778 0.5061348804237038 0.030203112347491224 0.07193067933840304

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.020218433719950298 28.99939251263682 1.7305108656707113 4.121324343599367

[END] 2026-03-25 21:53:00.866




[START] 2026-03-25 21:53:00.885
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.00309411337562138 0.41506042495390216 0.0485273452437651 0.09915149694582216

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.17727963775808145 23.78121059276503 2.780412073441989 5.680962307399882

[END] 2026-03-25 22:08:27.826




[START] 2026-03-25 22:08:27.842
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.001851477041226652 0.43401756316313544 0.048607184841248534 0.10115831861615807

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.10608182032765627 24.86737460380029 2.7849865454158134 5.795944719345523

[END] 2026-03-25 22:21:16.698




[START] 2026-03-25 22:21:23.237

best: exp3-4

[END] 2026-03-25 22:21:23.260
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 2</th>
      <th colspan="4">best exp 3</th>
    </tr>
    <tr>
      <th>grav mean</th>
      <th>acc mean</th>
      <th>grav mean</th>
      <th>grav p90</th>
      <th>acc mean</th>
      <th>acc p90</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3.38</td>
      <td>15.85</td>
      <td>0.57</td>
      <td>1.05</td>
      <td>6.28</td>
      <td>15.02</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-25 22:21:38.961

[Gravity]
RMSE norm: 0.14073004422147176

Gravity est/ref angle error in rad — min/max/mean/p90
3.125209421636435e-05 0.05890731085375066 0.010029771598682045 0.018406024138380106

Gravity est/ref angle error in deg — min/max/mean/p90
0.001790613099542887 3.375140294385099 0.5746635820846616 1.0545875007450976


[Linear accel]
RMSE norm: 0.4899786985228095

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0001147711116425533 3.0061230149379967 0.10967203797851129 0.26221086721946063

Linear accel est/ref angle error in deg — min/max/mean/p90
0.00657590030714309 172.23816145308973 6.283744906767174 15.023576034140321
. . .
[END] 2026-03-25 22:21:40.016
```

</details>

<br>

#### [Observation]

- Adding magnetometer correction again yields a large improvement over the best experiment 2 result
- The best result is exp 3-4, continuing the same pattern as Dataset 01
- Innovation-only gating (exp 3-2) improves slightly over ungated magnetometer correction, but the largest gain appears when innovation gating is combined with fixed norm-based gating
- The fixed-gating variants outperform the time-varying variants on this dataset
- This suggests that, for this relatively consistent sequence, a fixed gating structure is better matched than a time-varying schedule

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-3-res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data03_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data03_exp3_02.png" width="952" height="631">

#### [Chosen parameters]

- quasi-static: (41487, 43669, 2182)
- suggested σ_gyro: 0.4822681
- suggested σ_acc : 0.6755996
- suggested σ_mag : 5.0989239

<br>

| exp |  tau   |      K      | mag_gain |     σ_acc    |    σ_gyro    |     σ_mag    | σ_mag_err |
|:---:|-------:|------------:|---------:|-------------:|-------------:|-------------:|----------:|
| 3-1 |  3.93  | 0.002542621 | 3.577178 |          inf |          inf |          inf |       inf |
| 3-2 |  4.32  | 0.002311993 | 3.388323 |          inf |          inf |          inf | 0.7612136 |
| 3-3 |  3.96  | 0.002523787 | 3.241114 |    2.0484861 |    1.3120595 |   34.8068405 |       inf |
| 3-4 |  4.09  | 0.002444338 | 3.375010 |    2.3851260 |    1.1860780 |   40.2588639 | 0.7412543 |
| 3-5 |  4.45  | 0.002246526 | 3.298374 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  4.17  | 0.002397647 | 4.227693 | time-varying | time-varying | time-varying | 1.5525143 |

<br>

** `σ = inf` means gating not applied<br>

<br>

| exp |    p    |    win_s     | update_ratio |  ema_alpha   |
|:---:|--------:|-------------:|-------------:|-------------:|
| 3-5 |   80    | 8.373621     | 0.464217     | 0.188064     |
| 3-6 |   79    | 9.474137     | 0.339160     | 0.185937     |

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-25 22:36:24.378

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786
Suggested mag_sigma:  5.098923949491506

[END] 2026-03-25 22:36:25.156
```

</details>

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| b2  | <ul><li>0.24289 rad</li><li>13.91633 deg</li></ul> | <ul><li>0.38963 rad</li><li>22.32396 deg</li></ul> |
| 3-1 | <ul><li>0.04335 rad</li><li>2.48355 deg</li></ul>  | <ul><li>0.07790 rad</li><li>4.46346 deg</li></ul> |
| 3-2 | <ul><li>0.04219 rad</li><li>2.41742 deg</li></ul>  | <ul><li>0.07543 rad</li><li>4.32170 deg</li></ul> |
| 3-3 | <ul><li>0.03418 rad</li><li>1.95857 deg</li></ul>  | <ul><li>0.06214 rad</li><li>3.56029 deg</li></ul> |
| 3-4 | <ul><li>0.03677 rad</li><li>2.10667 deg</li></ul>  | <ul><li>0.06585 rad</li><li>3.77265 deg</li></ul> |
| 3-5 | <ul><li>0.06298 rad</li><li>3.60866 deg</li></ul>  | <ul><li>0.11860 rad</li><li>6.79522 deg</li></ul> |
| 3-6 | <ul><li>0.06038 rad</li><li>3.45952 deg</li></ul>  | <ul><li>0.11016 rad</li><li>6.31160 deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-25 22:36:25.196
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0009205019993881509 0.4253150125622266 0.04334613532077613 0.0779020739570321

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.05274087959829493 24.368755183369174 2.4835506120834188 4.463460053053944

[END] 2026-03-25 22:44:31.294




[START] 2026-03-25 22:44:31.305
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0010724171541030324 0.4038463545977173 0.04219187418047651 0.07542797577246489

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.06144497680753457 23.138691690192868 2.417416320288293 4.321704668977263

[END] 2026-03-25 22:52:58.524




[START] 2026-03-25 22:52:58.541
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0005615014128801123 0.25009170692673555 0.03418342702097173 0.0621387057555901

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.03217166114866312 14.329199298124642 1.9585660975951367 3.56028558420059

[END] 2026-03-25 23:08:04.911




[START] 2026-03-25 23:08:04.927
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.001358626679512281 0.26715531641816487 0.03676829735821092 0.06584523358043846

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.07784357466992681 15.306872105242917 2.1066682585075 3.7726539852122065

[END] 2026-03-25 23:20:05.288




[START] 2026-03-25 23:20:05.307
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0024426687498780323 0.74029348386357 0.06298306263195788 0.11859888418759126

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.13995461011650817 42.41569222641867 3.6086636696193133 6.7952155189098145

[END] 2026-03-25 23:41:01.545




[START] 2026-03-25 23:41:01.559
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0014641836184843284 0.7190657547776307 0.060379996229155096 0.11015816122655236

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.08389154177134513 41.19943294114725 3.4595189509464124 6.311597717203118

[END] 2026-03-26 00:02:29.468




[START] 2026-03-26 00:02:37.681

best: exp3-3

[END] 2026-03-26 00:02:37.704
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 2</th>
      <th colspan="4">best exp 3</th>
    </tr>
    <tr>
      <th>grav mean</th>
      <th>acc mean</th>
      <th>grav mean</th>
      <th>grav p90</th>
      <th>acc mean</th>
      <th>acc p90</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2.79</td>
      <td>11.99</td>
      <td>0.87</td>
      <td>1.27</td>
      <td>6.75</td>
      <td>15.75</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 00:02:58.190

[Gravity]
RMSE norm: 0.1592124639346473

Gravity est/ref angle error in rad — min/max/mean/p90
3.385052875872364e-05 0.040561468702415865 0.015160784556686638 0.02222937333150332

Gravity est/ref angle error in deg — min/max/mean/p90
0.0019394924321610818 2.3240009675004085 0.8686489692052611 1.2736492731158064


[Linear accel]
RMSE norm: 0.5462331177016112

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00011680936816655787 2.897137986395199 0.11781902519210632 0.274874443518591

Linear accel est/ref angle error in deg — min/max/mean/p90
0.006692683803533557 165.99377928747464 6.750532889853216 15.749145509622393
. . .
[END] 2026-03-26 00:02:59.532
```

</details>

<br>

#### [Observation]

- Magnetometer correction substantially improves orientation accuracy relative to the best experiment 2 result on this dataset as well
- The best result is exp 3-3, indicating that fixed norm-based gating provides the best overall result under the selected ranking criterion
- Innovation-only gating (exp 3-2) improves slightly over ungated magnetometer correction, but does not match the best fixed norm-gated configuration
- Adding innovation gating on top of norm-based gating (exp 3-4) remains competitive, but does not surpass exp 3-3 on this sequence
- Time-varying sigma variants again perform worse than the best fixed-gating configurations

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-3-res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data04_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data04_exp3_02.png" width="952" height="631">

#### [Chosen parameters]

- quasi-static: (252162, 310194, 58032)
- suggested σ_gyro: 0.2752885
- suggested σ_acc : 0.5070689
- suggested σ_mag : 190.9833338

<br>

| exp |  tau   |      K      | mag_gain |     σ_acc    |    σ_gyro    |     σ_mag    | σ_mag_err |
|:---:|-------:|------------:|---------:|-------------:|-------------:|-------------:|----------:|
| 3-1 |  2.28  | 0.004408465 | 0.075637 |          inf |          inf |          inf |       inf |
| 3-2 |  2.44  | 0.004115562 | 0.081290 |          inf |          inf |          inf | 0.1141336 |
| 3-3 |  2.61  | 0.003856468 | 0.090535 |    0.9858854 |    0.5807959 |   54.0217667 |       inf |
| 3-4 |  2.46  | 0.004088848 | 0.103103 |    1.0180438 |    0.6821457 |   64.1990739 | 0.1663950 |
| 3-5 |  2.23  | 0.004508710 | 0.128425 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  2.13  | 0.004730251 | 0.145923 | time-varying | time-varying | time-varying | 1.7665532 |

<br>

** `σ = inf` means gating not applied<br>

<br>

| exp |    p    |    win_s     | update_ratio |  ema_alpha   |
|:---:|--------:|-------------:|-------------:|-------------:|
| 3-5 |   58    | 8.312611     | 0.224684     | 0.113612     |
| 3-6 |   54    | 9.118412     | 0.251041     | 0.169355     |

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 00:08:46.982

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965
Suggested mag_sigma:  190.98333381383583

[END] 2026-03-26 00:08:48.118
```

</details>

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| b2  | <ul><li>0.86428 rad</li><li>49.51985 deg</li></ul> | <ul><li>2.01278 rad</li><li>115.32398 deg</li></ul> |
| 3-1 | <ul><li>0.77836 rad</li><li>44.59698 deg</li></ul> | <ul><li>1.99740 rad</li><li>114.44248 deg</li></ul> |
| 3-2 | <ul><li>0.63786 rad</li><li>36.54686 deg</li></ul> | <ul><li>1.75871 rad</li><li>100.76688 deg</li></ul> |
| 3-3 | <ul><li>0.62563 rad</li><li>35.84594 deg</li></ul> | <ul><li>1.51352 rad</li><li>86.71842 deg</li></ul>  |
| 3-4 | <ul><li>0.66606 rad</li><li>38.16220 deg</li></ul> | <ul><li>1.70196 rad</li><li>97.51531 deg</li></ul>  |
| 3-5 | <ul><li>0.54571 rad</li><li>31.26704 deg</li></ul> | <ul><li>1.57967 rad</li><li>90.50838 deg</li></ul>  |
| 3-6 | <ul><li>0.54640 rad</li><li>31.30636 deg</li></ul> | <ul><li>1.60174 rad</li><li>91.77311 deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 00:08:48.169
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0010410408266554162 3.141540450466659 0.778364216206735 1.9973979935476887

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.059647245668165684 179.99700898136703 44.59698451265423 114.4424750381814

[END] 2026-03-26 01:10:02.735




[START] 2026-03-26 01:10:02.746
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0006932543037092364 3.1395432880363785 0.6378630075132432 1.7587138609525377

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.039720545731819816 179.88258000310984 36.54685823803035 100.76688160373833

[END] 2026-03-26 02:02:03.748




[START] 2026-03-26 02:02:03.760
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0034484376418402497 3.1414837495374583 0.6256297470135747 1.5135219592318112

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.1975809227914925 179.99376025742936 35.84594404171525 86.71842046435422

[END] 2026-03-26 03:33:28.681




[START] 2026-03-26 03:33:28.693
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0033279856458617494 3.139968905643992 0.6660560448646144 1.7019631438984009

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.19067953178799765 179.90696609571256 38.162200289918616 97.51530503219517

[END] 2026-03-26 04:35:02.755




[START] 2026-03-26 04:35:02.770
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002293188814071561 3.1395883511652887 0.5457128822460509 1.579669146004168

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.13139004067291088 179.88516193020806 31.267044978618387 90.50837509307387

[END] 2026-03-26 06:35:06.344




[START] 2026-03-26 06:35:06.360
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.002838688037336622 3.1399855733387 0.5463990043190333 1.6017429500652784

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.1626448438936635 179.90792108427354 31.30635687763105 91.77311090357422

[END] 2026-03-26 08:42:48.769




[START] 2026-03-26 08:43:33.245

best: exp3-6

[END] 2026-03-26 08:43:33.399
```

<br>

** `best` refers to the best experiment 2 result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift)<br>

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 2</th>
      <th colspan="4">best exp 3</th>
    </tr>
    <tr>
      <th>grav mean</th>
      <th>acc mean</th>
      <th>grav mean</th>
      <th>grav p90</th>
      <th>acc mean</th>
      <th>acc p90</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.30</td>
      <td>17.86</td>
      <td>3.06</td>
      <td>6.97</td>
      <td>36.69</td>
      <td>94.64</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 08:45:37.579

[Gravity]
RMSE norm: 0.6622473674007994

Gravity est/ref angle error in rad — min/max/mean/p90
8.903737594981226e-05 0.22727808470864533 0.05343581756267686 0.12158854004598849

Gravity est/ref angle error in deg — min/max/mean/p90
0.005101465860843863 13.02207502962219 3.0616468211724253 6.966510181792537


[Linear accel]
RMSE norm: 0.865159181947471

Linear accel est/ref angle error in rad — min/max/mean/p90
0.0001587564854961757 3.13076227607994 0.6403025597915342 1.6518147480497065

Linear accel est/ref angle error in deg — min/max/mean/p90
0.009096076589260736 179.379465078152 36.68663428747795 94.64201360071361
. . .
[END] 2026-03-26 08:45:46.518
```

</details>

<br>

#### [Observation]

- Unlike the first three datasets, the long and highly variable sequence favors a time-varying gating configuration
- The best result is exp 3-6, with exp 3-5 remaining very close, suggesting that adaptive norm-based gating is the main useful component on this dataset
- Fixed-gating variants improve over the experiment 2 baseline, but they do not match the best time-varying configurations
- This indicates that dataset 04 is more non-stationary than the other datasets, and that a fixed sigma is less suitable under this condition
- Large error plateaus still remain, suggesting that gating and correction alone may not be sufficient once the integrated state has already drifted substantially
- Although magnetometer correction reduces orientation angle error on this dataset, the secondary validation becomes worse than in experiment 2
- In particular, gravity and linear-acceleration estimates degrade substantially, indicating that improved orientation agreement does not necessarily imply better downstream gravity / linear-accel decomposition on this sequence
- This suggests that, for long uncontrolled motion, robustness may need to be improved not only at the correction stage, but also at the gyro integration stage itself

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-3-data-sum"></a>

| Dataset | best | best 2 Mean   | best 3 Mean   | best 2 p90    | best 3 p90    | best grav Mean | best acc Mean  |
|:--------|-----:|--------------:|--------------:|--------------:|--------------:|---------------:|---------------:|
| data 01 | 3-4  | 19.35529 deg  |  2.17366 deg  |  29.27439 deg |  4.35842 deg  | 1.02404 deg    |  8.15466 deg   |
| data 02 | 3-4  |  9.11787 deg  |  1.73052 deg  |  18.81032 deg |  4.12132 deg  | 0.57466 deg    |  6.28374 deg   |
| data 03 | 3-3  | 13.91633 deg  |  1.95857 deg  |  22.32396 deg |  3.56029 deg  | 0.86865 deg    |  6.75053 deg   |
| data 04 | 3-6  | 49.51985 deg  | 31.30636 deg  | 115.32398 deg | 91.77311 deg  | 3.06165 deg    | 36.68663 deg   |

<br>

** `best` refers to the best experiment result which makes minimum error (calculated by 0.4 * mean error + 0.3 * p95 + 0.2 * p99 + 0.1 * drift) per dataset<br>

<br>

Across all datasets:<br>

- On all four evaluated datasets, the best Experiment 3 configuration improves orientation angle error relative to the best Experiment 2 result
- The improvement is especially large on data 01, data 02, and data 03, where magnetometer correction appears to provide a strong additional heading constraint
- Data 04 also improves in orientation angle error, but remains substantially more difficult than the other datasets and behaves differently in the secondary validation, possibly due to its long duration and greater environmental variation

<br>
<br>


##### [Norm gating and innovation gating]

1. Norm-based gating

- Norm-based gating is included in the best-performing Experiment 3 configuration on all four datasets
- In the evaluated settings, it appears to be the most consistently useful magnetometer-gating component

<br>

2. Innovation gating

[2-1] For data 01 and data 02:<br>

- Innovation-only gating improves slightly over ungated magnetometer correction
- When combined with norm-based gating, it produces the best overall result on these two datasets

<br>

[2-2] For data 03:<br>

- Innovation-only gating remains better than ungated correction, but fixed norm-based gating alone performs best
- Adding innovation gating on top of norm-based gating remains competitive, but does not improve further on this dataset

<br>

[2-3] For data 04:<br>

- The effect of innovation gating is not clear-cut on this dataset
- Innovation-only gating performs better than ungated correction, but when combined with fixed norm-based gating, it performs worse than fixed norm-based gating alone
- The best result is obtained with a time-varying configuration that also includes innovation gating, although exp 3-5 is very close and the additional contribution of innovation gating is therefore difficult to isolate
- This suggests that the usefulness of innovation gating is dataset-dependent

<br>
<br>

##### [Fixed sigma and time-varying sigma]

For data 01, data 02, data 03:<br>

- Fixed-gating configurations perform best
- This suggests that, for these relatively consistent datasets, a fixed gating scale is sufficient and better matched than a time-varying schedule

<br>

For data 04:<br>

- Time-varying gating performs best
- This suggests that adaptive gating may become more useful when the sequence is longer and more non-stationary

<br>
<br>

#####  [Secondary validation]

For data 01, data 02, data 03:<br>

- Secondary validation remains similar to, or better than the corresponding Experiment 2 result
- In these datasets, the reduction in orientation error is not accompanied by an obvious degradation in gravity or linear-acceleration extraction

<br>

For data 04:<br>

- Secondary validation becomes markedly worse than in Experiment 2, even though orientation angle error improves
- Gravity and linear-acceleration angle errors increase substantially
- This indicates that better orientation agreement does not necessarily translate into better gravity / linear-accel decomposition on this dataset

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-3-conclusion"></a>

Experiment 3 suggests:<br>

1. In the evaluated datasets, adding magnetometer correction improves orientation angle error relative to the best gyro+accelerometer configuration from experiment 2 
2. For the relatively consistent datasets in this experiment, fixed-gating configurations perform best among the evaluated variants, with norm-based gating present in each case
3. For the long non-stationary dataset, a time-varying gating configuration performs best under the orientation-only criterion, suggesting that adaptive gating may become more useful under changing conditions, although it does not resolve all failure modes.
4. These results suggest that the next step should focus on improving robustness during gyro integration itself, especially by detecting and suppressing abnormal integrated behavior before it propagates
5. The effect of innovation gating is dataset-dependent: it is sometimes beneficial, especially when combined with norm-based gating, but it is not uniformly dominant across all sequences
6. Secondary validation reveals an important limitation. On dataset 04, better orientation error does not necessarily lead to better gravity and linear-acceleration estimation, indicating that orientation agreement alone is not always a sufficient criterion for model selection.
7. These results suggest that future improvements should be evaluated not only by orientation error, but also by their effect on gravity / linear-accel decomposition, especially for long uncontrolled sequences

<br>

Next steps:<br>
- Experiment 4: refine the model-selection criterion and reduce tuning cost
- Experiment 5: improve gyro integration robustness on long uncontrolled sequences

<br>
<br>
<br>
<br>
