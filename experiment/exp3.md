
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

This experiment evaluates whether magnetometer-based heading correction further improves orientation accuracy beyond the best gyro+accelerometer configuration from Experiment 2.<br>

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

** `best exp 2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

Key hypothesis:<br>

Adding magnetometer correction is expected to reduce heading drift and significantly improve total orientation accuracy.<br>

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
- compute a heading (yaw) correction axis from the mismatch between measured and predicted horizontal magnetic directions
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
** `weight_mag`: Mag magnitude residual-based confidence and modulated by binary innovation rejection (approaches to 1 when `| ||m|| - m0 |` is small and the mag error axis remains within the normal range)<br>
** `e_axis_mag`: Heading correction axis from horizontal magnetic alignment<br>

<br>

##### [Implementation]

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

##### [Implementation]

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

<br>
<br>
<br>

### Magnetometer gating <a name="exp-3-method-mag-gating"></a>

#### [Norm-based gating]

Norm-based magnetometer gating reduces trust when the magnetometer magnitude deviates from its expected nominal value.<br>

<br>

A Gaussian-like confidence weight is used.<br>

<br>

```py
# pipeline.py

def calc_mag_gating(m0: float, mag_sigma: float, m_meas: Vec3) -> float:
        . . .
        dev: float = abs(float(np.linalg.norm(m_meas)) - m0)
        weight_mag : float = np.exp(-0.5 * (dev / mag_sigma) ** 2)
        return weight_mag
```

<br>
<br>

#### [Innovation-based gating]

Innovation gating checks whether the magnetometer correction axis itself is abnormally large.<br>
If the innovation exceeds a threshold, the update is rejected.<br>

<br>

This gating is intended to block clearly implausible heading updates when the measured magnetic direction is inconsistent with the predicted one.<br>

<br>

```py
# pipeline.py

def calc_mag_innovation_gating(e_axis_mag: Vec3, mag_err_sigma: float) -> float:
        . . .
        e_axis_norm: float = float(np.linalg.norm(e_axis_mag))
        if e_axis_norm > 5 * mag_err_sigma:
                return 0
        return 1
```

<br>
<br>
<br>

### Suggest Gate Sigma <a name="exp-3-method-sigma"></a>

As in Experiment 2, fixed sigma values are first suggested from robust data statistics,<br>
and time-varying sigma schedules are optionally generated using percentile estimates over a sliding window with EMA smoothing.<br>

<br>

For the magnetometer, the residual is computed as the deviation from deviation around the global magnetometer norm baseline.<br>

<br>

#### [Fixed sigma]

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

<br>
<br>

#### [Time-varying sigma]

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

<br>
<br>
<br>

### Hyperparameter Search with Optuna <a name="exp-3-method-optuna"></a>

#### [Optimization target]

| exp | trial |             Target             |
|:---:|------:|:-------------------------------|
| 3-1 |    20 | <ul><li>tau</li><li>mag_gain</li></ul> |
| 3-2 |    20 | <ul><li>tau</li><li>mag_gain</li><li>mag_err_sigma</li></ul> |
| 3-3 |    30 | <ul><li>tau</li><li>mag_gain</li><li>acc_gate_sigma</li><li>gyro_gate_sigma</li><li>mag_gate_sigma</li></ul> |
| 3-4 |    30 | <ul><li>tau</li><li>mag_gain</li><li>acc_gate_sigma</li><li>gyro_gate_sigma</li><li>mag_gate_sigma</li><li>mag_err_sigma</li></ul> |
| 3-5 |    40 | <ul><li>tau</li><li>mag_gain</li><li>percentile (`p`)</li><li>sliding window size (`win_s`)</li><li>update ratio (`update_ratio`)</li><li>EMA factor (`ema_alpha`)</li></ul> |
| 3-6 |    40 | <ul><li>tau</li><li>mag_gain</li><li>percentile (`p`)</li><li>sliding window size (`win_s`)</li><li>update ratio (`update_ratio`)</li><li>EMA factor (`ema_alpha`)</li><li>mag_err_sigma</li></ul> |

** For exp 3-5 and exp 3-6, Optuna optimizes the parameters of the function `suggest_timevarying_gate_sigma(...)`, rather than the sigma values directly.<br>

#### [Implementation]

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
| 3-2 |  4.00  | 0.002502711 | 4.125977 |          inf |          inf |          inf | 0.3837193 |
| 3-3 |  3.25  | 0.003073947 | 3.053092 |    2.8671055 |    3.0789585 |   30.1908042 |       inf |
| 3-4 |  2.44  | 0.004096940 | 3.394846 |   23.1545801 |    0.1405356 |   23.1260387 | 0.2318842 |
| 3-5 |  2.27  | 0.004400636 | 2.831962 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  2.37  | 0.004217863 | 3.653864 | time-varying | time-varying | time-varying | 8.0660255 |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-15 12:50:43.650

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301
Suggested mag_sigma:  6.850455808768257

[END] 2026-03-15 12:50:44.616
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.39107 rad</li><li>22.40684 deg</li></ul> | <ul><li>0.56630 rad</li><li>32.44667 deg</li></ul> |
| b2  | <ul><li>0.33841 rad</li><li>19.38932 deg</li></ul> | <ul><li>0.51083 rad</li><li>29.26866 deg</li></ul> |
| 3-1 | <ul><li>0.04217 rad</li><li>2.41633 deg</li></ul>  | <ul><li>0.08175 rad</li><li>4.68411 deg</li></ul>  |
| 3-2 | <ul><li>0.04318 rad</li><li>2.47422 deg</li></ul>  | <ul><li>0.08290 rad</li><li>4.75006 deg</li></ul>  |
| 3-3 | <ul><li>0.03908 rad</li><li>2.23920 deg</li></ul>  | <ul><li>0.07565 rad</li><li>4.33467 deg</li></ul>  |
| 3-4 | <ul><li>0.04825 rad</li><li>2.76465 deg</li></ul>  | <ul><li>0.08024 rad</li><li>4.59742 deg</li></ul>  |
| 3-5 | <ul><li>0.06104 rad</li><li>3.49749 deg</li></ul>  | <ul><li>0.10907 rad</li><li>6.24952 deg</li></ul>  |
| 3-6 | <ul><li>0.05925 rad</li><li>3.39504 deg</li></ul>  | <ul><li>0.10355 rad</li><li>5.93276 deg</li></ul>  |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-15 12:50:44.766
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.002818835027033219 0.7172417482362367 0.042172969711376365 0.08175313998327868

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.16150735019264875 41.09492506452112 2.416333173994919 4.68410988298409

[END] 2026-03-15 12:53:36.655




[START] 2026-03-15 12:53:36.670
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.002820728064878768 0.7945026278374672 0.043183333586495294 0.08290412206364806

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.16161581327165725 45.52164738714002 2.474222759811717 4.750056298484442

[END] 2026-03-15 12:57:10.235




[START] 2026-03-15 12:57:10.258
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0027866181005946527 0.5847782863177167 0.03908133886716545 0.07565423443524776

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.15966145627883546 33.50532775689802 2.239195774809166 4.334668335432996

[END] 2026-03-15 13:03:19.026




[START] 2026-03-15 13:03:19.049
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0012669969968351258 0.7418838811030043 0.04825221769122014 0.080240176875303

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.07259358057440282 42.50681527598751 2.764648425853399 4.597423482338087

[END] 2026-03-15 13:09:07.393




[START] 2026-03-15 13:09:07.407
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0014789461159227593 0.7768795498036078 0.06104275685644935 0.10907470949304478

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.08473737056963991 44.51191939377017 3.497492337717816 6.249520505567001

[END] 2026-03-15 13:15:47.904




[START] 2026-03-15 13:15:47.925
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.001598575554759997 0.7530869629101015 0.05925455189937533 0.10354613980308267

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.09159163252053204 43.14870458107398 3.395035740773102 5.932756795588221

[END] 2026-03-15 13:23:52.416




[START] 2026-03-15 13:23:56.260

best: exp3-3

[END] 2026-03-15 13:23:56.267
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

|   best exp 2    |            best exp 3             |
|:---------------:|:---------------------------------:|
|  grav  |  acc   |      grav       |       acc       |
|:------:|:------:|:---------------:|:---------------:|
|  mean  |  mean  |  mean  |   p90  |  mean  |   p90  |
|-------:|-------:|-------:|-------:|-------:|-------:|
|   1.34 |  10.23 |   1.06 |   2.05 |   8.15 |  18.59 |

<br>

```
[START] 2026-03-15 13:24:05.027

[Gravity]
RMSE norm: 0.21606680649250662

Gravity est/ref angle error in rad — min/max/mean/p90
0.0006645244271012906 0.09464804292556657 0.018446895807495575 0.035861888397220265

Gravity est/ref angle error in deg — min/max/mean/p90
0.0380744450562529 5.422933398808014 1.0569292748870691 2.0547348505298975


[Linear accel]
RMSE norm: 0.7407984636891747

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00018870243403817054 2.8168264749191327 0.14225758002977087 0.32451767215278243

Linear accel est/ref angle error in deg — min/max/mean/p90
0.01081185305423298 161.39226863357953 8.150758939450416 18.593492991764556
. . .
[END] 2026-03-15 13:24:05.505
```

#### [Observation]

- Adding magnetometer correction produces a dramatic improvement over the best Exp 2 result
- The best result is exp 3-3, meaning fixed norm-based gyro/acc/mag gating performs best on this dataset
- Innovation gating alone (exp 3-2) is slightly worse than no magnetometer gating (exp 3-1)
- Combining innovation gating with norm-based gating (exp 3-4) improves over exp 3-2, suggesting that most of the gain comes from norm-based gating rather than the innovation gate itself
- Time-varying sigma variants (exp 3-5, exp 3-6) perform worst among Exp 3 runs, indicating that a fixed gating structure is more suitable for this relatively consistent sequence

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
| 3-2 |  3.94  | 0.002537197 | 5.782205 |          inf |          inf |          inf | 0.3414472 |
| 3-3 |  2.47  | 0.004049031 | 1.713536 |    1.6613410 |    5.9852156 |   48.7764070 |       inf |
| 3-4 |  3.80  | 0.002628174 | 4.638281 |    1.6655318 |    5.2753572 |   35.1315509 | 0.7029849 |
| 3-5 |  2.67  | 0.003740069 | 8.398177 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  2.12  | 0.004710965 | 6.516492 | time-varying | time-varying | time-varying | 3.5544619 |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-14 12:17:49.094

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874
Suggested mag_sigma:  5.051061836028509

[END] 2026-03-14 12:17:49.968
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.38383 rad</li><li>21.99183 deg</li></ul> | <ul><li>0.54410 rad</li><li>31.17450 deg</li></ul> |
| b2  | <ul><li>0.16965 rad</li><li>9.72036 deg</li></ul>  | <ul><li>0.32722 rad</li><li>18.74848 deg</li></ul> |
| 3-1 | <ul><li>0.04564 rad</li><li>2.61517 deg</li></ul>  | <ul><li>0.08512 rad</li><li>4.87730 deg</li></ul>  |
| 3-2 | <ul><li>0.04721 rad</li><li>2.70495 deg</li></ul>  | <ul><li>0.08457 rad</li><li>4.84575 deg</li></ul>  |
| 3-3 | <ul><li>0.03317 rad</li><li>1.90078 deg</li></ul>  | <ul><li>0.06883 rad</li><li>3.94388 deg</li></ul>  |
| 3-4 | <ul><li>0.03335 rad</li><li>1.91095 deg</li></ul>  | <ul><li>0.07966 rad</li><li>4.56409 deg</li></ul>  |
| 3-5 | <ul><li>0.05503 rad</li><li>3.15278 deg</li></ul>  | <ul><li>0.11605 rad</li><li>6.64928 deg</li></ul>  |
| 3-6 | <ul><li>0.04832 rad</li><li>2.76856 deg</li></ul>  | <ul><li>0.09452 rad</li><li>5.41566 deg</li></ul>  |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-14 12:17:50.098
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0014937441061020549 0.3703654566537466 0.04564329581096676 0.08512497175927315

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.08558523295218959 21.220377543695115 2.6151682130355454 4.877301612976674

[END] 2026-03-14 12:24:06.795




[START] 2026-03-14 12:24:06.818
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.001461101659713782 0.413797346733879 0.047210213064651306 0.08457428627168499

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.0837149585411595 23.70884154156281 2.7049459585178997 4.845749658698769

[END] 2026-03-14 12:30:28.544




[START] 2026-03-14 12:30:28.566
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0009270817827165907 0.4391216399910695 0.033174801085153104 0.06883370828676039

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.05311787341312508 25.159816664351432 1.9007760883652964 3.943880973066051

[END] 2026-03-14 12:41:53.269




[START] 2026-03-14 12:41:53.292
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.00037329840420935375 0.5564817624602978 0.033352379462621116 0.07965846526186691

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.021388423060164613 31.884056364976672 1.9109505799269946 4.564093861994454

[END] 2026-03-14 12:53:30.874




[START] 2026-03-14 12:53:30.898
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.001309377045516123 0.40911179557283 0.05502637294115517 0.11605184795478786

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.07502177849938294 23.44037923534208 3.152778931441066 6.649281092503279

[END] 2026-03-14 13:08:37.680




[START] 2026-03-14 13:08:37.702
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0011181581820877934 0.349865289111564 0.04832056125403205 0.09452101267596816

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.06406574466165116 20.04580446421697 2.7685642235594092 5.415655101635531

[END] 2026-03-14 13:23:55.792




[START] 2026-03-14 13:24:02.064

best: exp3-3

[END] 2026-03-14 13:24:02.077
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

|   best exp 2    |            best exp 3             |
|:---------------:|:---------------------------------:|
|  grav  |  acc   |      grav       |       acc       |
|:------:|:------:|:---------------:|:---------------:|
|  mean  |  mean  |  mean  |   p90  |  mean  |   p90  |
|-------:|-------:|-------:|-------:|-------:|-------:|
|   3.07 |  14.61 |   0.60 |   1.07 |   5.90 |  14.36 |

<br>

```
[START] 2026-03-14 13:24:16.509

[Gravity]
RMSE norm: 0.11506990760845125

Gravity est/ref angle error in rad — min/max/mean/p90
6.137771836312499e-05 0.03176373803415814 0.010540202265754386 0.018588600653700025

Gravity est/ref angle error in deg — min/max/mean/p90
0.0035166842183496734 1.819928130916432 0.603909105041954 1.0650483645111346


[Linear accel]
RMSE norm: 0.4806977657990752

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00021693350637203048 2.9726772490497213 0.10294921852894368 0.25057983760177666

Linear accel est/ref angle error in deg — min/max/mean/p90
0.012429374350091697 170.32186022510894 5.898555725878486 14.357167125655371
. . .
[END] 2026-03-14 13:24:17.330
```

#### [Observation]

- As in Dataset 01, magnetometer correction produces a very large improvement over the best Exp 2 result
- The best result is again exp 3-3
- Innovation gating alone (exp 3-2) is worse than ungated magnetometer correction (exp 3-1)
- The combined fixed-gating configuration (exp 3-4) improves over innovation-only gating, but still does not beat exp 3-3
- Time-varying sigma variants are again the weakest among Exp 3 runs, suggesting that this dataset also benefits most from a fixed gating structure under relatively stable conditions

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-3-res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data03_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data03_exp3_02.png" width="952" height="631">

#### [Chosen parameters]

- quasi-static: 
- suggested σ_gyro: 0.0000000
- suggested σ_acc : 0.0000000
- suggested σ_mag : 0.0000000

<br>

| exp |  tau   |      K      | mag_gain |     σ_acc    |    σ_gyro    |     σ_mag    | σ_mag_err |
|:---:|-------:|------------:|---------:|-------------:|-------------:|-------------:|----------:|
| 3-1 |    |  |  |          inf |          inf |          inf |       inf |
| 3-2 |    |  |  |          inf |          inf |          inf |  |
| 3-3 |    |  |  |  |  |  |       inf |
| 3-4 |    |  |  |  |  |  |  |
| 3-5 |    |  |  | time-varying | time-varying | time-varying |       inf |
| 3-6 |    |  |  | time-varying | time-varying | time-varying |  |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| b2  | <ul><li>0.24070 rad</li><li>13.79096 deg</li></ul> | <ul><li>0.39439 rad</li><li>22.59707 deg</li></ul> |
| 3-1 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-2 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-3 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-4 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-5 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-6 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

```
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

|   best exp 2    |            best exp 3             |
|:---------------:|:---------------------------------:|
|  grav  |  acc   |      grav       |       acc       |
|:------:|:------:|:---------------:|:---------------:|
|  mean  |  mean  |  mean  |   p90  |  mean  |   p90  |
|-------:|-------:|-------:|-------:|-------:|-------:|
|   2.75 |  11.88 |        |        |        |        |

<br>

```
```

#### [Observation]

- 

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-3-res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data04_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data04_exp3_02.png" width="952" height="631">

#### [Chosen parameters]

- quasi-static: 
- suggested σ_gyro: 0.0000000
- suggested σ_acc : 0.0000000
- suggested σ_mag : 0.0000000

<br>

| exp |  tau   |      K      | mag_gain |     σ_acc    |    σ_gyro    |     σ_mag    | σ_mag_err |
|:---:|-------:|------------:|---------:|-------------:|-------------:|-------------:|----------:|
| 3-1 |    |  |  |          inf |          inf |          inf |       inf |
| 3-2 |    |  |  |          inf |          inf |          inf |  |
| 3-3 |    |  |  |  |  |  |       inf |
| 3-4 |    |  |  |  |  |  |  |
| 3-5 |    |  |  | time-varying | time-varying | time-varying |       inf |
| 3-6 |    |  |  | time-varying | time-varying | time-varying |  |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| b2  | <ul><li>0.75918 rad</li><li>43.49773 deg</li></ul> | <ul><li>1.77312 rad</li><li>101.59235 deg</li></ul> |
| 3-1 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-2 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-3 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-4 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-5 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
| 3-6 | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

```
```

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp3]

|   best exp 2    |            best exp 3             |
|:---------------:|:---------------------------------:|
|  grav  |  acc   |      grav       |       acc       |
|:------:|:------:|:---------------:|:---------------:|
|  mean  |  mean  |  mean  |   p90  |  mean  |   p90  |
|-------:|-------:|-------:|-------:|-------:|-------:|
|   2.17 |  28.50 |        |        |        |        |

<br>

```
```

#### [Observation]

- 

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-3-data-sum"></a>

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-3-conclusion"></a>

<br>
<br>
<br>
<br>
