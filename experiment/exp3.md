
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

Adding magnetometer correction is expected to reduce heading drift and potentially improve total orientation accuracy.<br>

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
** `weight_mag`: Mag magnitude residual-based confidence and modulated by binary innovation rejection (approaches to 1 when `| ||m|| - m0 |` is small and the mag error axis remains within the accepted range)<br>
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

For the magnetometer, the residual is computed as the deviation from deviation from the global magnetometer norm baseline.<br>

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
      <td>1.34</td>
      <td>10.23</td>
      <td>1.06</td>
      <td>2.05</td>
      <td>8.15</td>
      <td>18.59</td>
    </tr>
  </tbody>
</table>

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

- Adding magnetometer correction produces a dramatic improvement over the best exp 2 result
- The best result is exp 3-3, meaning fixed norm-based gyro/acc/mag gating performs best on this dataset
- Innovation gating alone (exp 3-2) is slightly worse than no magnetometer gating (exp 3-1)
- Combining innovation gating with norm-based gating (exp 3-4) improves over exp 3-2, suggesting that most of the gain comes from norm-based gating rather than the innovation gate itself
- Time-varying sigma variants (exp 3-5, exp 3-6) perform worst among exp 3 runs, indicating that a fixed gating structure is more suitable for this relatively consistent sequence

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
      <td>3.07</td>
      <td>14.61</td>
      <td>0.60</td>
      <td>1.07</td>
      <td>5.90</td>
      <td>14.36</td>
    </tr>
  </tbody>
</table>

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

- Adding magnetometer correction produces a dramatic improvement over the best exp 2 result
- The best result is exp 3-3, reinforcing the same pattern observed in Dataset 01
- Combining innovation gating with norm-based gating (exp 3-4) improves over exp 3-2, suggesting that most of the gain likely comes from norm-based gating rather than from the innovation gate alone
- Time-varying sigma variants (exp 3-5, exp 3-6) perform worst among the exp 3 runs on this dataset, indicating that a fixed gating structure is more suitable for this relatively consistent sequence

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
| 3-2 |  3.35  | 0.002987808 | 2.131268 |          inf |          inf |          inf | 0.1900067 |
| 3-3 |  2.49  | 0.004021669 | 1.403544 |    1.9785102 |    1.7974045 |   23.2823962 |       inf |
| 3-4 |  3.98  | 0.002512575 | 2.670871 |    6.2055300 |    4.4092371 |   44.7035222 | 0.8666832 |
| 3-5 |  3.47  | 0.002880201 | 5.377969 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  3.44  | 0.002903001 | 7.139949 | time-varying | time-varying | time-varying | 6.1261591 |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-14 13:25:16.592

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786
Suggested mag_sigma:  5.098923949491506

[END] 2026-03-14 13:25:17.550
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul> | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul> |
| b2  | <ul><li>0.24070 rad</li><li>13.79096 deg</li></ul> | <ul><li>0.39439 rad</li><li>22.59707 deg</li></ul> |
| 3-1 | <ul><li>0.04335 rad</li><li>2.48355 deg</li></ul>  | <ul><li>0.07790 rad</li><li>4.46346 deg</li></ul>  |
| 3-2 | <ul><li>0.04584 rad</li><li>2.62661 deg</li></ul>  | <ul><li>0.08220 rad</li><li>4.70992 deg</li></ul>  |
| 3-3 | <ul><li>0.04215 rad</li><li>2.41480 deg</li></ul>  | <ul><li>0.07740 rad</li><li>4.43479 deg</li></ul>  |
| 3-4 | <ul><li>0.04328 rad</li><li>2.47981 deg</li></ul>  | <ul><li>0.07758 rad</li><li>4.44493 deg</li></ul>  |
| 3-5 | <ul><li>0.06176 rad</li><li>3.53880 deg</li></ul>  | <ul><li>0.11218 rad</li><li>6.42756 deg</li></ul>  |
| 3-6 | <ul><li>0.06180 rad</li><li>3.54069 deg</li></ul>  | <ul><li>0.11427 rad</li><li>6.54703 deg</li></ul>  |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-14 13:25:17.604
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0009205019993881509 0.4253150125622266 0.04334613532077613 0.0779020739570321

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.05274087959829493 24.368755183369174 2.4835506120834188 4.463460053053944

[END] 2026-03-14 13:34:06.384




[START] 2026-03-14 13:34:06.403
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.001351753088159024 0.4234746843207192 0.04584291670204892 0.08220358954980264

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.07744974689528757 24.26331214221207 2.626605647597194 4.70991874202941

[END] 2026-03-14 13:43:12.929




[START] 2026-03-14 13:43:12.952
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.002436525772065005 0.3035286839807431 0.04214626617664195 0.07740170674121596

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.1396026434141792 17.3909125532567 2.4148031741565563 4.434791123380967

[END] 2026-03-14 13:59:03.475




[START] 2026-03-14 13:59:03.500
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0014149341229442365 0.3901367537430255 0.04328076320574701 0.07757865952716583

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.08106975353374948 22.353189422410082 2.4798050657944066 4.444929771188977

[END] 2026-03-14 14:15:05.149




[START] 2026-03-14 14:15:05.172
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002847387797902579 0.7120212228973223 0.06176375731503987 0.1121820572653222

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.16314330345686717 40.79581099576022 3.53880262102205 6.427558418397876

[END] 2026-03-14 14:36:12.787




[START] 2026-03-14 14:36:12.814
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0010956640352578484 0.6719577491000192 0.06179676164356974 0.11426716713423904

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.06277692498454773 38.50034303454179 3.5406936297524743 6.547026413707887

[END] 2026-03-14 14:57:38.602




[START] 2026-03-14 14:57:46.465

best: exp3-3

[END] 2026-03-14 14:57:46.481
```

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
      <td>2.75</td>
      <td>11.88</td>
      <td>1.01</td>
      <td>1.38</td>
      <td>6.78</td>
      <td>15.71</td>
    </tr>
  </tbody>
</table>

<br>

```
[START] 2026-03-14 14:58:05.863

[Gravity]
RMSE norm: 0.1800490753281907

Gravity est/ref angle error in rad — min/max/mean/p90
6.344569720577262e-05 0.03952086951592452 0.01761454253787616 0.024120060778194733

Gravity est/ref angle error in deg — min/max/mean/p90
0.0036351706781557313 2.2643790259497076 1.009238945473962 1.3819776841895903


[Linear accel]
RMSE norm: 0.5500461373406331

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00023568751622905288 2.728179621257335 0.11840303812910537 0.27423599127868414

Linear accel est/ref angle error in deg — min/max/mean/p90
0.013503899963845826 156.31317805164468 6.7839943663243005 15.712564890855054
. . .
[END] 2026-03-14 14:58:06.946
```

#### [Observation]

- The best result is exp 3-3, continuing the same trend as Dataset 01 and 02
- Innovation-only gating (exp 3-2) is again worse than ungated magnetometer correction
- Adding norm-based gating together with innovation gating (exp 3-4) improves over exp 3-2, but still does not surpass exp 3-3
- Time-varying sigma variants perform worst on this dataset as well
- Across the first three datasets, fixed norm-based gating appears to be the most reliable and consistently effective strategy among the evaluated Experiment 3 configurations

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
| 3-2 |  3.87  | 0.002595857 | 0.069912 |          inf |          inf |          inf | 0.9698417 |
| 3-3 |  1.89  | 0.005334749 | 0.013543 |    0.7888448 |    0.4831832 | 1642.0560361 |       inf |
| 3-4 |  1.82  | 0.005519675 | 8.147755 |    2.7964138 |    1.8038140 |   12.0085436 | 0.9774938 |
| 3-5 |  2.85  | 0.003525506 | 0.115153 | time-varying | time-varying | time-varying |       inf |
| 3-6 |  2.69  | 0.003742875 | 0.209838 | time-varying | time-varying | time-varying | 9.9520389 |

<br>

** `σ = inf` means gating not applied<br>

<br>

```
[START] 2026-03-14 14:59:06.317

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965
Suggested mag_sigma:  190.98333381383583

[END] 2026-03-14 14:59:07.447
```

<br>

#### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-2 | <ul><li>0.88808 rad</li><li>50.88316 deg</li></ul> | <ul><li>2.03692 rad</li><li>116.70718 deg</li></ul> |
| b2  | <ul><li>0.75918 rad</li><li>43.49773 deg</li></ul> | <ul><li>1.77312 rad</li><li>101.59235 deg</li></ul> |
| 3-1 | <ul><li>0.77836 rad</li><li>44.59698 deg</li></ul> | <ul><li>1.99740 rad</li><li>114.44248 deg</li></ul> |
| 3-2 | <ul><li>0.77879 rad</li><li>44.62122 deg</li></ul> | <ul><li>1.87853 rad</li><li>107.63194 deg</li></ul> |
| 3-3 | <ul><li>0.74514 rad</li><li>42.69326 deg</li></ul> | <ul><li>1.91506 rad</li><li>109.72506 deg</li></ul> |
| 3-4 | <ul><li>0.81872 rad</li><li>46.90937 deg</li></ul> | <ul><li>2.21074 rad</li><li>126.66597 deg</li></ul> |
| 3-5 | <ul><li>0.55539 rad</li><li>31.82141 deg</li></ul> | <ul><li>1.54722 rad</li><li>88.64902 deg</li></ul>  |
| 3-6 | <ul><li>0.55466 rad</li><li>31.77986 deg</li></ul> | <ul><li>1.62392 rad</li><li>93.04375 deg</li></ul>  |

<br>

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

```
[START] 2026-03-14 14:59:07.515
. . .
[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0010410408266554162 3.141540450466659 0.778364216206735 1.9973979935476887

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.059647245668165684 179.99700898136703 44.59698451265423 114.4424750381814

[END] 2026-03-14 16:04:04.061




[START] 2026-03-14 16:04:04.084
. . .
[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0003003369552661075 3.14154221278597 0.7787872335478037 1.8785317889543232

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.017208039968557364 179.99710995482567 44.62122162095831 107.631943188243

[END] 2026-03-14 17:10:33.353




[START] 2026-03-14 17:10:33.375
. . .
[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.00420210376570762 3.138358156891251 0.7451378729690504 1.9150635771071582

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.24076281085107673 179.81467699032453 42.69325527648186 109.72506046746646

[END] 2026-03-14 19:07:08.227




[START] 2026-03-14 19:07:08.251
. . .
[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0008595901018048794 3.141555252455268 0.8187229539842753 2.210738301304351

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.04925088494464036 179.99785707284272 46.90936985378248 126.66597427266025

[END] 2026-03-14 21:05:25.190




[START] 2026-03-14 21:05:25.214
. . .
[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002189388844674431 3.141076690382266 0.5553883357825457 1.5472173633231987

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.12544274051286824 179.97043748582468 31.821407631134466 88.64902490777858

[END] 2026-03-14 23:40:52.441




[START] 2026-03-14 23:40:52.463
. . .
[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0037251091962698536 3.1394613589301508 0.5546631381703352 1.6239197559160932

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.21343303517163284 179.8778858111037 31.77985686864184 93.04374828190694

[END] 2026-03-15 02:18:28.861




[START] 2026-03-15 02:19:13.993

best: exp3-5

[END] 2026-03-15 02:19:14.076
```

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
      <td>2.17</td>
      <td>28.50</td>
      <td>3.41</td>
      <td>7.61</td>
      <td>38.41</td>
      <td>100.18</td>
    </tr>
  </tbody>
</table>

<br>

```
[START] 2026-03-15 02:21:13.623

[Gravity]
RMSE norm: 0.7395152105393297

Gravity est/ref angle error in rad — min/max/mean/p90
5.9182467519623604e-05 0.19396061444713947 0.05944699847800072 0.13274668226565928

Gravity est/ref angle error in deg — min/max/mean/p90
0.00339090561004451 11.113124599585273 3.40606211751007 7.60582463818641


[Linear accel]
RMSE norm: 0.9255998973575063

Linear accel est/ref angle error in rad — min/max/mean/p90
4.4346043482774956e-05 3.137215371382285 0.6703123279056576 1.748513949129926

Linear accel est/ref angle error in deg — min/max/mean/p90
0.0025408411296666353 179.74920020377206 38.4060673445835 100.18246970489707
. . .
[END] 2026-03-15 02:21:21.627
```

#### [Observation]

- Unlike the first three datasets, fixed-gating magnetometer configurations do not provide the best result here
- The best result is exp 3-5, i.e. time-varying norm-based gating
- exp 3-6 is very close, suggesting that time-varying norm-based gating is the main useful component, while innovation gating provides little additional benefit
- This indicates that Dataset 04 is substantially more non-stationary than the other datasets, and that a fixed sigma is less suitable under this condition
- Large error plateaus still remain, suggesting that gating and correction alone may not be sufficient once the integrated state has already drifted badly
- This suggests that for long uncontrolled sequences, robustness may need to be improved not only at the correction stage, but also at the gyro integration stage itself

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-3-data-sum"></a>

| Dataset | best | best 2 Mean   | best 3 Mean   | best 2 p90    | best 3 p90    | best grav Mean | best acc Mean  |
|:--------|-----:|--------------:|--------------:|--------------:|--------------:|---------------:|---------------:|
| data 01 | 3-3  | 19.38932 deg  |  2.23920 deg  |  29.26866 deg |  4.33467 deg  | 1.05693 deg    |  8.15075 deg   |
| data 02 | 3-3  |  9.72036 deg  |  1.90078 deg  |  18.74848 deg |  3.94388 deg  | 0.60391 deg    |  5.89856 deg   |
| data 03 | 3-3  | 13.79096 deg  |  2.41480 deg  |  22.59707 deg |  4.43479 deg  | 1.00924 deg    |  6.78399 deg   |
| data 04 | 3-5  | 43.49773 deg  | 31.82141 deg  | 101.59235 deg | 88.64902 deg  | 3.40606 deg    | 38.40607 deg   |

<br>

** best = minimum error (0.8 * mean + 0.2 * p90) across exp 2 and exp 3 (per dataset)<br>

<br>

Across all datasets:<br>

- In these datasets, adding magnetometer correction consistently improves over the best gyro+acc configuration from experiment 2
- The improvement is especially large on Dataset 01, 02, and 03, where heading correction appears to be highly effective
- Dataset 04 also improves, but remains substantially harder than the others, likely due to its long duration and larger environmental variation

<br>
<br>

##### [Norm gating and innovation gating]

1. Norm-based gating

Norm-based gating is useful across all four datasets.<br>

<br>

2. Innovation gating

For data 01, data 02, data 03:<br>

- In these datasets, innovation gating alone does not improve performance, and produces worse results than ungated or norm-gated magnetometer correction
- When combined with norm gating (exp 3-4), innovation gating usually performs better than innovation gating alone, suggesting that the observed gain mainly comes from the norm-based gate

<br>

For data 04:<br>

- The only dataset where innovation-related variants are not clearly harmful is Dataset 04, which also appears to be the most non-stationary sequence in this experiment

<br>

##### [Fixed sigma and time-varying sigma]

For data 01, data 02, data 03:<br>

- Fixed sigma gating performs best
- This suggests that these datasets are sufficiently consistent that a fixed gating scale is more effective than an adaptive schedule

<br>

For data 04:<br>

- Time-varying sigma performs best on this dataset
- This indicates that adaptive gating can become useful when the environment changes significantly over time

<br>
<br>

Overall interpretation:<br>

- In this experiment, magnetometer correction provides the dominant performance gain beyond experiment 2
- Among the evaluated configurations, fixed norm-based gating (exp 3-3) is the most reliable choice on the short-to-medium sequences
- Time-varying sigma is not universally helpful, but may become useful in the long non-stationary sequence
- Long uncontrolled motion still reveals a deeper limitation — once gyro integration drifts into a bad regime, downstream correction and gating may no longer be sufficient to recover fully

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-3-conclusion"></a>

Experiment 3 suggests:<br>

1. Adding magnetometer correction yields a major improvement over gyro+accelerometer fusion alone on all evaluated datasets
2. For the relatively consistent datasets in this experiment, fixed norm-based gating is the most effective choice among the evaluated configurations
3. For the long non-stationary dataset, time-varying sigma can help, but it does not solve all failure modes
4. These results suggest that the next step should focus on improving robustness during gyro integration itself, especially by detecting and suppressing abnormal integrated behavior before it propagates

<br>

Next steps:<br>
- Experiment 4: improve gyro integration robustness on long uncontrolled sequences

<br>
<br>
<br>
<br>
