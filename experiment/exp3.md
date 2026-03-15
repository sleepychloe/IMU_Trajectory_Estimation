
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
