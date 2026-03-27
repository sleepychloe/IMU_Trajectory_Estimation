
 * [Experiment 4](#exp-4) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-4-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why Segment-based Tuning](#exp-4-why-seg) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Segment-based Proxy Objective](#exp-4-why-seg-proxy) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Segment Policies](#exp-4-why-seg-policy) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Method](#exp-4-method) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Redefine the Best-Model Criterion](#exp-4-method-best) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Hyperparameter Search with Optuna](#exp-4-method-optuna) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Experiment Structure](#exp-4-method-structure) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Evaluation and Measurements](#exp-4-method-eval) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Why Segment-based Tuning Sometimes Works Better](#exp-4-method-why-seg-better)<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Results](#exp-4-res) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 01 — 5 min](#exp-4-res-data-01) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 02 — 9 min](#exp-4-res-data-02) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 03 — 13 min](#exp-4-res-data-03) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Dataset 04 — 96 min](#exp-4-res-data-04) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Cross-dataset Summary](#exp-4-data-sum) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Conclusion](#exp-4-conclusion) <br>

<br>
<br>

## Experiment 4 — Segment-based Proxy Objective for Tuning Cost Reduction <a name="exp-4"></a>

### Goal <a name="exp-4-goal"></a>

This experiment evaluates whether a segment-based proxy objective can replace full-data tuning without causing a large loss in final full-sequence performance.<br>

<br>

In Experiment 3, each configuration was tuned on the full dataset and then evaluated on the full dataset.<br>
While this is conceptually the most direct optimization target, the tuning cost becomes increasingly impractical for long recordings.<br>

<br>

Therefore, Experiment 4 introduces segment-based tuning:<br>

- tuning is performed on a subset of representative segments
- final evaluation is still performed on the full dataset
- runtime reduction and final performance preservation are both examined

<br>

Key hypothesis:<br>

A segment-based proxy objective is expected to preserve comparable full-sequence performance, or degrade it only marginally, while substantially reducing the tuning cost.<br>

<br>
<br>
<br>
<br>

### Why Segment-based Tuning? <a name="exp-4-why-seg"></a>

In long datasets, the total tuning cost of Experiment 3 becomes very large.<br>

<br>

Using the full sequence for every Optuna trial is computationally expensive because:<br>

- each trial requires running the full recursive estimation pipeline
- the cost grows directly with sequence length
- the same expensive full-sequence evaluation is repeated many times across all experiment variants

<br>

As a result, full-data tuning becomes increasingly impractical for long recordings.<br>

<br>

Experiment 4 asks two practical questions:<br>

1. How much runtime can be reduced by replacing the full-data tuning objective with a segment-based proxy objective?<br>
2. How much of the final full-data performance can still be preserved after doing so?<br>

<br>
<br>
<br>

### Segment-based Proxy Objective <a name="exp-4-why-seg-proxy"></a>

Instead of tuning on the full sequence, Experiment 4 uses a segment-based proxy objective.<br>

<br>

- choose multiple representative sub-segments from the sequence
- evaluate candidate parameters only on those segments during tuning
- after tuning, run the selected parameters once on the full sequence for the final evaluation

<br>

This reduces the total optimization cost because each Optuna trial runs on a much smaller amount of data.<br>

<br>

For each selected segment, the local score is evaluated after resetting the initial orientation to the reference orientation at the segment start:<br>

- each segment begins from `q0 = q_ref[seg_start]`
- the local fusion behavior inside the segment is evaluated
- long-horizon accumulated drift from earlier parts of the sequence is not carried into the segment score

<br>

The segment-based scorer behaves more like a local-quality proxy than a full-trajectory optimization target.<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# autotune.py

def calc_score_quasi_ori_from_seg(seg: list[tuple[int,int]], . . .) -> float:
        . . .
        for seg_s, seg_e in seg:
                seg_len: int = seg_e - seg_s
                . . .
                seg_kwargs["q0"] = q_ref[seg_s]
                seg_kwargs["dt"] = dt[seg_s:seg_e]
                seg_kwargs["w"] = w[seg_s:seg_e]
                seg_kwargs["a"] = a[seg_s:seg_e]
                seg_kwargs["m"] = m[seg_s:seg_e]

                q_est, extra = runner_func(K=K, **seg_kwargs)
                . . .
                if best_quasi_static is not None:
                        qs, qe, _ = best_quasi_static
                        inter_s: int = max(seg_s, qs)
                        inter_e: int = min(seg_e, qe)
                        if inter_e - inter_s >= 2:
                                local_s = inter_s - seg_s
                                local_e = inter_e - seg_s

                                gb = g_body_est[local_s:local_e]
                                . . .
                                mean_dir: Vec3 = as_vec3(np.mean(gb_unit, axis=0))
                                . . .
                                quasi_score += score_angle_err(ang)
                ori_score += score_angle_err(calc_angle_err(q_est, q_ref[seg_s:seg_e]))
                . . .
        . . .
        return (quasi_score + ori_score) / cnt


# optuna_exp_4.py

def exp_4_1(seg: list[tuple[int, int]], . . .) -> tuple[float, float, float]:
        def objective(trial):
                . . .
                score: float = calc_score_quasi_ori_from_seg(seg, . . .)
                return score
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
	. . .
        return best_tau, best_K, best_mag_gain
```

</details>

<br>
<br>
<br>

### Segment Policies <a name="exp-4-why-seg-policy"></a>

Three segment policies are prepared.<br>

<br>

The purpose of these designs is to preserve representative temporal coverage while reducing the amount of data used during tuning.<br>

<br>

For all datasets:<br>

- `seg_1`: head 5 s + 3 s window every 15 s + tail 5 s
- `seg_2`: head 10 s + 5 s window every 15 s + tail 10 s

<br>

For datasets longer than 30 minutes:<br>

- `seg_3`: head 30 s + 10 s window every 20 s + tail 30 s

<br>

- the head captures early transient behavior
- the middle windows sample recurring motion across time
- the tail reflects late-stage drift behavior
- `seg_3` is intended to preserve more long-horizon characteristics for long recordings

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# autotune.py

def build_segment(len: int, hz: int,
                  head_s: float, tail_s: float, stride_s: float, win_s: float
                  ) -> list[tuple[int, int]]:
        . . .
        seg: list[tuple[int, int]] = []
        seg.append((0, min(head, len)))
        . . .
        start: int = stride
        while start < tail_start:
                end: int = min(start + win, tail_start)
                . . .
		seg.append((start, end))
                start += stride
        . . .
	seg.append((tail_start, len))
        return seg


# 04_exp_4_seg.ipynb

with Tee(path_log):
	. . .
	seg_1 = build_segment(len(dt), 100, 5, 5, 15, 3)
	print("[seg_1] head: 5, tail: 5, stride: 15, win: 3")

	seg_2 = build_segment(len(dt), 100, 10, 10, 15, 5)
	print("[seg_2] head: 10, tail: 10, stride: 15, win: 5")

	seg_3: list[tuple[int, int]] = None
	if len(dt) > 30*60*100:
		seg_3 = build_segment(len(dt), 100,30, 30, 20, 10)
		print("[seg_3] head: 30, tail: 30, stride: 20, win: 10")
```

</details>

<br>
<br>
<br>
<br>

### Method <a name="exp-4-method"></a>

### Redefine the Best-Model Criterion <a name="exp-4-method-best"></a>

A major change in Experiment 4 is the definition of the `best` result.<br>

<br>

In Experiment 3, the best configuration per dataset was selected mainly from the orientation-angle-error criterion.<br>
However, on Dataset 04, it was observed that orientation error improved while the secondary validation became worse (gravity / linear-acceleration estimation degraded).<br>
This showed that lower orientation error does not always imply a better downstream solution.<br>

<br>

Therefore, in Experiment 4, the best experiment is selected using a combined score:<br>

```
total_score = ori_score + g_score + a_score

ori_score: orientation angle error score
g_score: gravity direction error score
a_score: linear-acceleration direction error score
```

<br>

This revised criterion is intended to prefer configurations that are not only good in orientation agreement, but also more reliable for gravity / linear-acceleration decomposition.<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# parse_log_exp3.py

def run_exp3_from_param(. . .) -> float:
        . . .
        angle_err_exp3: ScalarBatch = calc_angle_err(q_est_exp3, q_ref)

        ori_score: float = score_angle_err(angle_err_exp3)
        g_score: float = score_angle_err(calc_vec3_direction_angle_err(g_body_est_exp3, g_ref))
        a_score: float = score_angle_err(calc_vec3_direction_angle_err(a_lin_est_exp3, a_lin_ref))
        total_score: float = ori_score + g_score + a_score
        . . .
        return total_score


# 04_exp_4_seg.ipynb

with Tee(path_log):
        . . .
        for i, c in enumerate(candidate, start=1):
                ori_score: float = score_angle_err(candidate[i - 1].angle_err)
		. . .
                g_score: float = score_angle_err(calc_vec3_direction_angle_err(g, g_ref_interp))
		. . .
                a_score: float = score_angle_err(calc_vec3_direction_angle_err(a, a_lin_ref_interp))
                total_score: float = ori_score + g_score + a_score
                . . .
                if total_score < best_score:
                        . . .
                        best_score = total_score
	. . .
```

</details>

<br>
<br>
<br>

### Hyperparameter Search with Optuna <a name="exp-4-method-optuna"></a>

#### [Optimization target]

| exp | trial | seed |             Target ∈ Range (min, max)             |
|:---:|------:|-----:|:--------------------------------------------------|
| 4-1 |    20 |   42 | <ul><li>tau ∈ (0.1, 4)</li><li>mag_gain ∈ (0.01, 10)</li></ul> |
| 4-2 |    20 |   42 | <ul><li>tau ∈ 4-1_best*(0.9, 1.1)</li><li>mag_gain ∈ 4-1_best*(0.7, 1.3)</li><li>mag_err_sigma ∈ (0.01, 2)</li></ul> |
| 4-3 |    30 |   42 | <ul><li>tau ∈ 4-2_best*(0.9, 1.1)</li><li>mag_gain ∈ 4-2_best*(0.7, 1.3)</li><li>acc_gate_sigma ∈ suggested_acc*(0.01, 10)</li><li>gyro_gate_sigma ∈ suggested_gyro*(0.1, 10)</li><li>mag_gate_sigma ∈ suggested_mag*(0.01, 10)</li></ul> |
| 4-4 |    20 |   42 | <ul><li>tau ∈ 4-3_best*(0.9, 1.1)</li><li>mag_gain ∈ 4-3_best*(0.7, 1.3)</li><li>acc_gate_sigma ∈ 4-3_best*(0.7, 1.3)</li><li>gyro_gate_sigma ∈ 4-3_best*(0.7, 1.3)</li><li>mag_gate_sigma ∈ 4-3_best*(0.7, 1.3)</li><li>mag_err_sigma ∈ 4-2_best*(0.5, 1.5)</li></ul> |
| 4-5 |    40 |   42 | <ul><li>tau ∈ 4-4_best*(0.9, 1.1)</li><li>mag_gain ∈ 4-4_best*(0.7, 1.3)</li><li>percentile `p` ∈ (50, 80)</li><li>sliding window size `win_s` ∈ (5, 10)</li><li>update ratio `update_ratio` ∈ (0.1, 0.5)</li><li>EMA factor `ema_alpha` ∈ (0.02, 0.2)</li></ul> |
| 4-6 |    40 |   42 | <ul><li>tau ∈ 4-5_best*(0.9, 1.1)</li><li>mag_gain ∈ 4-5_best*(0.7, 1.3)</li><li>percentile `p` ∈ (50, 80)</li><li>sliding window size `win_s` ∈ (5, 10)</li><li>update ratio `update_ratio` ∈ (0.1, 0.5)</li><li>EMA factor `ema_alpha` ∈ (0.02, 0.2)</li><li>mag_err_sigma ∈ (0.01, 2)</li></ul> |

<br>

** 4-x_best: Optimal value selected from experiment 4-x<br>
** x_gate_sigma: fixed gate sigma estimated from robust percentiles of the data<br>
** For exp 4-5 and exp 4-6, Optuna optimizes the parameters of the function `suggest_timevarying_gate_sigma(...)`, rather than the sigma values directly<br>

<br>

<details>
<summary><b><ins>Implementation</ins></b></summary>

```py
# optuna_exp_4.py

def exp_4_1(seg: list[tuple[int, int]], . . .) -> tuple[float, float, float]:
        def objective(trial):
                . . .
                score: float = calc_score_quasi_ori_from_seg(. . .)
                return score
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        . . .

def exp_4_2(seg: list[tuple[int, int]], . . .) -> tuple[float, ...]:
        . . .
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        . . .

def exp_4_3(seg: list[tuple[int, int]], . . .) -> tuple[float, ...]:
        . . .
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        . . .

def exp_4_4(seg: list[tuple[int, int]], . . .) -> tuple[float, ...]:
        . . .
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        . . .

def exp_4_5(seg: list[tuple[int, int]], . . .) -> tuple[Any, ...]:
        . . .
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        . . .

def exp_4_6(seg: list[tuple[int, int]], . . .) -> tuple[Any, ...]:
        . . .
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        . . .
```

</details>

<br>
<br>
<br>

### Experiment Structure <a name="exp-4-method-structure"></a>

Experiment 4 mirrors the structure of Experiment 3 as close as possible.<br>

<br>

For each Experiment 3 variant, a corresponding Experiment 4 variant is defined:<br>

- [exp 4-1] Gyro + Acc + Mag, without gating
- [exp 4-2] Gyro + Acc + Mag with Magnetometer innovation gating
- [exp 4-3] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating — fixed sigma
- [exp 4-4] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating + Magnetometer innovation gating — fixed sigma
- [exp 4-5] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating — time-varying sigma
- [exp 4-6] Gyro + Acc + Mag with Gyro/Acc/Mag norm gating + Magnetometer innovation gating — time-varying sigma

<br>

For a fair comparison with Experiment 3, each corresponding pair uses:<br>

- the same search space
- the same number of Optuna trials
- the same random seed
- the same dataset
- full-data evaluation after tuning

<br>

The only major difference is the tuning objective:<br>

- exp 3: full tuning → full evaluation
- exp 4: segment tuning → full evaluation

<br>

This allows the effect of the proxy objective itself to be isolated.<br>

<br>
<br>
<br>

### Evaluation and Measurements <a name="exp-4-method-eval"></a>

All final comparisons are performed on the full dataset, not on the tuning segments.<br>

<br>

For each experiment, the following are compared:<br>

- angle error statistics: min / max / mean / p90 (in rad and deg)
- total running time
- speedup relative to full-data tuning
- selected best-parameter differences relative to Experiment 3
- best experiment under the revised total score

<br>

This design is important because the goal is not merely to perform well on the segments,<br>
but to verify whether segment-based tuning preserves useful performance on the full sequence.<br>

<br>
<br>
<br>

### Why Segment-based Tuning Sometimes Works Better <a name="exp-4-method-why-seg-better"></as>

Although the main purpose of Experiment 4 is runtime reduction,<br>
it is also possible that segment-based tuning may sometimes produce slightly better full-sequence results.<br>

<br>

This may happen for several reasons:<br>

1. Reduced long-horizon artifact dominance

The optimization target is recursive, so long-horizon effects can dominate the score.<br>
Some parameters may appear good only because they match a specific late drift pattern, a particular disturbance regime, or a sequence-specific accumulated artifact.<br>
By cutting the sequence into segments, the proxy objective reduces the influence of such accumulated long-horizon behavior and may favor parameters that are locally more stable.<br>

<br>

2. Simpler optimization target under finite trials

Optuna does not search infinitely many trials.<br>
When the objective is complex and noisy, a limited number of trials may fail to approach a good basin.<br>
A segment-based objective can be computationally cheaper and also easier to optimize, so within the same finite trial budget it may sometimes produce a better parameter set.<br>

<br>

3. Segment reset emphasizes local fusion quality

In the segment-based scorer, each segment starts from `q0 = q_ref[seg_start]`.<br>
This means the objective focuses more on local estimation quality inside representative windows,<br>
rather than on the entire long accumulated trajectory.<br>
As a result, the selected parameters may better reflect local fusion robustness.<br>

<br>

4. Regularization-like effect

Segment-based tuning can behave like a weak regularizer.<br>
Instead of fitting the exact full-sequence trajectory, it fits a set of representative windows distributed over time.<br>
This may reduce overreaction to a specific portion of the dataset and lead to a parameter set that is more practically robust.<br>

<br>

5. Full-data objective is not always the most practically useful objective

The full sequence may contain very long easy intervals, repeated motion regimes, or prolonged disturbances that are not equally important for practical generalization.<br>
A distributed segment set may reduce this temporal bias by sampling representative regions more evenly.<br>

These possibilities do not imply that segment-based tuning is intrinsically superior to full-data tuning.<br>

<br>
<br>
<br>
<br>

### Results <a name="exp-4-res"></a>

Each plot compares:<br>

- blue: exp 1-2
- orange: best exp 2
- green: best exp 3
- red: best exp 4

### Dataset 01 — 5 min <a name="exp-4-res-data-01"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data01_exp4.png" width="952" height="311">

#### [Chosen parameters]

- quasi-static: (2523, 3540, 1017)
- suggested σ_gyro: 0.4479197
- suggested σ_acc : 2.9085688
- suggested σ_mag : 6.8504558
- `seg_1` head: 5, tail: 5, stride: 15, win: 3
- `seg_2` head: 10, tail: 10, stride: 15, win: 5

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 18:05:00.298

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (2523, 3540, 1017)

Suggested gyro_sigma:  0.44791971689543864
Suggested acc_sigma:  2.908568806018301
Suggested mag_sigma:  6.850455808768257

[seg_1] head: 5, tail: 5, stride: 15, win: 3
[seg_2] head: 10, tail: 10, stride: 15, win: 5

[END] 2026-03-26 18:05:01.044
```

</details>

<br>

##### [exp 4-1]

|              |     exp 3-1    |    exp 4-1-1   |    exp 4-1-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   16:47:11.139 |   18:05:01.116 |   18:05:44.357 |
|   end time   |   16:50:30.032 |   18:05:44.357 |   18:06:50.944 |
| running time |   00:03:18.893 |   00:00:43.241 |   00:01:06.587 |
|    speedup   |              - | 4.60× (−02m35s)| 3.00× (-02m12s)|
|      tau     |           3.93 |           3.93 |           3.93 |
|       K      |    0.002542621 |    0.002542621 |    0.002542621 |
|   mag_gain   |       3.577178 |       3.577178 |       3.577178 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.04217 rad</li><li>2.41633 deg</li></ul> | <ul><li>0.04217 rad</li><li>2.41633 deg</li></ul> | <ul><li>0.04217 rad</li><li>2.41633 deg</li></ul> |
|   p90 error  | <ul><li>0.08175 rad</li><li>4.68411 deg</li></ul> | <ul><li>0.08175 rad</li><li>4.68411 deg</li></ul> | <ul><li>0.08175 rad</li><li>4.68410 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-1) - (running time of exp 4-1-X)

<br>

<details>
<summary><b><ins>Logs exp 3-1</ins></b></summary>

```
[START] 2026-03-26 16:47:11.139


[chosen value]
tau= 3.932469060126895 , K= 0.0025426212244773093
mag_gain= 3.5771776662400905
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.002818835027033219 0.7172417482362367 0.042172969711376365 0.08175313998327868

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.16150735019264875 41.09492506452112 2.416333173994919 4.68410988298409

[END] 2026-03-26 16:50:30.032
```

</details>

<details>
<summary><b><ins>Logs exp 4-1</ins></b></summary>

```
[START] 2026-03-26 18:05:01.116

start :  2026-03-26 18:05:01.116

[chosen value]
tau= 3.932469060126895 , K= 0.0025426212244773093
mag_gain= 3.5771776662400905
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.002818835027033219 0.7172417482362367 0.042172969711376365 0.08175313998327868

[exp 4-1-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.16150735019264875 41.09492506452112 2.416333173994919 4.68410988298409

end :  2026-03-26 18:05:44.357



start :  2026-03-26 18:05:44.357

[chosen value]
tau= 3.932469060126895 , K= 0.0025426212244773093
mag_gain= 3.5771776662400905
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-2] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.002818835027033219 0.7172417482362367 0.042172969711376365 0.08175313998327868

[exp 4-1-2] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.16150735019264875 41.09492506452112 2.416333173994919 4.68410988298409

end :  2026-03-26 18:06:50.944




[END] 2026-03-26 18:06:50.944
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   16:50:30.047 |   18:06:50.961 |   18:07:34.493 |
|   end time   |   16:53:51.713 |   18:07:34.493 |   18:08:41.699 |
| running time |   00:03:21.666 |   00:00:43.532 |   00:01:07.206 |
|    speedup   |              - | 4.67× (−02m38s)| 3.00× (-02m14s)|
|      tau     |           4.37 |           4.31 |           4.32 |
|       K      |    0.002290369 |    0.002318593 |    0.002311993 |
|   mag_gain   |       4.124038 |       3.586378 |       3.388323 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |      0.7518797 |      1.7999687 |      0.7612136 |
|  Mean error  | <ul><li>0.04447 rad</li><li>2.54800 deg</li></ul> | <ul><li>0.04131 rad</li><li>2.36665 deg</li></ul> | <ul><li>0.04144 rad</li><li>2.37406 deg</li></ul> |
|   p90 error  | <ul><li>0.08333 rad</li><li>4.77430 deg</li></ul> | <ul><li>0.07946 rad</li><li>4.55272 deg</li></ul> | <ul><li>0.08010 rad</li><li>4.58945 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-2) - (running time of exp 4-2-X)

<br>

<details>
<summary><b><ins>Logs exp 3-2</ins></b></summary>

```
[START] 2026-03-26 16:50:30.047


[chosen value]
tau= 4.282566197670456 , K= 0.002334763512194607
mag_gain= 3.519929920245394
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=1.6224342

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.00280806873989446 0.6962182150580377 0.04136929699102485 0.07967920247718345

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.1608904873785719 39.89036534295706 2.37028611900898 4.565282016910945

[END] 2026-03-26 16:53:51.713
```

</details>

<details>
<summary><b><ins>Logs exp 4-2</ins></b></summary>

```
[START] 2026-03-26 18:06:50.961

start :  2026-03-26 18:06:50.961

[chosen value]
tau= 4.312433686162575 , K= 0.0023185931714062844
mag_gain= 3.5863784326810215
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=1.7999687

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0028076995829799255 0.6989043582864635 0.04130586062209385 0.07945992702475674

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.160869336245391 40.044270013113504 2.3666514828015988 4.552718458936074

end :  2026-03-26 18:07:34.493



start :  2026-03-26 18:07:34.493

[chosen value]
tau= 4.324743961186654 , K= 0.0023119933541997347
mag_gain= 3.3883234350286653
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.7612136

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0028060746198260754 0.6459638822521032 0.0414352367068872 0.08010108897878507

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.16077623271481112 37.01100417093118 2.3740641864301844 4.589454332886258

end :  2026-03-26 18:08:41.699




[END] 2026-03-26 18:08:41.699
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   16:53:51.728 |   18:08:41.719 |   18:09:52.109 |
|   end time   |   16:59:42.931 |   18:09:52.109 |   18:11:42.449 |
| running time |   00:05:51.203 |   00:01:10.390 |   00:01:50.340 |
|    speedup   |              - | 5.01× (−04m40s)| 3.19× (-04m00s)|
|      tau     |           4.54 |           4.76 |           4.67 |
|       K      |    0.002202910 |    0.002102056 |    0.002142129 |
|   mag_gain   |       3.892036 |       3.846369 |       3.674369 |
|     σ_acc    |      1.7562440 |      9.9262628 |      5.7291508 |
|    σ_gyro    |      6.2834979 |      0.2901186 |      0.4576878 |
|     σ_mag    |     16.2250324 |     44.4812203 |     39.1623427 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.03320 rad</li><li>1.90235 deg</li></ul> | <ul><li>0.03703 rad</li><li>2.12167 deg</li></ul> | <ul><li>0.03609 rad</li><li>2.06771 deg</li></ul> |
|   p90 error  | <ul><li>0.08865 rad</li><li>5.07946 deg</li></ul> | <ul><li>0.05096 rad</li><li>2.92002 deg</li></ul> | <ul><li> rad</li><li>3.08194 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-3) - (running time of exp 4-3-X)

<br>

<details>
<summary><b><ins>Logs exp 3-3</ins></b></summary>

```
[START] 2026-03-26 16:53:51.728


[chosen value]
tau= 4.572570503039086 , K= 0.0021866867422237557
mag_gain= 4.397713410255224
acc_gate_sigma=20.4765482
gyro_gate_sigma=1.1233809
mag_gate_sigma=25.3407412
mag_err_sigma=inf

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0022300678365621764 0.5431561440288503 0.038150425089703295 0.07351459839444256

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.12777347506288297 31.12055466945299 2.1858583442700037 4.2120762206007765

[END] 2026-03-26 16:59:42.931
```

</details>

<details>
<summary><b><ins>Logs exp 4-3</ins></b></summary>

```
[START] 2026-03-26 18:08:41.719

start :  2026-03-26 18:08:41.719

[chosen value]
tau= 4.756666899378352 , K= 0.002102055811010377
mag_gain= 3.846368915623853
acc_gate_sigma=9.9262628
gyro_gate_sigma=0.2901186
mag_gate_sigma=44.4812203
mag_err_sigma=inf

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0020093365570454944 0.655451638793824 0.03703012581309122 0.05096400471693658

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.11512650434005461 37.55461257781941 2.121669923928573 2.9200223773652856

end :  2026-03-26 18:09:52.109



start :  2026-03-26 18:09:52.109

[chosen value]
tau= 4.667684021939419 , K= 0.0021421285695179715
mag_gain= 3.6743687405123153
acc_gate_sigma=5.7291508
gyro_gate_sigma=0.4576878
mag_gate_sigma=39.1623427
mag_err_sigma=inf

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0007536155275271733 0.6100054306376016 0.036088372773354735 0.05379003003914591

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.043178989102832147 34.95073665559485 2.067711449408056 3.081941701124979

end :  2026-03-26 18:11:42.449
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   16:59:42.948 |   18:11:42.467 |   18:12:33.388 |
|   end time   |   17:03:45.382 |   18:12:33.387 |   18:13:51.416 |
| running time |   00:04:02.434 |   00:00:50.920 |   00:01:18.028 |
|    speedup   |              - | 4.84× (−03m11s)| 3.10× (-02m44s)|
|      tau     |           4.19 |           5.02 |           5.01 |
|       K      |    0.002387545 |    0.001993253 |    0.001996605 |
|   mag_gain   |       4.519486 |       4.367921 |       4.148098 |
|     σ_acc    |      1.8465124 |      5.0794965 |      5.1120395 |
|    σ_gyro    |      6.8600581 |      0.3945917 |      0.3595636 |
|     σ_mag    |     13.1831565 |     27.5380465 |     37.0785850 |
|   σ_mag_err  |      0.8853287 |      0.8856901 |      0.8695816 |
|  Mean error  | <ul><li>0.03020 rad</li><li>1.73052 deg</li></ul> | <ul><li>0.03390 rad</li><li>1.94246 deg</li></ul> | <ul><li>0.03458 rad</li><li>1.98155 deg</li></ul> |
|   p90 error  | <ul><li>0.07193 rad</li><li>4.12132 deg</li></ul> | <ul><li>0.04893 rad</li><li>2.80326 deg</li></ul> | <ul><li>0.04782 rad</li><li>2.73973 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-4) - (running time of exp 4-4-X)

<br>

<details>
<summary><b><ins>Logs exp 3-4</ins></b></summary>

```
[START] 2026-03-26 16:59:42.948


[chosen value]
tau= 4.7699779116997645 , K= 0.002096189852861593
mag_gain= 5.678210094405326
acc_gate_sigma=15.8903128
gyro_gate_sigma=0.9994370
mag_gate_sigma=18.0364040
mag_err_sigma=1.5374059

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0017333672375821516 0.5303552091541307 0.037937464957498106 0.07606885323359422

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.09931462705970755 30.387115127309734 2.1736566274900984 4.358424242685033

[END] 2026-03-26 17:03:45.382
```

</details>

<details>
<summary><b><ins>Logs exp 4-4</ins></b></summary>

```
[START] 2026-03-26 18:11:42.466

start :  2026-03-26 18:11:42.467

[chosen value]
tau= 5.016312670696481 , K= 0.001993252804054321
mag_gain= 4.367920821449954
acc_gate_sigma=5.0794965
gyro_gate_sigma=0.3945917
mag_gate_sigma=27.5380465
mag_err_sigma=0.8856901

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0006888292569035345 0.5534559532416731 0.033902277492087594 0.048926046078887606

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.03946700922570526 31.7106902671377 1.9424574161779844 2.80325594858285

end :  2026-03-26 18:12:33.387



start :  2026-03-26 18:12:33.388

[chosen value]
tau= 5.007891116670216 , K= 0.0019966047711371254
mag_gain= 4.148098120776027
acc_gate_sigma=5.1120395
gyro_gate_sigma=0.3595636
mag_gate_sigma=37.0785850
mag_err_sigma=0.8695816

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0006503528769452465 0.6221986265349649 0.03458455384894321 0.04781733703074702

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.03726247504315361 35.64935531929 1.981548971887373 2.7397315994164275

end :  2026-03-26 18:13:51.416




[END] 2026-03-26 18:13:51.417
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   17:03:45.400 |   18:13:51.435 |   18:15:27.488 |
|   end time   |   17:11:37.212 |   18:15:27.488 |   18:17:58.874 |
| running time |   00:07:51.812 |   00:01:36.053 |   00:02:31.386 |
|    speedup   |              - | 4.91× (−06m15s)| 3.12× (-05m20s)|
|      tau     |           3.79 |           5.22 |           5.49 |
|       K      |    0.002635225 |    0.001914986 |    0.001820333 |
|   mag_gain   |       5.460352 |       5.207657 |       4.652301 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             80 |             80 |             80 |
|     win_s    |       9.996655 |       9.244633 |       9.148510 |
| update_ratio |       0.194599 |       0.411771 |       0.291020 |
|   ema_alpha  |       0.096770 |       0.047866 |       0.020566 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.04852 rad</li><li>2.78041 deg</li></ul> | <ul><li>0.05663 rad</li><li>3.24449 deg</li></ul> | <ul><li>0.06079 rad</li><li>3.48297 deg</li></ul> |
|   p90 error  | <ul><li>0.09915 rad</li><li>5.68096 deg</li></ul> | <ul><li>0.10513 rad</li><li>6.02340 deg</li></ul> | <ul><li>0.11554 rad</li><li>6.62021 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-5) - (running time of exp 4-5-X)

<br>

<details>
<summary><b><ins>Logs exp 3-5</ins></b></summary>

```
[START] 2026-03-26 17:03:45.400


[chosen value]
tau= 5.173698030271023 , K= 0.0019326174891492838
mag_gain= 4.930731964833801
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 76 , best_win_s= 5.777591402642944
best_update_ratio= 0.39546840952149465 , best_ema_alpha= 0.050896341548929844
mag_err_sigma=inf

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002409636697114314 0.7816545685336467 0.05958371638396706 0.1093147442364582

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.13806201290449366 44.785507814097315 3.4138954765058074 6.263273483301096

[END] 2026-03-26 17:11:37.212
```

</details>

<details>
<summary><b><ins>Logs exp 4-5</ins></b></summary>

```
[START] 2026-03-26 18:13:51.435

start :  2026-03-26 18:13:51.435

[chosen value]
tau= 5.221331923598967 , K= 0.0019149863374299725
mag_gain= 5.207656843830228
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 9.244633285867781
best_update_ratio= 0.4117712809246715 , best_ema_alpha= 0.04786559441531377
mag_err_sigma=inf

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002721592801659664 0.7823485285013769 0.05662707153250685 0.10512824833904005

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.1559357810882841 44.825268791399296 3.2444922049980534 6.023404937430202

end :  2026-03-26 18:15:27.488



start :  2026-03-26 18:15:27.488

[chosen value]
tau= 5.4928298272975455 , K= 0.0018203329815878067
mag_gain= 4.6523005715228685
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 9.148509985104303
best_update_ratio= 0.2910204473093046 , best_ema_alpha= 0.020566329226358745
mag_err_sigma=inf

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0018884649692533593 0.7995658913621941 0.06078932824448653 0.11554452753733846

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.10820107249652026 45.81175101766941 3.482971947844488 6.620213773722613

end :  2026-03-26 18:17:58.874




[END] 2026-03-26 18:17:58.874
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   17:11:37.225 |   18:17:58.896 |   18:19:36.642 |
|   end time   |   17:19:36.843 |   18:19:36.642 |   18:22:10.496 |
| running time |   00:07:59.618 |   00:01:37.746 |   00:02:33.854 |
|    speedup   |              - | 4.94× (−06m21s)| 3.13× (-05m25s)|
|      tau     |           3.56 |           5.09 |           5.01 |
|       K      |    0.002812494 |    0.001962530 |    0.001996456 |
|   mag_gain   |       6.998810 |       6.025085 |       5.866344 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             79 |             78 |             78 |
|     win_s    |       9.474137 |       8.146240 |       7.546818 |
| update_ratio |       0.339160 |       0.418640 |       0.412702 |
|   ema_alpha  |       0.185937 |       0.066146 |       0.023297 |
|   σ_mag_err  |      1.5525143 |      1.3935356 |      1.4270680 |
|  Mean error  | <ul><li>0.04861 rad</li><li>2.78499 deg</li></ul> | <ul><li>0.05550 rad</li><li>3.17997 deg</li></ul> | <ul><li>0.05964 rad</li><li>3.41724 deg</li></ul> |
|   p90 error  | <ul><li>0.10116 rad</li><li>5.79594 deg</li></ul> | <ul><li>0.09866 rad</li><li>5.65289 deg</li></ul> | <ul><li>0.11592 rad</li><li>6.64166 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-6) - (running time of exp 4-6-X)

<br>

<details>
<summary><b><ins>Logs exp 3-6</ins></b></summary>

```
[START] 2026-03-26 17:11:37.225


[chosen value]
tau= 5.16003687902853 , K= 0.0019377340765753268
mag_gain= 5.600152346494399
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 5.046138345222255
best_update_ratio= 0.2659964606854568 , best_ema_alpha= 0.025141451103732223
mag_err_sigma=1.9951553608116073

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0017723107976843774 0.7672592415278287 0.05703574444543082 0.10546355981071441

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.10154592869277913 43.96071633195324 3.267907438109914 6.042616869579463

[END] 2026-03-26 17:19:36.843
```

</details>

<details>
<summary><b><ins>Logs exp 4-6</ins></b></summary>

```
[START] 2026-03-26 18:17:58.896

start :  2026-03-26 18:17:58.896

[chosen value]
tau= 5.09484111682139 , K= 0.001962530149147634
mag_gain= 6.0250852848497845
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 78 , best_win_s= 8.146240296679151
best_update_ratio= 0.4186404782579588 , best_ema_alpha= 0.06614602399076905
mag_err_sigma=1.3935355895818147

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0026660231486258976 0.803228592070217 0.05550101897374265 0.09866147870295279

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.15275187450044295 46.0216083098587 3.179974145870957 5.65288633019905

end :  2026-03-26 18:19:36.642



start :  2026-03-26 18:19:36.642

[chosen value]
tau= 5.008264829716271 , K= 0.0019964557859544004
mag_gain= 5.866343555506204
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 78 , best_win_s= 7.546817548467518
best_update_ratio= 0.41270197351062915 , best_ema_alpha= 0.0232973084065637
mag_err_sigma=1.4270679915391082

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.002494842814012129 0.8035655209478598 0.05964201474016935 0.11591886618856465

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.1429439637914368 46.04091291254371 3.417235726268749 6.641661798546493

end :  2026-03-26 18:22:10.496




[END] 2026-03-26 18:22:10.496
```

</details>

<br>

##### [best]

|              |      exp 3      | exp 4 (`seg_2`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-4      |     exp4-4      |
|  total_best  |     exp3-4      |     exp4-4      |

<br>

<details>
<summary><b><ins>Logs exp 3</ins></b></summary>

```
[START] 2026-03-26 17:19:40.614

best: exp3-4

[END] 2026-03-26 17:19:40.631
```

</details>

<details>
<summary><b><ins>Logs exp 4</ins></b></summary>

```
[START] 2026-03-26 18:22:10.510

exp 3-1: total_score=3.3566744 | ori=0.1218349, g=2.8178185, a=0.4170210
exp 3-2: total_score=3.3482921 | ori=0.1176081, g=2.8178604, a=0.4128236
exp 3-3: total_score=3.3312404 | ori=0.1065418, g=2.8182785, a=0.4064201
exp 3-4: total_score=3.3266962 | ori=0.1030058, g=2.8184694, a=0.4052210
exp 3-5: total_score=3.4440627 | ori=0.2225412, g=2.8213231, a=0.4001984
exp 3-6: total_score=3.4327346 | ori=0.2206603, g=2.8205616, a=0.3915126

ori_best: exp3-4
total_best: exp3-4

[END] 2026-03-26 18:23:05.216




[START] 2026-03-26 18:23:05.231

[seg_2]
exp 4-1: total_score=0.5773476 | ori=0.1218349, g=0.0384917, a=0.4170210
exp 4-2: total_score=0.5725357 | ori=0.1232297, g=0.0369855, a=0.4123204
exp 4-3: total_score=0.5301036 | ori=0.1102111, g=0.0277979, a=0.3920946
exp 4-4: total_score=0.5232454 | ori=0.1093450, g=0.0261144, a=0.3877860
exp 4-5: total_score=0.6670798 | ori=0.2291574, g=0.0332575, a=0.4046649
exp 4-6: total_score=0.6628805 | ori=0.2276878, g=0.0326517, a=0.4025410

best: exp4-4

[END] 2026-03-26 18:23:05.833
```

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp4]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 3</th>
      <th colspan="4">best exp 4</th>
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
      <td>0.57</td>
      <td>6.28</td>
      <td>0.85</td>
      <td>1.16</td>
      <td>8.00</td>
      <td>18.05</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 18:23:06.882

[Gravity]
RMSE norm: 0.17186439968691072

Gravity est/ref angle error in rad — min/max/mean/p90
0.0001940564580462738 0.0955435852519144 0.014845698265978232 0.020237530803624882

Gravity est/ref angle error in deg — min/max/mean/p90
0.011118616033309013 5.474244194483072 0.8505958545652373 1.159525102813703


[Linear accel]
RMSE norm: 0.7310720809634389

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00021757833597491768 3.040019383571576 0.13954152389176744 0.315040627008148

Linear accel est/ref angle error in deg — min/max/mean/p90
0.01246632036484223 174.18028031661345 7.995140385822216 18.050498302722055
. . .
[END] 2026-03-26 18:23:07.378
```

</details>

<br>

#### [Observation]

<br>
<br>
<br>

### Dataset 02 — 9 min <a name="exp-4-res-data-02"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data02_exp4.png" width="952" height="311">

#### [Chosen parameters]

- quasi-static: (0, 1635, 1635)
- suggested σ_gyro: 0.6304231
- suggested σ_acc : 2.5176967
- suggested σ_mag : 5.0510618
- `seg_1` head: 5, tail: 5, stride: 15, win: 3
- `seg_2` head: 10, tail: 10, stride: 15, win: 5

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-25 23:26:13.239

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (0, 1635, 1635)

Suggested gyro_sigma:  0.6304230586083284
Suggested acc_sigma:  2.5176966756605874
Suggested mag_sigma:  5.051061836028509

[seg_1] head: 5, tail: 5, stride: 15, win: 3
[seg_2] head: 10, tail: 10, stride: 15, win: 5

[END] 2026-03-25 23:26:14.389
```

</details>

<br>

##### [exp 4-1]

|              |     exp 3-1    |    exp 4-1-1   |    exp 4-1-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:23:46.312 |    |    |
|   end time   |   21:28:40.496 |    |    |
| running time |   00:04:54.184 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           3.97 |            |            |
|       K      |    0.002518346 |     |     |
|   mag_gain   |       3.724649 |        |        |
|     σ_acc    |            inf |             |             |
|    σ_gyro    |            inf |             |             |
|     σ_mag    |            inf |             |             |
|   σ_mag_err  |            inf |             |             |
|  Mean error  | <ul><li>0.04564 rad</li><li>2.61517 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.08512 rad</li><li>4.87730 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-1) - (running time of exp 4-1-X)

<br>

<details>
<summary><b><ins>Logs exp 3-1</ins></b></summary>

```
[START] 2026-03-25 21:23:46.312


[chosen value]
tau= 3.9703762188793768 , K= 0.0025183455535897066
mag_gain= 3.724648901612812
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0014937441061020549 0.3703654566537466 0.04564329581096676 0.08512497175927315

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.08558523295218959 21.220377543695115 2.6151682130355454 4.877301612976674

[END] 2026-03-25 21:28:40.496
```

</details>

<details>
<summary><b><ins>Logs exp 4-1</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:28:40.507 |    |    |
|   end time   |   21:33:33.667 |    |    |
| running time |   00:04:53.160 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           4.37 |            |            |
|       K      |    0.002290369 |     |     |
|   mag_gain   |       4.124038 |        |        |
|     σ_acc    |            inf |             |             |
|    σ_gyro    |            inf |             |             |
|     σ_mag    |            inf |             |             |
|   σ_mag_err  |      0.7518797 |             |             |
|  Mean error  | <ul><li>0.04447 rad</li><li>2.54800 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.08333 rad</li><li>4.77430 deg</li></ul>  | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-2) - (running time of exp 4-2-X)

<br>

<details>
<summary><b><ins>Logs exp 3-2</ins></b></summary>

```
[START] 2026-03-25 21:28:40.507


[chosen value]
tau= 4.365576053036914 , K= 0.002290368825423975
mag_gain= 4.124038246451787
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.7518797

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0009744398208139917 0.3921464219975504 0.044471033910069374 0.08332721312806343

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.05583128912212591 22.468334931615782 2.548002553630142 4.774297630825141

[END] 2026-03-25 21:33:33.667
```

</details>

<details>
<summary><b><ins>Logs exp 4-2</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:33:33.679 |    |    |
|   end time   |   21:44:20.840 |    |    |
| running time |   00:10:47.161 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           4.54 |            |            |
|       K      |    0.002202910 |     |     |
|   mag_gain   |       3.892036 |        |        |
|     σ_acc    |      1.7562440 |             |             |
|    σ_gyro    |      6.2834979 |             |             |
|     σ_mag    |     16.2250324 |             |             |
|   σ_mag_err  |            inf |             |             |
|  Mean error  | <ul><li>0.03320 rad</li><li>1.90235 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.08865 rad</li><li>5.07946 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-3) - (running time of exp 4-3-X)

<br>

<details>
<summary><b><ins>Logs exp 3-3</ins></b></summary>

```
[START] 2026-03-25 21:33:33.679


[chosen value]
tau= 4.5388961519628355 , K= 0.0022029099063148295
mag_gain= 3.892036037781301
acc_gate_sigma=1.7562440
gyro_gate_sigma=6.2834979
mag_gate_sigma=16.2250324
mag_err_sigma=inf

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0006318141187635655 0.4940754860093368 0.03320228894818098 0.08865336899765028

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.03620028244192965 28.308440109209954 1.9023510269046275 5.079463883181298

[END] 2026-03-25 21:44:20.840
```

</details>

<details>
<summary><b><ins>Logs exp 4-3</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:44:20.855 |    |    |
|   end time   |   21:53:00.866 |    |    |
| running time |   00:08:40.011 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           4.19 |            |            |
|       K      |    0.002387545 |     |     |
|   mag_gain   |       4.519486 |        |        |
|     σ_acc    |      1.8465124 |             |             |
|    σ_gyro    |      6.8600581 |             |             |
|     σ_mag    |     13.1831565 |             |             |
|   σ_mag_err  |      0.8853287 |             |             |
|  Mean error  | <ul><li>0.03020 rad</li><li>1.73052 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.07193 rad</li><li>4.12132 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-4) - (running time of exp 4-4-X)

<br>

<details>
<summary><b><ins>Logs exp 3-4</ins></b></summary>

```
[START] 2026-03-25 21:44:20.855


[chosen value]
tau= 4.187892089899052 , K= 0.0023875446363600564
mag_gain= 4.519486446118292
acc_gate_sigma=1.8465124
gyro_gate_sigma=6.8600581
mag_gate_sigma=13.1831565
mag_err_sigma=0.8853287

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0003528782380093778 0.5061348804237038 0.030203112347491224 0.07193067933840304

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.020218433719950298 28.99939251263682 1.7305108656707113 4.121324343599367

[END] 2026-03-25 21:53:00.866
```

</details>

<details>
<summary><b><ins>Logs exp 4-4</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:53:00.885 |    |    |
|   end time   |   22:08:27.826 |    |    |
| running time |   00:15:26.941 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           3.79 |            |            |
|       K      |    0.002635225 |     |     |
|   mag_gain   |       5.460352 |        |        |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             80 |              |              |
|     win_s    |       9.996655 |        |        |
| update_ratio |       0.194599 |        |        |
|   ema_alpha  |       0.096770 |        |        |
|   σ_mag_err  |            inf |             |             |
|  Mean error  | <ul><li>0.04852 rad</li><li>2.78041 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.09915 rad</li><li>5.68096 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-5) - (running time of exp 4-5-X)

<br>

<details>
<summary><b><ins>Logs exp 3-5</ins></b></summary>

```
[START] 2026-03-25 21:53:00.885


[chosen value]
tau= 3.7942796171771516 , K= 0.002635224681815103
mag_gain= 5.460351841605867
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 9.9966554989337
best_update_ratio= 0.194599378050732 , best_ema_alpha= 0.09677042284269513
mag_err_sigma=inf

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.00309411337562138 0.41506042495390216 0.0485273452437651 0.09915149694582216

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.17727963775808145 23.78121059276503 2.780412073441989 5.680962307399882

[END] 2026-03-25 22:08:27.826
```

</details>

<details>
<summary><b><ins>Logs exp 4-5</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:08:27.842 |    |    |
|   end time   |   22:21:16.698 |    |    |
| running time |   00:12:48.856 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           3.56 |            |            |
|       K      |    0.002812494 |     |     |
|   mag_gain   |       6.998810 |        |        |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             79 |              |              |
|     win_s    |       9.474137 |        |        |
| update_ratio |       0.339160 |        |        |
|   ema_alpha  |       0.185937 |        |        |
|   σ_mag_err  |      1.5525143 |             |             |
|  Mean error  | <ul><li>0.04861 rad</li><li>2.78499 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.10116 rad</li><li>5.79594 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-6) - (running time of exp 4-6-X)

<br>

<details>
<summary><b><ins>Logs exp 3-6</ins></b></summary>

```
[START] 2026-03-25 22:08:27.842


[chosen value]
tau= 3.5551295540084142 , K= 0.002812493650370505
mag_gain= 6.998810213808275
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 79 , best_win_s= 9.474136752138243
best_update_ratio= 0.3391599915244341 , best_ema_alpha= 0.18593736230416105
mag_err_sigma=1.5525143184886179

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.001851477041226652 0.43401756316313544 0.048607184841248534 0.10115831861615807

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.10608182032765627 24.86737460380029 2.7849865454158134 5.795944719345523

[END] 2026-03-25 22:21:16.698
```

</details>

<details>
<summary><b><ins>Logs exp 4-6</ins></b></summary>

```
```

</details>

<br>

##### [best]

|              |       exp 3     | exp 4 (`seg_2`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-4      |                 |
|  total_best  |                 |                 |

<br>

<details>
<summary><b><ins>Logs exp 3</ins></b></summary>

```
[START] 2026-03-25 22:21:23.237

best: exp3-4

[END] 2026-03-25 22:21:23.260
```

</details>

<details>
<summary><b><ins>Logs exp 4</ins></b></summary>

```
```

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp4]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 3</th>
      <th colspan="4">best exp 4</th>
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
      <td>0.57</td>
      <td>6.28</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
```

</details>

<br>

#### [Observation]

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-4-res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data03_exp4.png" width="952" height="311">

#### [Chosen parameters]

- quasi-static: (41487, 43669, 2182)
- suggested σ_gyro: 0.4822681
- suggested σ_acc : 0.6755996
- suggested σ_mag : 5.0989239
- `seg_1` head: 5, tail: 5, stride: 15, win: 3
- `seg_2` head: 10, tail: 10, stride: 15, win: 5

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 00:07:59.033

Detected accel unit in [m/s²]
Selected g_world_unit: [ 0  0 -1]

Best quasi static(start, end, length):  (41487, 43669, 2182)

Suggested gyro_sigma:  0.48226806557261265
Suggested acc_sigma:  0.6755995626475786
Suggested mag_sigma:  5.098923949491506

[seg_1] head: 5, tail: 5, stride: 15, win: 3
[seg_2] head: 10, tail: 10, stride: 15, win: 5

[END] 2026-03-26 00:07:59.803
```

</details>

<br>

##### [exp 4-1]

|              |     exp 3-1    |    exp 4-1-1   |    exp 4-1-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:36:25.196 |    |    |
|   end time   |   22:44:31.294 |    |    |
| running time |   00:08:06.098 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           3.93 |            |            |
|       K      |    0.002542621 |     |     |
|   mag_gain   |       3.577178 |        |        |
|     σ_acc    |            inf |             |             |
|    σ_gyro    |            inf |             |             |
|     σ_mag    |            inf |             |             |
|   σ_mag_err  |            inf |             |             |
|  Mean error  | <ul><li>0.04335 rad</li><li>2.48355 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.07790 rad</li><li>4.46346 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-1) - (running time of exp 4-1-X)

<br>

<details>
<summary><b><ins>Logs exp 3-1</ins></b></summary>

```
[START] 2026-03-25 22:36:25.196


[chosen value]
tau= 3.932469060126895 , K= 0.002542621224480923
mag_gain= 3.5771776662400905
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0009205019993881509 0.4253150125622266 0.04334613532077613 0.0779020739570321

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.05274087959829493 24.368755183369174 2.4835506120834188 4.463460053053944

[END] 2026-03-25 22:44:31.294
```

</details>

<details>
<summary><b><ins>Logs exp 4-1</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:44:31.305 |    |    |
|   end time   |   22:52:58.524 |    |    |
| running time |   00:08:27.219 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           4.32 |            |            |
|       K      |    0.002311993 |     |     |
|   mag_gain   |       3.388323 |        |        |
|     σ_acc    |            inf |             |             |
|    σ_gyro    |            inf |             |             |
|     σ_mag    |            inf |             |             |
|   σ_mag_err  |      0.7612136 |             |             |
|  Mean error  | <ul><li>0.04219 rad</li><li>2.41742 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.07543 rad</li><li>4.32170 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-2) - (running time of exp 4-2-X)

<br>

<details>
<summary><b><ins>Logs exp 3-2</ins></b></summary>

```
[START] 2026-03-25 22:44:31.305


[chosen value]
tau= 4.324743961186654 , K= 0.0023119933542030207
mag_gain= 3.3883234350286653
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.7612136

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0010724171541030324 0.4038463545977173 0.04219187418047651 0.07542797577246489

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.06144497680753457 23.138691690192868 2.417416320288293 4.321704668977263

[END] 2026-03-25 22:52:58.524
```

</details>

<details>
<summary><b><ins>Logs exp 4-2</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:52:58.541 |    |    |
|   end time   |   23:08:04.911 |    |    |
| running time |   00:15:06.370 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           3.96 |            |            |
|       K      |    0.002523787 |     |     |
|   mag_gain   |       3.241114 |        |        |
|     σ_acc    |      2.0484861 |             |             |
|    σ_gyro    |      1.3120595 |             |             |
|     σ_mag    |     34.8068405 |             |             |
|   σ_mag_err  |            inf |             |             |
|  Mean error  | <ul><li>0.03418 rad</li><li>1.95857 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.06214 rad</li><li>3.56029 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-3) - (running time of exp 4-3-X)

<br>

<details>
<summary><b><ins>Logs exp 3-3</ins></b></summary>

```
[START] 2026-03-25 22:52:58.541


[chosen value]
tau= 3.9618151001265405 , K= 0.002523787467157119
mag_gain= 3.2411142281661203
acc_gate_sigma=2.0484861
gyro_gate_sigma=1.3120595
mag_gate_sigma=34.8068405
mag_err_sigma=inf

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0005615014128801123 0.25009170692673555 0.03418342702097173 0.0621387057555901

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.03217166114866312 14.329199298124642 1.9585660975951367 3.56028558420059

[END] 2026-03-25 23:08:04.911
```

</details>

<details>
<summary><b><ins>Logs exp 4-3</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   23:08:04.927 |    |    |
|   end time   |   23:20:05.288 |    |    |
| running time |   00:12:00.361 |    |    |
|    speedup   |              - | × (−) | × (-) |
|      tau     |           4.09 |            |            |
|       K      |    0.002444338 |     |     |
|   mag_gain   |       3.375010 |        |        |
|     σ_acc    |      2.3851260 |             |             |
|    σ_gyro    |      1.1860780 |             |             |
|     σ_mag    |     40.2588639 |             |             |
|   σ_mag_err  |      0.7412543 |             |             |
|  Mean error  | <ul><li>0.03677 rad</li><li>2.10667 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.06585 rad</li><li>3.77265 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-4) - (running time of exp 4-4-X)

<br>

<details>
<summary><b><ins>Logs exp 3-4</ins></b></summary>

```
[START] 2026-03-25 23:08:04.927


[chosen value]
tau= 4.090587639086756 , K= 0.00244433811938215
mag_gain= 3.375009970286213
acc_gate_sigma=2.3851260
gyro_gate_sigma=1.1860780
mag_gate_sigma=40.2588639
mag_err_sigma=0.7412543

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.001358626679512281 0.26715531641816487 0.03676829735821092 0.06584523358043846

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.07784357466992681 15.306872105242917 2.1066682585075 3.7726539852122065

[END] 2026-03-25 23:20:05.288
```

</details>

<details>
<summary><b><ins>Logs exp 4-4</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   23:20:05.307 |    |    |
|   end time   |   23:41:01.545 |    |    |
| running time |   00:20:56.238 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           4.45 |            |            |
|       K      |    0.002246526 |     |     |
|   mag_gain   |       3.298374 |        |        |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             80 |              |              |
|     win_s    |       8.373621 |        |        |
| update_ratio |       0.464217 |        |        |
|   ema_alpha  |       0.188064 |        |        |
|   σ_mag_err  |            inf |             |             |
|  Mean error  | <ul><li>0.06298 rad</li><li>3.60866 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.11860 rad</li><li>6.79522 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-5) - (running time of exp 4-5-X)

<br>

<details>
<summary><b><ins>Logs exp 3-5</ins></b></summary>

```
[START] 2026-03-25 23:20:05.307


[chosen value]
tau= 4.4507748928505535 , K= 0.0022465255012008366
mag_gain= 3.29837371025142
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 8.373621422596637
best_update_ratio= 0.46421729781274007 , best_ema_alpha= 0.18806376872844685
mag_err_sigma=inf

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0024426687498780323 0.74029348386357 0.06298306263195788 0.11859888418759126

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.13995461011650817 42.41569222641867 3.6086636696193133 6.7952155189098145

[END] 2026-03-25 23:41:01.545
```

</details>

<details>
<summary><b><ins>Logs exp 4-5</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   23:41:01.559 |    |    |
|   end time   |   00:02:29.468 |    |    |
| running time |   00:21:27.909 |    |    |
|    speedup   |              - | × (−)| × (-)|
|      tau     |           4.17 |            |            |
|       K      |    0.002397647 |     |     |
|   mag_gain   |       4.227693 |        |        |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             79 |              |              |
|     win_s    |       9.474137 |        |        |
| update_ratio |       0.339160 |        |        |
|   ema_alpha  |       0.185937 |        |        |
|   σ_mag_err  |      1.5525143 |             |             |
|  Mean error  | <ul><li>0.06038 rad</li><li>3.45952 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>0.11016 rad</li><li>6.31160 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-6) - (running time of exp 4-6-X)

<br>

<details>
<summary><b><ins>Logs exp 3-6</ins></b></summary>

```
[START] 2026-03-25 23:41:01.559


[chosen value]
tau= 4.170246517462413 , K= 0.0023976470587588733
mag_gain= 4.2276930648255115
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 79 , best_win_s= 9.474136752138243
best_update_ratio= 0.3391599915244341 , best_ema_alpha= 0.18593736230416105
mag_err_sigma=1.5525143184886179

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0014641836184843284 0.7190657547776307 0.060379996229155096 0.11015816122655236

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.08389154177134513 41.19943294114725 3.4595189509464124 6.311597717203118

[END] 2026-03-26 00:02:29.468
```

</details>

<details>
<summary><b><ins>Logs exp 4-6</ins></b></summary>

```
```

</details>

<br>

##### [best]

|              |       exp 3     | exp 4 (`seg_2`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-3      |                 |
|  total_best  |                 |                 |

<br>

<details>
<summary><b><ins>Logs exp 3</ins></b></summary>

```
[START] 2026-03-26 00:02:37.681

best: exp3-3

[END] 2026-03-26 00:02:37.704
```

</details>

<details>
<summary><b><ins>Logs exp 4</ins></b></summary>

```
```

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp4]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 3</th>
      <th colspan="4">best exp 4</th>
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
      <td>0.87</td>
      <td>6.75</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
```

</details>

<br>

#### [Observation]

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-4-res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data04_exp4.png" width="952" height="311">

#### [Chosen parameters]

- quasi-static: (252162, 310194, 58032)
- suggested σ_gyro: 0.2752885
- suggested σ_acc : 0.5070689
- suggested σ_mag : 190.9833338
- `seg_1` head: 5, tail: 5, stride: 15, win: 3
- `seg_2` head: 10, tail: 10, stride: 15, win: 5
- `seg_3` head: 30, tail: 30, stride: 20, win: 10

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 08:19:16.985

Detected accel unit in [g] → converting to [m/s²]
Selected g_world_unit: [0 0 1]

Best quasi static(start, end, length):  (252162, 310194, 58032)

Suggested gyro_sigma:  0.27528854262917185
Suggested acc_sigma:  0.507068924693965
Suggested mag_sigma:  190.98333381383583

[seg_1] head: 5, tail: 5, stride: 15, win: 3
[seg_2] head: 10, tail: 10, stride: 15, win: 5
[seg_3] head: 30, tail: 30, stride: 20, win: 10

[END] 2026-03-26 08:19:18.238
```

</details>

<br>

##### [exp 4-1]

|              |     exp 3-1    |    exp 4-1-1   |    exp 4-1-2   |    exp 4-1-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   00:08:48.169 |    |    |    |
|   end time   |   01:10:02.735 |    |    |    |
| running time |   01:01:14.566 |    |    |    |
|    speedup   |              - | × (−) | × (-)| × (-)|
|      tau     |           2.28 |            |            |            |
|       K      |    0.004408465 |     |     |     |
|   mag_gain   |       0.075637 |        |        |        |
|     σ_acc    |            inf |             |             |             |
|    σ_gyro    |            inf |             |             |             |
|     σ_mag    |            inf |             |             |             |
|   σ_mag_err  |            inf |             |             |             |
|  Mean error  | <ul><li>0.77836 rad</li><li>44.59698 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>1.99740 rad</li><li>114.44248 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-1) - (running time of exp 4-1-X)

<br>

<details>
<summary><b><ins>Logs exp 3-1</ins></b></summary>

```
[START] 2026-03-26 00:08:48.169


[chosen value]
tau= 2.281709702042365 , K= 0.0044084652319855465
mag_gain= 0.07563695396781411
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 3-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0010410408266554162 3.141540450466659 0.778364216206735 1.9973979935476887

[exp 3-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.059647245668165684 179.99700898136703 44.59698451265423 114.4424750381814

[END] 2026-03-26 01:10:02.735
```

</details>

<details>
<summary><b><ins>Logs exp 4-1</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |    exp 4-2-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   01:10:02.746 |    |    |    |
|   end time   |   02:02:03.748 |    |    |    |
| running time |   00:52:01.002 |    |    |    |
|    speedup   |              - | × (−)| × (-)| × (-)|
|      tau     |           2.44 |            |            |            |
|       K      |    0.004115562 |     |     |     |
|   mag_gain   |       0.081290 |        |        |        |
|     σ_acc    |            inf |             |             |             |
|    σ_gyro    |            inf |             |             |             |
|     σ_mag    |            inf |             |             |             |
|   σ_mag_err  |      0.1141336 |             |             |             |
|  Mean error  | <ul><li>0.63786 rad</li><li>36.54686 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>1.75871 rad</li><li>100.76688 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-2) - (running time of exp 4-2-X)

<br>

<details>
<summary><b><ins>Logs exp 3-2</ins></b></summary>

```
[START] 2026-03-26 01:10:02.746


[chosen value]
tau= 2.444098421677088 , K= 0.004115561714587462
mag_gain= 0.08129027557279966
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.1141336

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0006932543037092364 3.1395432880363785 0.6378630075132432 1.7587138609525377

[exp 3-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.039720545731819816 179.88258000310984 36.54685823803035 100.76688160373833

[END] 2026-03-26 02:02:03.748
```

</details>

<details>
<summary><b><ins>Logs exp 4-2</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |    exp 4-3-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   02:02:03.760 |    |    |    |
|   end time   |   03:33:28.681 |    |    |    |
| running time |   01:31:24.921 |    |    |    |
|    speedup   |              - | × (−)| × (-)| × (-)|
|      tau     |           2.61 |            |            |            |
|       K      |    0.003856468 |     |     |     |
|   mag_gain   |       0.090535 |        |        |        |
|     σ_acc    |      0.9858854 |             |             |             |
|    σ_gyro    |      0.5807959 |             |             |             |
|     σ_mag    |     54.0217667 |             |             |             |
|   σ_mag_err  |            inf |             |             |             |
|  Mean error  | <ul><li>0.62563 rad</li><li>35.84594 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>1.51352 rad</li><li>86.71842 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-3) - (running time of exp 4-3-X)

<br>

<details>
<summary><b><ins>Logs exp 3-3</ins></b></summary>

```
[START] 2026-03-26 02:02:03.760


[chosen value]
tau= 2.6083034535925176 , K= 0.0038564676502971457
mag_gain= 0.09053476509453695
acc_gate_sigma=0.9858854
gyro_gate_sigma=0.5807959
mag_gate_sigma=54.0217667
mag_err_sigma=inf

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0034484376418402497 3.1414837495374583 0.6256297470135747 1.5135219592318112

[exp 3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.1975809227914925 179.99376025742936 35.84594404171525 86.71842046435422

[END] 2026-03-26 03:33:28.681
```

</details>

<details>
<summary><b><ins>Logs exp 4-3</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |    exp 4-4-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   03:33:28.693 |    |    |    |
|   end time   |   04:35:02.755 |    |    |    |
| running time |   01:01:34.062 |    |    |    |
|    speedup   |              - | × (−)| × (-)| × (-)|
|      tau     |           2.46 |            |            |            |
|       K      |    0.004088848 |     |     |     |
|   mag_gain   |       0.103103 |        |        |        |
|     σ_acc    |      1.0180438 |             |             |             |
|    σ_gyro    |      0.6821457 |             |             |             |
|     σ_mag    |     64.1990739 |             |             |             |
|   σ_mag_err  |      0.1663950 |             |             |             |
|  Mean error  | <ul><li>0.66606 rad</li><li>38.16220 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>1.70196 rad</li><li>97.51531 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-4) - (running time of exp 4-4-X)

<br>

<details>
<summary><b><ins>Logs exp 3-4</ins></b></summary>

```
[START] 2026-03-26 03:33:28.693


[chosen value]
tau= 2.4600662863134968 , K= 0.004088848315551456
mag_gain= 0.1031033547069933
acc_gate_sigma=1.0180438
gyro_gate_sigma=0.6821457
mag_gate_sigma=64.1990739
mag_err_sigma=0.1663950

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0033279856458617494 3.139968905643992 0.6660560448646144 1.7019631438984009

[exp 3-4] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.19067953178799765 179.90696609571256 38.162200289918616 97.51530503219517

[END] 2026-03-26 04:35:02.755
```

</details>

<details>
<summary><b><ins>Logs exp 4-4</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |    exp 4-5-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   04:35:02.770 |    |    |    |
|   end time   |   06:35:06.344 |    |    |    |
| running time |   02:00:03.574 |    |    |    |
|    speedup   |              - | × (−)| × (-)| × (-)|
|      tau     |           2.23 |            |            |            |
|       K      |    0.004508710 |     |     |     |
|   mag_gain   |       0.128425 |        |        |        |
|     σ_acc    |   time-varying |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |   time-varying |
|       p      |             58 |              |              |              |
|     win_s    |       8.312611 |        |        |        |
| update_ratio |       0.224684 |        |        |        |
|   ema_alpha  |       0.113612 |        |        |        |
|   σ_mag_err  |            inf |             |             |             |
|  Mean error  | <ul><li>0.54571 rad</li><li>31.26704 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |
|   p90 error  | <ul><li>1.57967 rad</li><li>90.50838 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-5) - (running time of exp 4-5-X)

<br>

<details>
<summary><b><ins>Logs exp 3-5</ins></b></summary>

```
[START] 2026-03-26 04:35:02.770


[chosen value]
tau= 2.230979265968493 , K= 0.004508709715225082
mag_gain= 0.128424738669596
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 58 , best_win_s= 8.31261142176991
best_update_ratio= 0.2246844304357644 , best_ema_alpha= 0.11361224381200596
mag_err_sigma=inf

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.002293188814071561 3.1395883511652887 0.5457128822460509 1.579669146004168

[exp 3-5] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.13139004067291088 179.88516193020806 31.267044978618387 90.50837509307387

[END] 2026-03-26 06:35:06.344
```

</details>

<details>
<summary><b><ins>Logs exp 4-5</ins></b></summary>

```
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |    exp 4-6-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   06:35:06.360 |    |    |    |
|   end time   |   08:42:48.769 |    |    |    |
| running time |   02:07:42.409 |    |    |    |
|    speedup   |              - | × (−)| × (-)| × (-)|
|      tau     |           2.13 |            |            |            |
|       K      |    0.004730251 |     |     |     |
|   mag_gain   |       0.145923 |        |        |        |
|     σ_acc    |   time-varying |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |   time-varying |
|       p      |             54 |              |              |              |
|     win_s    |       9.118412 |        |        |        |
| update_ratio |       0.251041 |        |        |        |
|   ema_alpha  |       0.169355 |        |        |        |
|   σ_mag_err  |      1.7665532 |             |             |             |
|  Mean error  | <ul><li>0.54640 rad</li><li>31.30636 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul
|   p90 error  | <ul><li>1.60174 rad</li><li>91.77311 deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul> | <ul><li> rad</li><li> deg</li></ul

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: (running time of exp 3-6) - (running time of exp 4-6-X)

<br>

<details>
<summary><b><ins>Logs exp 3-6</ins></b></summary>

```
[START] 2026-03-26 06:35:06.360


[chosen value]
tau= 2.12649137036349 , K= 0.004730250981088377
mag_gain= 0.14592291528777135
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 54 , best_win_s= 9.118411564358787
best_update_ratio= 0.25104080513830873 , best_ema_alpha= 0.16935531512989532
mag_err_sigma=1.7665531582622025

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.002838688037336622 3.1399855733387 0.5463990043190333 1.6017429500652784

[exp 3-6] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.1626448438936635 179.90792108427354 31.30635687763105 91.77311090357422

[END] 2026-03-26 08:42:48.769
```

</details>

<details>
<summary><b><ins>Logs exp 4-6</ins></b></summary>

```
```

</details>

<br>

##### [best]

|              |       exp 3     | exp 4 (`seg_3`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-6      |                 |
|  total_best  |                 |                 |

<br>

<details>
<summary><b><ins>Logs exp 3</ins></b></summary>

```
```

</details>

<details>
<summary><b><ins>Logs exp 4</ins></b></summary>

```
[START] 2026-03-26 08:43:33.245

best: exp3-6

[END] 2026-03-26 08:43:33.399
```

</details>

<br>

#### [Secondary validation — Gravity & Linear Accel from Best Exp4]

<table>
  <thead>
    <tr>
      <th colspan="2">best exp 3</th>
      <th colspan="4">best exp 4</th>
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
      <td>3.06</td>
      <td>36.69</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
```

</details>

<br>

#### [Observation]

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-4-data-sum"></a>

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-4-conclusion"></a>

<br>
<br>
<br>
<br>










res: data 1-data3
- 실험 3과 실험 4의 결과 경향이 거의 동일 (어떤 계열이 좋은지, gating이 없는 것보다 fixed norm + mag innov 쪽이 좋은지, time-varying 계열은 상대적으로 불리한지)
- SEG 기반 튜닝을 써도 full-data 성능이 크게 무너지지 않음
- 경우에 따라서 오히려 더 좋아질 때도 있음을 관찰
- 시간 절감 효과 매우 큼
- data 1,2: exp 4가 더 좋거나 거의 동일
- data 3: best가 바뀌었지만 원래 exp3-3/3-4차이 미미했음
- 두 SEG 설정 중에서는 SEG2(5s window) 가 더 안정적으로 좋은 결과를 보임



res: 4(stress case), why exp4 doesn't follow tendency of exp3?

- mag 관련 스케일이 매우 이상함 (suggested mag_sigma 매우 큼)
- 전체 orientation error가 아예 매우 큼
- exp3에서 time-varying gate sigma는 adaptive하다는 장점보다 불안정한 adaptive behavior를 만들었을 가능성(second validation 큰 linear acc 오차)
- orientation 오차는 exp4가 exp3에 비해 크지만 exp4의 best로 추정한 중력/선형가속도의 오차는 exp3의 best로부터 추정한 것보다 줄었음
(Although exp4 yielded larger full-sequence orientation error than exp3 on Data 4, its selected parameter set produced clearly better secondary-validation results for gravity and linear-acceleration estimation. Therefore, for this dataset, exp4 appears to have identified a more practically useful solution.)

Therefore: redefined the segmentation policy for data4 and re-evaluated (add seg3)
-초반 transient / 초기 정렬 / bias settling을 더 많이 포함
- 중간 더 촘촘
- tail을 크게 잡아 후반 drift regime 반영
- 기존 seg1/seg2보다 long-horizon characteristic 더 보존

On Data 4, the full-sequence orientation metric favored the time-varying sigma variant selected by Exp. 3, but the secondary validation showed that the fixed-sigma variant selected by Exp. 4 produced substantially better gravity and linear-acceleration estimates. This suggests that optimizing the full-sequence orientation objective does not necessarily yield the most reliable solution for the downstream estimation target. Therefore, at least for Data 4, the Exp. 4 selection appears more practically trustworthy, although this interpretation should be re-examined after introducing the SEG3

- seg3 도입 후 거의 모든 실험에서(exp4-4제외: seg2로 돌린 결과보다 오차 소폭 상승) seg2로 돌린 결과보다 오차가 소폭 줄거나 비슷한 결과: 약간의 오차 개선에도 불구하고 러팅타임이 길어지므로(시간비용 증가) 추후 실험에서는 seg3 대신 seg2를 써도 품질이 아주 하락하진 않을것 같다고 판단됨



- conclusion:

FULL DATA는 의미 없다X
SEG가 더 우수하다X

Full-data tuning is conceptually the most direct optimization target, since it optimizes the objective on the entire sequence itself. However, for long recordings its computational cost became increasingly impractical. Across most datasets, segment-based tuning preserved comparable full-sequence accuracy while substantially reducing runtime, and in some cases even yielded slightly better results. Therefore, we did not observe sufficient practical advantage to justify continued reliance on full-data tuning for the remaining experiments.

