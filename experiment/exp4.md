
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

This experiment investigates whether a segment-based proxy objective can replace full-sequence tuning while preserving final full-sequence performance.<br>

<br>

In Experiment 3, each configuration was:<br>

- tuned on the full dataset
- evaluated on the full dataset

<br>

This provides a direct optimization target, but becomes computationally expensive for long recordings.<br>

<br>

To address this, Experiment 4 introduces:<br>

- tuning on representative segments
- final evaluation on the full sequence

<br>

Key hypothesis:<br>

A segment-based proxy objective is expected to preserve comparable full-sequence performance, while substantially reducing the tuning cost, with limited degradation in downstream metrics.<br>

<br>
<br>
<br>
<br>

### Why Segment-based Tuning? <a name="exp-4-why-seg"></a>

The cost of full-data tuning grows linearly with sequence length.<br>

<br>

Each Optuna trial requires:<br>

- running the full recursive estimation pipeline
- processing the entire sequence
- repeating this process across all trials

<br>

As a result, full-data tuning becomes increasingly impractical for long recordings.<br>

<br>

Therefore, Experiment 4 evaluates:<br>

1. How much runtime reduction is achievable using segment-based tuning?
2. How well the resulting parameters transfer to the full sequence?

<br>
<br>
<br>

### Segment-based Proxy Objective <a name="exp-4-why-seg-proxy"></a>

Instead of optimizing over the full sequence, tuning is performed on a set of representative segments.<br>

<br>

Under the assumption that local estimation quality correlates with global performance, the segment-based objective serves as a proxy for full-sequence optimization.<br>

<br>

- select multiple segments across the sequence
- evaluate candidate parameters only on those segments
- after optimization, run the selected parameters once on the full dataset

<br>

This reduces cost because each trial processes only a fraction of the data.<br>

<br>

Localized evaluation:<br>

Each segment is evaluated independently with a reset initial condition `q0 = q_ref[seg_start]`.<br>

This removes long-horizon error accumulation and isolates the intrinsic estimation behavior of the model.<br>

<br>

As a result, the proxy objective behaves more like a measure of local estimation quality than a full-trajectory optimization target.<br>

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

To maintain temporal representativeness, three segment policies are defined.<br>

<br>

For all datasets:<br>

- `seg_1`: head 5 s + 3 s window every 15 s + tail 5 s
- `seg_2`: head 10 s + 5 s window every 15 s + tail 10 s

<br>

For long datasets (>30 minutes):<br>

- `seg_3`: head 30 s + 10 s window every 20 s + tail 30 s

<br>

Each component captures different behavior:<br>

- head: initialization / transient response
- middle windows: recurring motion patterns
- tail: accumulated drift / late-stage behavior

<br>

`seg_3` increases temporal coverage for long recordings, preserving more long-horizon characteristics.<br>

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

Experiment 3 revealed an important limitation: lower orientation error does not necessarily imply better downstream estimation.<br>

<br>

In particular, on dataset 04, orientation error improved but gravity and linear acceleration estimation degraded.<br>

<br>

To address this, Experiment 4 defines a combined evaluation metric:<br>

```
total_score = ori_score + g_score + a_score

ori_score: orientation angle error score
g_score: gravity direction error score
a_score: linear-acceleration direction error score

All three components are computed using the same scoring function score_angle_err(...), so they can be combined on a comparable scale
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

All optimization targets are evaluated using the segment-based proxy objective defined above.<br>

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

Experiment 4 mirrors the structure of Experiment 3 as closely as possible.<br>

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

- identical search space
- identical number of Optuna trials
- identical random seed
- identical dataset
- identical evaluation (full sequence)

<br>

The only major difference is the tuning objective:<br>

- exp 3: full tuning → full evaluation
- exp 4: segment tuning → full evaluation

<br>

This isolates the effect of the proxy objective.<br>

<br>
<br>
<br>

### Evaluation and Measurements <a name="exp-4-method-eval"></a>

All final comparisons are performed on the full dataset, not on the tuning segments.<br>

<br>

For each experiment, the following are compared:<br>

- angle error statistics: min / max / mean / p90 (in rad and deg)
- total runtime
- speedup relative to full-data tuning
- selected best-parameter differences relative to Experiment 3
- best configuration under the total score

<br>

This evaluation protocol is critical because the objective is not to optimize segment performance itself, but to assess generalization to the full sequence.<br>

<br>
<br>
<br>

### Why Segment-based Tuning Sometimes Works Better <a name="exp-4-method-why-seg-better"></a>

Although the main purpose of Experiment 4 is runtime reduction, segment-based tuning can occasionally yield slightly better full-sequence results.<br>

<br>

1. **Reduced long-horizon bias**

Full-sequence optimization may overfit:<br>

- drift patterns
- specific disturbances
- late-stage artifacts

<br>

Segment-based tuning reduces this dominance.<br>

<br>

2. **Simpler optimization landscape**

With finite Optuna trials:<br>

- full objective: complex, noisy
- segment objective: simpler, cheaper

The proxy objective introduces a smoother and less noisy optimization landscape,
increasing the probability of finding a better solution under limited trials.<br>

<br>

3. **Emphasis on local fusion quality**

Segment reset (`q0 = q_ref[seg_start]`) removes:<br>

- accumulated error
- trajectory dependency

The selected parameters may better reflect local fusion robustness.<br>

<br>

4. **Regularization effect**

Segment-based tuning can behave like a weak regularizer.<br>

<br>

It avoids overfitting to:<br>

- specific time intervals
- repeated motion patterns

<br>

5. **Better temporal balance**

Full sequences may contain:<br>

- long easy regions
- dominant motion regimes

<br>

Segment selection distributes attention more evenly across time.<br>

<br>

These effects do not guarantee better performance, but explain why segment-based tuning can sometimes match or outperform full-data tuning.<br>

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
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
|      tau     |           4.28 |           4.31 |           4.32 |
|       K      |    0.002334764 |    0.002318593 |    0.002311993 |
|   mag_gain   |       3.519930 |       3.586378 |       3.388323 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |      1.6224342 |      1.7999687 |      0.7612136 |
|  Mean error  | <ul><li>0.04137 rad</li><li>2.37029 deg</li></ul> | <ul><li>0.04131 rad</li><li>2.36665 deg</li></ul> | <ul><li>0.04144 rad</li><li>2.37406 deg</li></ul> |
|   p90 error  | <ul><li>0.07968 rad</li><li>4.77430 deg</li></ul> | <ul><li>0.07946 rad</li><li>4.56528 deg</li></ul> | <ul><li>0.08010 rad</li><li>4.58945 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
|      tau     |           4.57 |           4.76 |           4.67 |
|       K      |    0.002186687 |    0.002102056 |    0.002142129 |
|   mag_gain   |       4.397713 |       3.846369 |       3.674369 |
|     σ_acc    |     20.4765482 |      9.9262628 |      5.7291508 |
|    σ_gyro    |      1.1233809 |      0.2901186 |      0.4576878 |
|     σ_mag    |     25.3407412 |     44.4812203 |     39.1623427 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.03815 rad</li><li>2.18586 deg</li></ul> | <ul><li>0.03703 rad</li><li>2.12167 deg</li></ul> | <ul><li>0.03609 rad</li><li>2.06771 deg</li></ul> |
|   p90 error  | <ul><li>0.07351 rad</li><li>4.21208 deg</li></ul> | <ul><li>0.05096 rad</li><li>2.92002 deg</li></ul> | <ul><li>0.05379 rad</li><li>3.08194 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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




[END] 2026-03-26 18:11:42.449
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
|      tau     |           4.76 |           5.02 |           5.01 |
|       K      |    0.002096190 |    0.001993253 |    0.001996605 |
|   mag_gain   |       5.678210 |       4.367921 |       4.148098 |
|     σ_acc    |     15.8903128 |      5.0794965 |      5.1120395 |
|    σ_gyro    |      0.9994370 |      0.3945917 |      0.3595636 |
|     σ_mag    |     18.0364040 |     27.5380465 |     37.0785850 |
|   σ_mag_err  |      1.5374059 |      0.8856901 |      0.8695816 |
|  Mean error  | <ul><li>0.03794 rad</li><li>2.17366 deg</li></ul> | <ul><li>0.03390 rad</li><li>1.94246 deg</li></ul> | <ul><li>0.03458 rad</li><li>1.98155 deg</li></ul> |
|   p90 error  | <ul><li>0.07607 rad</li><li>4.35842 deg</li></ul> | <ul><li>0.04893 rad</li><li>2.80326 deg</li></ul> | <ul><li>0.04782 rad</li><li>2.73973 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
|      tau     |           5.17 |           5.22 |           5.49 |
|       K      |    0.001932617 |    0.001914986 |    0.001820333 |
|   mag_gain   |       4.930732 |       5.207657 |       4.652301 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             76 |             80 |             80 |
|     win_s    |       5.777591 |       9.244633 |       9.148510 |
| update_ratio |       0.395468 |       0.411771 |       0.291020 |
|   ema_alpha  |       0.050896 |       0.047866 |       0.020566 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.05958 rad</li><li>3.41390 deg</li></ul> | <ul><li>0.05663 rad</li><li>3.24449 deg</li></ul> | <ul><li>0.06079 rad</li><li>3.48297 deg</li></ul> |
|   p90 error  | <ul><li>0.10931 rad</li><li>6.26327 deg</li></ul> | <ul><li>0.10513 rad</li><li>6.02340 deg</li></ul> | <ul><li>0.11554 rad</li><li>6.62021 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
|      tau     |           5.16 |           5.09 |           5.01 |
|       K      |    0.001937734 |    0.001962530 |    0.001996456 |
|   mag_gain   |       5.600152 |       6.025085 |       5.866344 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             80 |             78 |             78 |
|     win_s    |       5.046138 |       8.146240 |       7.546818 |
| update_ratio |       0.265996 |       0.418640 |       0.412702 |
|   ema_alpha  |       0.025141 |       0.066146 |       0.023297 |
|   σ_mag_err  |      1.9951554 |      1.3935356 |      1.4270680 |
|  Mean error  | <ul><li>0.05704 rad</li><li>3.26791 deg</li></ul> | <ul><li>0.05550 rad</li><li>3.17997 deg</li></ul> | <ul><li>0.05964 rad</li><li>3.41724 deg</li></ul> |
|   p90 error  | <ul><li>0.10546 rad</li><li>6.04262 deg</li></ul> | <ul><li>0.09866 rad</li><li>5.65289 deg</li></ul> | <ul><li>0.11592 rad</li><li>6.64166 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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

- Segment-based tuning preserves broadly comparable full-sequence orientation performance relative to full-data tuning on this dataset
- No large degradation is observed when replacing full-data tuning with either `seg_1` or `seg_2`
- In some configurations (notably exp 4-2), segment-based tuning yields slightly improved results
- The best-performing configuration remains consistent with Experiment 3 (fixed norm gating combined with magnetometer innovation gating)
- Runtime is reduced substantially, by approximately 3× to 5× depending on the segmentation policy
- `seg_1` and `seg_2` produce similar results on this dataset, without a strong consistent advantage for either policy

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
|  start time  |   21:23:46.312 |   23:26:14.461 |   23:27:57.487 |
|   end time   |   21:28:40.496 |   23:27:57.487 |   23:30:31.250 |
| running time |   00:04:54.184 |   00:01:43.026 |   00:02:33.763 |
|    speedup   |              - | 2.85× (−03m11s)| 1.92× (-02m20s)|
|      tau     |           3.97 |           3.97 |           3.97 |
|       K      |    0.002518346 |    0.002518346 |    0.002518346 |
|   mag_gain   |       3.724649 |       3.724649 |       3.724649 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.04564 rad</li><li>2.61517 deg</li></ul> | <ul><li>0.04564 rad</li><li>2.61517 deg</li></ul> | <ul><li>0.04564 rad</li><li>2.61517 deg</li></ul> |
|   p90 error  | <ul><li>0.08512 rad</li><li>4.87730 deg</li></ul> | <ul><li>0.08512 rad</li><li>4.87730 deg</li></ul> | <ul><li>0.08512 rad</li><li>4.87730 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-25 23:26:14.461

start :  2026-03-25 23:26:14.461

[chosen value]
tau= 3.9703762188793768 , K= 0.0025183455535897066
mag_gain= 3.724648901612812
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0014937441061020549 0.3703654566537466 0.04564329581096676 0.08512497175927315

[exp 4-1-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.08558523295218959 21.220377543695115 2.6151682130355454 4.877301612976674

end :  2026-03-25 23:27:57.487



start :  2026-03-25 23:27:57.487

[chosen value]
tau= 3.9703762188793768 , K= 0.0025183455535897066
mag_gain= 3.724648901612812
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-2] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0014937441061020549 0.3703654566537466 0.04564329581096676 0.08512497175927315

[exp 4-1-2] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.08558523295218959 21.220377543695115 2.6151682130355454 4.877301612976674

end :  2026-03-25 23:30:31.250




[END] 2026-03-25 23:30:31.250
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:28:40.507 |   23:30:31.272 |   23:32:11.832 |
|   end time   |   21:33:33.667 |   23:32:11.832 |   23:34:47.579 |
| running time |   00:04:53.160 |   00:01:40.560 |   00:02:35.747 |
|    speedup   |              - | 2.93× (−03m12s)| 1.89× (-02m17s)|
|      tau     |           4.37 |           4.36 |           4.37 |
|       K      |    0.002290369 |    0.002291161 |    0.002289921 |
|   mag_gain   |       4.124038 |       4.212683 |       4.822195 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |      0.7518797 |      0.7664148 |      0.5562251 |
|  Mean error  | <ul><li>0.04447 rad</li><li>2.54800 deg</li></ul> | <ul><li>0.04445 rad</li><li>2.54693 deg</li></ul> | <ul><li>0.04459 rad</li><li>2.55465 deg</li></ul> |
|   p90 error  | <ul><li>0.08333 rad</li><li>4.77430 deg</li></ul>  | <ul><li>0.08310 rad</li><li>4.76101 deg</li></ul> | <ul><li>0.08118 rad</li><li>4.65106 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-25 23:30:31.272

start :  2026-03-25 23:30:31.272

[chosen value]
tau= 4.364065984239298 , K= 0.0022911613465523895
mag_gain= 4.212683051899651
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.7664148

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0009483527717162655 0.39325831805278255 0.04445239765980317 0.08309527003459297

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.05433661130887564 22.53204188283783 2.546934775143939 4.761008270482075

end :  2026-03-25 23:32:11.832



start :  2026-03-25 23:32:11.832

[chosen value]
tau= 4.366430225571531 , K= 0.002289920777466318
mag_gain= 4.822194884752359
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.5562251

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0006487525585265768 0.40356033972427807 0.04458712576859642 0.08117630832563494

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.03717078355188678 23.122304245066832 2.5546541271595715 4.651059863511568

end :  2026-03-25 23:34:47.579




[END] 2026-03-25 23:34:47.579
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:33:33.679 |   23:34:47.605 |   23:37:37.185 |
|   end time   |   21:44:20.840 |   23:37:37.184 |   23:42:04.581 |
| running time |   00:10:47.161 |   00:02:49.597 |   00:04:27.396 |
|    speedup   |              - | 3.83× (−07m57s)| 2.42× (-06m19s)|
|      tau     |           4.54 |           4.79 |           4.58 |
|       K      |    0.002202910 |    0.002089426 |    0.002183965 |
|   mag_gain   |       3.892036 |       5.035091 |       4.942928 |
|     σ_acc    |      1.7562440 |      2.2571340 |      2.4943361 |
|    σ_gyro    |      6.2834979 |      3.0676142 |      3.0676142 |
|     σ_mag    |     16.2250324 |     27.3866293 |     14.5919279 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.03320 rad</li><li>1.90235 deg</li></ul> | <ul><li>0.03066 rad</li><li>1.75667 deg</li></ul> | <ul><li>0.03222 rad</li><li>1.84626 deg</li></ul> |
|   p90 error  | <ul><li>0.08865 rad</li><li>5.07946 deg</li></ul> | <ul><li>0.05954 rad</li><li>3.41168 deg</li></ul> | <ul><li>0.06229 rad</li><li>3.56894 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-25 23:34:47.605

start :  2026-03-25 23:34:47.605

[chosen value]
tau= 4.785418869267393 , K= 0.0020894261443040445
mag_gain= 5.035091163381099
acc_gate_sigma=2.2571340
gyro_gate_sigma=3.0676142
mag_gate_sigma=27.3866293
mag_err_sigma=inf

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0007829980204998273 0.4877178328970591 0.030659606009070903 0.05954497062396075

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.04486248194173802 27.944173418268228 1.7566660258537004 3.4116755079834187

end :  2026-03-25 23:37:37.184



start :  2026-03-25 23:37:37.185

[chosen value]
tau= 4.578268316681022 , K= 0.0021839653347669502
mag_gain= 4.9429280616754045
acc_gate_sigma=2.4943361
gyro_gate_sigma=3.0676142
mag_gate_sigma=14.5919279
mag_err_sigma=inf

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0006303447287448243 0.47056595202021106 0.032223270815208246 0.06228967586075258

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.03611609259539714 26.96144303331369 1.8462574198185122 3.5689355340590465

end :  2026-03-25 23:42:04.581




[END] 2026-03-25 23:42:04.581
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:44:20.855 |   23:42:04.641 |   23:44:04.313 |
|   end time   |   21:53:00.866 |   23:44:04.313 |   23:47:11.512 |
| running time |   00:08:40.011 |   00:01:59.672 |   00:03:07.199 |
|    speedup   |              - | 4.37× (−06m40s)| 2.78× (-05m32s)|
|      tau     |           4.19 |           4.75 |           4.87 |
|       K      |    0.002387545 |    0.002105088 |    0.002051649 |
|   mag_gain   |       4.519486 |       5.136019 |       5.296570 |
|     σ_acc    |      1.8465124 |      2.1881591 |      2.3563286 |
|    σ_gyro    |      6.8600581 |      3.7356068 |      2.1600169 |
|     σ_mag    |     13.1831565 |     15.5485159 |     16.9703900 |
|   σ_mag_err  |      0.8853287 |      0.6492924 |      0.6641613 |
|  Mean error  | <ul><li>0.03020 rad</li><li>1.73052 deg</li></ul> | <ul><li>0.03043 rad</li><li>1.74329 deg</li></ul> | <ul><li>0.03067 rad</li><li>1.75700 deg</li></ul> |
|   p90 error  | <ul><li>0.07193 rad</li><li>4.12132 deg</li></ul> | <ul><li>0.05908 rad</li><li>3.38502 deg</li></ul> | <ul><li>0.06043 rad</li><li>3.46246 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-25 23:42:04.639

start :  2026-03-25 23:42:04.641

[chosen value]
tau= 4.749815240065617 , K= 0.0021050880490153675
mag_gain= 5.136019364294149
acc_gate_sigma=2.1881591
gyro_gate_sigma=3.7356068
mag_gate_sigma=15.5485159
mag_err_sigma=0.6492924

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0006426356126626715 0.49417586405815517 0.030426073919845278 0.05907970401520231

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.036820308370375 28.314191347763003 1.7432856227601994 3.385017694953196

end :  2026-03-25 23:44:04.313



start :  2026-03-25 23:44:04.313

[chosen value]
tau= 4.8735337059080255 , K= 0.002051648742014854
mag_gain= 5.2965701856207055
acc_gate_sigma=2.3563286
gyro_gate_sigma=2.1600169
mag_gate_sigma=16.9703900
mag_err_sigma=0.6641613

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0004839198083786891 0.48434836052844527 0.030665371487720926 0.0604313091626525

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.027726562642878417 27.751116872360704 1.7569963634472194 3.462458965470249

end :  2026-03-25 23:47:11.512




[END] 2026-03-25 23:47:11.512
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   21:53:00.885 |   23:47:11.538 |   23:51:02.371 |
|   end time   |   22:08:27.826 |   23:51:02.371 |   23:56:59.565 |
| running time |   00:15:26.941 |   00:03:50.833 |   00:05:57.194 |
|    speedup   |              - | 4.03× (−11m36s)| 2.59× (-09m29s)|
|      tau     |           3.79 |           4.50 |           4.47 |
|       K      |    0.002635225 |    0.002221352 |    0.002236531 |
|   mag_gain   |       5.460352 |       6.853748 |       6.865375 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             80 |             77 |             78 |
|     win_s    |       9.996655 |       7.787453 |       7.568446 |
| update_ratio |       0.194599 |       0.194212 |       0.202047 |
|   ema_alpha  |       0.096770 |       0.070864 |       0.091838 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.04852 rad</li><li>2.78041 deg</li></ul> | <ul><li>0.05234 rad</li><li>2.99895 deg</li></ul> | <ul><li>0.05122 rad</li><li>2.93459 deg</li></ul> |
|   p90 error  | <ul><li>0.09915 rad</li><li>5.68096 deg</li></ul> | <ul><li>0.12603 rad</li><li>7.22119 deg</li></ul> | <ul><li>0.12067 rad</li><li>6.91405 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-25 23:47:11.538

start :  2026-03-25 23:47:11.538

[chosen value]
tau= 4.501213283313147 , K= 0.002221352037229732
mag_gain= 6.853747810114865
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 77 , best_win_s= 7.787453028390081
best_update_ratio= 0.1942118959369718 , best_ema_alpha= 0.07086352017932579
mag_err_sigma=inf

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0020755191540742478 0.4224690739306768 0.05234150145369438 0.12603355293533147

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.11891848782701724 24.205694911028132 2.998947126674551 7.221190660233141

end :  2026-03-25 23:51:02.371



start :  2026-03-25 23:51:02.371

[chosen value]
tau= 4.470665072708886 , K= 0.0022365306132929976
mag_gain= 6.8653753304065495
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 78 , best_win_s= 7.568446067805139
best_update_ratio= 0.20204743016125776 , best_ema_alpha= 0.09183794720005499
mag_err_sigma=inf

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.001309542238579109 0.4346812365116626 0.05121820828047676 0.120672933817597

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.07503124336469688 24.905400285646213 2.934587168693324 6.914049809209813

end :  2026-03-25 23:56:59.565




[END] 2026-03-25 23:56:59.565
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:08:27.842 |   23:56:59.590 |   00:00:45.127 |
|   end time   |   22:21:16.698 |   00:00:45.127 |   00:05:55.352 |
| running time |   00:12:48.856 |   00:03:45.537 |   00:05:10.225 |
|    speedup   |              - | 3.41× (−09m03s)| 2.48× (-07m38s)|
|      tau     |           3.56 |           4.85 |           4.60 |
|       K      |    0.002812494 |    0.002063009 |    0.002174497 |
|   mag_gain   |       6.998810 |       8.493434 |       7.930354 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             79 |             77 |             79 |
|     win_s    |       9.474137 |       7.953781 |       7.907742 |
| update_ratio |       0.339160 |       0.453191 |       0.467391 |
|   ema_alpha  |       0.185937 |       0.044901 |       0.021029 |
|   σ_mag_err  |      1.5525143 |      1.4390070 |      1.0623418 |
|  Mean error  | <ul><li>0.04861 rad</li><li>2.78499 deg</li></ul> | <ul><li>0.05836 rad</li><li>3.34387 deg</li></ul> | <ul><li>0.05968 rad</li><li>3.41938 deg</li></ul> |
|   p90 error  | <ul><li>0.10116 rad</li><li>5.79594 deg</li></ul> | <ul><li>0.13955 rad</li><li>7.99570 deg</li></ul> | <ul><li>0.12551 rad</li><li>7.19133 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-25 23:56:59.590

start :  2026-03-25 23:56:59.590

[chosen value]
tau= 4.846697030283956 , K= 0.0020630089387508894
mag_gain= 8.493434110694148
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 77 , best_win_s= 7.953780592074017
best_update_ratio= 0.4531913297699775 , best_ema_alpha= 0.04490080532161986
mag_err_sigma=1.4390070367454333

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0019716478092152868 0.4225885320184767 0.05836161658890756 0.13955132271642842

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.11296709815425088 24.21253935528777 3.343874316105095 7.995701817119479

end :  2026-03-26 00:00:45.127



start :  2026-03-26 00:00:45.127

[chosen value]
tau= 4.598202503539651 , K= 0.0021744973800514934
mag_gain= 7.930353584877646
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 79 , best_win_s= 7.907742277291757
best_update_ratio= 0.46739134824119727 , best_ema_alpha= 0.02102892166264869
mag_err_sigma=1.062341808838822

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0019488813287196227 0.50596743199187 0.05967945098339161 0.12551244462976174

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.11166267490748241 28.989798424206658 3.41938066500621 7.191333353654782

end :  2026-03-26 00:05:55.352




[END] 2026-03-26 00:05:55.352
```

</details>

<br>

##### [best]

|              |       exp 3     | exp 4 (`seg_2`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-4      |     exp4-4      |
|  total_best  |     exp3-4      |     exp4-4      |

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
[START] 2026-03-26 00:05:55.365

exp 3-1: total_score=3.2750722 | ori=0.0961009, g=2.8174082, a=0.3615630
exp 3-2: total_score=3.2646396 | ori=0.0938745, g=2.8176856, a=0.3530795
exp 3-3: total_score=3.2515703 | ori=0.0882643, g=2.8256769, a=0.3376291
exp 3-4: total_score=3.2187652 | ori=0.0805507, g=2.8252133, a=0.3130012
exp 3-5: total_score=3.2508102 | ori=0.1000124, g=2.8180449, a=0.3327530
exp 3-6: total_score=3.2542212 | ori=0.0969588, g=2.8182067, a=0.3390557

ori_best: exp3-4
total_best: exp3-4

[END] 2026-03-26 00:07:37.289




[START] 2026-03-26 00:07:37.305

[seg_2]
exp 4-1: total_score=0.5021090 | ori=0.0961009, g=0.0444451, a=0.3615630
exp 4-2: total_score=0.4894822 | ori=0.0954347, g=0.0410016, a=0.3530459
exp 4-3: total_score=0.4059686 | ori=0.0839723, g=0.0169935, a=0.3050027
exp 4-4: total_score=0.4047399 | ori=0.0832023, g=0.0159329, a=0.3056046
exp 4-5: total_score=0.5131377 | ori=0.1074809, g=0.0533255, a=0.3523312
exp 4-6: total_score=0.5559652 | ori=0.1463290, g=0.0505558, a=0.3590804

best: exp4-4

[END] 2026-03-26 00:07:38.722
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
      <td>0.62</td>
      <td>0.98</td>
      <td>5.94</td>
      <td>14.36</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 00:07:41.498

[Gravity]
RMSE norm: 0.11489979820267389

Gravity est/ref angle error in rad — min/max/mean/p90
3.9600707085944206e-05 0.03249040917804814 0.010766515660757465 0.017108123780704716

Gravity est/ref angle error in deg — min/max/mean/p90
0.002268953381758416 1.8615633205552724 0.6168759074229075 0.9802232880217777


[Linear accel]
RMSE norm: 0.4816489471602898

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00013295198153897038 2.918325479965369 0.10361118700035109 0.2505444935273097

Linear accel est/ref angle error in deg — min/max/mean/p90
0.007617587420084238 167.20773324750593 5.936483725460858 14.355142059357616
. . .
[END] 2026-03-26 00:07:42.351
```

</details>

<br>

#### [Observation]

- Segment-based tuning again preserves broadly comparable full-sequence orientation performance on this dataset
- Differences between full-data and segment-based tuning remain small across the evaluated configurations
- In some configurations (especially exp 4-3 and exp 4-4), segment-based tuning yields slightly improved results
- The best-performing configuration remains unchanged from Experiment 3
- `seg_1` and `seg_2` produce comparable results, with no clear consistent advantage for one policy over the other
- Runtime reduction remains substantial, ranging from approximately 2× to 4×
- No large degradation is observed when using segment-based tuning instead of full-data tuning on this dataset

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
|  start time  |   22:36:25.196 |   00:07:59.854 |   00:10:07.466 |
|   end time   |   22:44:31.294 |   00:10:07.466 |   00:13:28.151 |
| running time |   00:08:06.098 |   00:02:07.612 |   00:03:20.685 |
|    speedup   |              - | 3.83× (−05m58s)| 2.43× (-04m45s)|
|      tau     |           3.93 |           3.93 |           3.93 |
|       K      |    0.002542621 |    0.002542621 |    0.002542621 |
|   mag_gain   |       3.577178 |       3.577178 |       3.577178 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.04335 rad</li><li>2.48355 deg</li></ul> | <ul><li>0.04335 rad</li><li>2.48355 deg</li></ul> | <ul><li>0.04335 rad</li><li>2.48355 deg</li></ul> |
|   p90 error  | <ul><li>0.07790 rad</li><li>4.46346 deg</li></ul> | <ul><li>0.07790 rad</li><li>4.46346 deg</li></ul> | <ul><li>0.07790 rad</li><li>4.46346 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 00:07:59.853

start :  2026-03-26 00:07:59.854

[chosen value]
tau= 3.932469060126895 , K= 0.002542621224480923
mag_gain= 3.5771776662400905
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0009205019993881509 0.4253150125622266 0.04334613532077613 0.0779020739570321

[exp 4-1-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.05274087959829493 24.368755183369174 2.4835506120834188 4.463460053053944

end :  2026-03-26 00:10:07.466



start :  2026-03-26 00:10:07.466

[chosen value]
tau= 3.932469060126895 , K= 0.002542621224480923
mag_gain= 3.5771776662400905
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-2] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.0009205019993881509 0.4253150125622266 0.04334613532077613 0.0779020739570321

[exp 4-1-2] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.05274087959829493 24.368755183369174 2.4835506120834188 4.463460053053944

end :  2026-03-26 00:13:28.151




[END] 2026-03-26 00:13:28.151
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:44:31.305 |   00:13:28.173 |   00:15:35.175 |
|   end time   |   22:52:58.524 |   00:15:35.175 |   00:18:51.164 |
| running time |   00:08:27.219 |   00:02:07.002 |   00:03:15.989 |
|    speedup   |              - | 3.99× (−06m20s)| 2.60× (-05m11s)|
|      tau     |           4.32 |           4.31 |           4.32 |
|       K      |    0.002311993 |    0.002318593 |    0.002313149 |
|   mag_gain   |       3.388323 |       3.586378 |       3.346182 |
|     σ_acc    |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |
|   σ_mag_err  |      0.7612136 |      1.7999687 |      0.7541864 |
|  Mean error  | <ul><li>0.04219 rad</li><li>2.41742 deg</li></ul> | <ul><li>0.04219 rad</li><li>2.41726 deg</li></ul> | <ul><li>0.04223 rad</li><li>2.41947 deg</li></ul> |
|   p90 error  | <ul><li>0.07543 rad</li><li>4.32170 deg</li></ul> | <ul><li>0.07550 rad</li><li>4.32595 deg</li></ul> | <ul><li>0.07553 rad</li><li>4.32739 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 00:13:28.173

start :  2026-03-26 00:13:28.173

[chosen value]
tau= 4.312433686162575 , K= 0.00231859317140958
mag_gain= 3.5863784326810215
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=1.7999687

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0009831923408568659 0.40759089772750956 0.04218906552640245 0.0755020007615878

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.05633277158068627 23.353238207734673 2.417255396263737 4.325945988432508

end :  2026-03-26 00:15:35.175



start :  2026-03-26 00:15:35.175

[chosen value]
tau= 4.3225829930085435 , K= 0.0023131491779488033
mag_gain= 3.346182088288985
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.7541864

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0010814601350299493 0.40322640926317377 0.042227741156865915 0.07552717405048157

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.06196310144886421 23.1031714389947 2.4194713466593014 4.327388311642585

end :  2026-03-26 00:18:51.164




[END] 2026-03-26 00:18:51.164
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   22:52:58.541 |   00:18:51.185 |   00:22:24.619 |
|   end time   |   23:08:04.911 |   00:22:24.619 |   00:27:58.214 |
| running time |   00:15:06.370 |   00:03:33.434 |   00:05:33.595 |
|    speedup   |              - | 4.25× (−11m32s)| 2.72× (-09m32s)|
|      tau     |           3.96 |           4.24 |           4.19 |
|       K      |    0.002523787 |    0.002356902 |    0.002387709 |
|   mag_gain   |       3.241114 |       3.622718 |       3.670548 |
|     σ_acc    |      2.0484861 |      1.7491436 |      1.6511250 |
|    σ_gyro    |      1.3120595 |      4.7170521 |      4.6796846 |
|     σ_mag    |     34.8068405 |     34.0296423 |     36.5415523 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.03418 rad</li><li>1.95857 deg</li></ul> | <ul><li>0.03007 rad</li><li>1.72294 deg</li></ul> | <ul><li>0.03015 rad</li><li>1.72740 deg</li></ul> |
|   p90 error  | <ul><li>0.06214 rad</li><li>3.56029 deg</li></ul> | <ul><li>0.05715 rad</li><li>3.27473 deg</li></ul> | <ul><li>0.05905 rad</li><li>3.38354 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 00:18:51.185

start :  2026-03-26 00:18:51.185

[chosen value]
tau= 4.242340507404989 , K= 0.002356901639422946
mag_gain= 3.6227183013798925
acc_gate_sigma=1.7491436
gyro_gate_sigma=4.7170521
mag_gate_sigma=34.0296423
mag_err_sigma=inf

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0006698719658229318 0.2278980105825779 0.030071034770553794 0.05715487887672676

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.038380836455785716 13.057594165809485 1.7229433779438823 3.2747333382178625

end :  2026-03-26 00:22:24.619



start :  2026-03-26 00:22:24.619

[chosen value]
tau= 4.187603893313796 , K= 0.002387708950423391
mag_gain= 3.670548403421193
acc_gate_sigma=1.6511250
gyro_gate_sigma=4.6796846
mag_gate_sigma=36.5415523
mag_err_sigma=inf

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.000559214054110156 0.22939883026300692 0.030148761541005376 0.059053839045386664

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.03204060514491238 13.143584799308242 1.72739679384594 3.383535741345526

end :  2026-03-26 00:27:58.214




[END] 2026-03-26 00:27:58.214
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   23:08:04.927 |   00:27:58.234 |   00:30:30.401 |
|   end time   |   23:20:05.288 |   00:30:30.401 |   00:34:22.811 |
| running time |   00:12:00.361 |   00:02:32.167 |   00:03:52.410 |
|    speedup   |              - | 4.74× (−09m28s)| 3.10× (-08m07s)|
|      tau     |           4.09 |           3.99 |           4.34 |
|       K      |    0.002444338 |    0.002503571 |    0.002303090 |
|   mag_gain   |       3.375010 |       3.372414 |       3.798423 |
|     σ_acc    |      2.3851260 |      1.6679678 |      1.9949000 |
|    σ_gyro    |      1.1860780 |      4.2382947 |      4.1465728 |
|     σ_mag    |     40.2588639 |     34.4579239 |     42.8469248 |
|   σ_mag_err  |      0.7412543 |      0.5835949 |      0.8122204 |
|  Mean error  | <ul><li>0.03677 rad</li><li>2.10667 deg</li></ul> | <ul><li>0.03012 rad</li><li>1.72558 deg</li></ul> | <ul><li>0.03168 rad</li><li>1.81525 deg</li></ul> |
|   p90 error  | <ul><li>0.06585 rad</li><li>3.77265 deg</li></ul> | <ul><li>0.05814 rad</li><li>3.33093 deg</li></ul> | <ul><li>0.05796 rad</li><li>3.32113 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 00:27:58.233

start :  2026-03-26 00:27:58.234

[chosen value]
tau= 3.993807474932584 , K= 0.002503570680272206
mag_gain= 3.3724140644923666
acc_gate_sigma=1.6679678
gyro_gate_sigma=4.2382947
mag_gate_sigma=34.4579239
mag_err_sigma=0.5835949

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.00039225580399048193 0.22453033051147936 0.03011712773000501 0.05813566874955086

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.02247460205816549 12.864640310985221 1.7255843099857044 3.3309284585198564

end :  2026-03-26 00:30:30.401



start :  2026-03-26 00:30:30.401

[chosen value]
tau= 4.341462829256334 , K= 0.0023030899238646527
mag_gain= 3.7984231856644057
acc_gate_sigma=1.9949000
gyro_gate_sigma=4.1465728
mag_gate_sigma=42.8469248
mag_err_sigma=0.8122204

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0008392422616374184 0.2496515918223503 0.03168204228160993 0.05796462780806503

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.04808503958083807 14.303982560143409 1.8152473090912742 3.3211285344487744

end :  2026-03-26 00:34:22.811




[END] 2026-03-26 00:34:22.811
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   23:20:05.307 |   00:34:22.836 |   00:39:05.433 |
|   end time   |   23:41:01.545 |   00:39:05.433 |   00:46:38.969 |
| running time |   00:20:56.238 |   00:04:42.597 |   00:07:33.536 |
|    speedup   |              - | 4.45× (−16m13s)| 2.77× (-13m22s)|
|      tau     |           4.45 |           4.76 |           4.76 |
|       K      |    0.002246526 |    0.002099760 |    0.002099760 |
|   mag_gain   |       3.298374 |       4.260123 |       4.260123 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             80 |             80 |             80 |
|     win_s    |       8.373621 |       9.148510 |       9.148510 |
| update_ratio |       0.464217 |       0.291020 |       0.291020 |
|   ema_alpha  |       0.188064 |       0.020566 |       0.020566 |
|   σ_mag_err  |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.06298 rad</li><li>3.60866 deg</li></ul> | <ul><li>0.05938 rad</li><li>3.40226 deg</li></ul> | <ul><li>0.05938 rad</li><li>3.40226 deg</li></ul> |
|   p90 error  | <ul><li>0.11860 rad</li><li>6.79522 deg</li></ul> | <ul><li>0.10830 rad</li><li>6.20521 deg</li></ul> | <ul><li>0.10830 rad</li><li>6.20521 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 00:34:22.836

start :  2026-03-26 00:34:22.836

[chosen value]
tau= 4.761868013316307 , K= 0.002099759856621843
mag_gain= 4.260122553283831
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 9.148509985104303
best_update_ratio= 0.2910204473093046 , best_ema_alpha= 0.020566329226358745
mag_err_sigma=inf

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0010247195274085826 0.694384440143722 0.059380695086943304 0.10830132362419623

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.05871210410515207 39.78529777978981 3.4022632130350745 6.205208759346921

end :  2026-03-26 00:39:05.433



start :  2026-03-26 00:39:05.433

[chosen value]
tau= 4.761868013316307 , K= 0.002099759856621843
mag_gain= 4.260122553283831
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 80 , best_win_s= 9.148509985104303
best_update_ratio= 0.2910204473093046 , best_ema_alpha= 0.020566329226358745
mag_err_sigma=inf

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0010247195274085826 0.694384440143722 0.059380695086943304 0.10830132362419623

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.05871210410515207 39.78529777978981 3.4022632130350745 6.205208759346921

end :  2026-03-26 00:46:38.969




[END] 2026-03-26 00:46:38.969
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |
|:------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |
|  start time  |   23:41:01.559 |   00:46:38.991 |   00:51:30.521 |
|   end time   |   00:02:29.468 |   00:51:30.521 |   00:59:11.508 |
| running time |   00:21:27.909 |   00:04:51.530 |   00:07:40.987 |
|    speedup   |              - | 4.42× (−16m36s)| 2.80× (-13m46s)|
|      tau     |           4.17 |           4.90 |           4.90 |
|       K      |    0.002397647 |    0.002041520 |    0.002041520 |
|   mag_gain   |       4.227693 |       4.926100 |       4.926100 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |
|       p      |             79 |             79 |             79 |
|     win_s    |       9.474137 |       7.907742 |       7.907742 |
| update_ratio |       0.339160 |       0.417634 |       0.417634 |
|   ema_alpha  |       0.185937 |       0.021029 |       0.021029 |
|   σ_mag_err  |      1.5525143 |      1.0495151 |      1.0495151 |
|  Mean error  | <ul><li>0.06038 rad</li><li>3.45952 deg</li></ul> | <ul><li>0.05883 rad</li><li>3.37079 deg</li></ul> | <ul><li>0.05883 rad</li><li>3.37079 deg</li></ul> |
|   p90 error  | <ul><li>0.11016 rad</li><li>6.31160 deg</li></ul> | <ul><li>0.10713 rad</li><li>6.13797 deg</li></ul> | <ul><li>0.10713 rad</li><li>6.13797 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 00:46:38.991

start :  2026-03-26 00:46:38.991

[chosen value]
tau= 4.897712770750926 , K= 0.0020415201472421506
mag_gain= 4.9261000255182905
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 79 , best_win_s= 7.907742277291757
best_update_ratio= 0.41763357596324446 , best_ema_alpha= 0.02102892166264869
mag_err_sigma=1.0495150821624968

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0007190206257814828 0.695410863839615 0.0588314187226086 0.10712779821963235

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.04119684724013432 39.844107525556694 3.3707919955724055 6.137970706514028

end :  2026-03-26 00:51:30.521



start :  2026-03-26 00:51:30.521

[chosen value]
tau= 4.897712770750926 , K= 0.0020415201472421506
mag_gain= 4.9261000255182905
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 79 , best_win_s= 7.907742277291757
best_update_ratio= 0.41763357596324446 , best_ema_alpha= 0.02102892166264869
mag_err_sigma=1.0495150821624968

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0007190206257814828 0.695410863839615 0.0588314187226086 0.10712779821963235

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.04119684724013432 39.844107525556694 3.3707919955724055 6.137970706514028

end :  2026-03-26 00:59:11.508




[END] 2026-03-26 00:59:11.509
```

</details>

<br>

##### [best]

|              |       exp 3     | exp 4 (`seg_2`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-3      |     exp4-4      |
|  total_best  |     exp3-4      |     exp4-4      |

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
[START] 2026-03-26 00:59:11.526

exp 3-1: total_score=3.2229077 | ori=0.0757255, g=2.8149010, a=0.3322812
exp 3-2: total_score=3.2185424 | ori=0.0726234, g=2.8151221, a=0.3307969
exp 3-3: total_score=3.2124597 | ori=0.0598384, g=2.8197735, a=0.3328477
exp 3-4: total_score=3.2060719 | ori=0.0623238, g=2.8175517, a=0.3261964
exp 3-5: total_score=3.2986983 | ori=0.1551030, g=2.8146794, a=0.3289160
exp 3-6: total_score=3.2926167 | ori=0.1478990, g=2.8143185, a=0.3303992

ori_best: exp3-3
total_best: exp3-4

[END] 2026-03-26 01:01:58.808




[START] 2026-03-26 01:01:58.831

[seg_2]
exp 4-1: total_score=0.4413041 | ori=0.0757255, g=0.0332974, a=0.3322812
exp 4-2: total_score=0.4346313 | ori=0.0726918, g=0.0311373, a=0.3308023
exp 4-3: total_score=0.4552170 | ori=0.0584643, g=0.0318952, a=0.3648575
exp 4-4: total_score=0.4131489 | ori=0.0570370, g=0.0186799, a=0.3374320
exp 4-5: total_score=0.5048393 | ori=0.1547801, g=0.0232926, a=0.3267666
exp 4-6: total_score=0.5053582 | ori=0.1558785, g=0.0229354, a=0.3265443

best: exp4-4

[END] 2026-03-26 01:02:00.988
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
      <td>0.75</td>
      <td>1.12</td>
      <td>6.52</td>
      <td>15.67</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 01:02:02.961

[Gravity]
RMSE norm: 0.13782494216404365

Gravity est/ref angle error in rad — min/max/mean/p90
0.00016522475014165244 0.03697557574713851 0.01308594273707843 0.019478257054331257

Gravity est/ref angle error in deg — min/max/mean/p90
0.009466680854220236 2.118544435377322 0.7497692897844667 1.116021921484104


[Linear accel]
RMSE norm: 0.5384373625637601

Linear accel est/ref angle error in rad — min/max/mean/p90
0.00029370543201043216 2.937704087344635 0.1137618711265993 0.2735186271176008

Linear accel est/ref angle error in deg — min/max/mean/p90
0.016828081674264313 168.31804566317896 6.51807508506532 15.671462952051034
. . .
[END] 2026-03-26 01:02:04.270
```

</details>

<br>

#### [Observation]

- Segment-based tuning preserves broadly comparable full-sequence performance on this dataset
- In some configurations (notably exp 4-3 and exp 4-4), segment-based tuning yields slightly improved mean and p90 errors relative to full-data tuning
- The best-performing configuration differs slightly from Experiment 3, although the gap between the top candidates remains small
- This suggests that model selection can be sensitive when several configurations perform similarly
- Runtime reduction remains substantial, with speedups of approximately 2.5× to 4×
- `seg_1` and `seg_2` produce similar results on this dataset, without a strong consistent advantage for either policy

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
|  start time  |   00:08:48.169 |   08:19:18.349 |   08:34:37.755 |   08:55:36.754 |
|   end time   |   01:10:02.735 |   08:34:37.755 |   08:55:36.754 |   09:22:08.581 |
| running time |   01:01:14.566 |   00:15:19.406 |   00:20:58.999 |   00:26:31.827 |
|    speedup   |              - | 4.00× (−45m55s)| 2.92× (-40m15s)| 2.31× (-34m42s)|
|      tau     |           2.28 |           3.62 |           3.62 |           3.62 |
|       K      |    0.004408465 |    0.002775150 |    0.002775150 |    0.002775150 |
|   mag_gain   |       0.075637 |       0.051682 |       0.051682 |       0.051682 |
|     σ_acc    |            inf |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |            inf |
|   σ_mag_err  |            inf |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.77836 rad</li><li>44.59698 deg</li></ul> | <ul><li>0.82778 rad</li><li>47.42830 deg</li></ul> | <ul><li>0.82778 rad</li><li>47.42830 deg</li></ul> | <ul><li>0.82778 rad</li><li>47.42830 deg</li></ul> |
|   p90 error  | <ul><li>1.99740 rad</li><li>114.44248 deg</li></ul> | <ul><li>2.18361 rad</li><li>125.11142 deg</li></ul> | <ul><li>2.18361 rad</li><li>125.11142 deg</li></ul> | <ul><li>2.18361 rad</li><li>125.11142 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 08:19:18.348

start :  2026-03-26  

[chosen value]
tau= 3.624609913306148 , K= 0.002775150466264329
mag_gain= 0.05168160656620317
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-1] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.00043224609575303604 3.1415919070014238 0.8277799304377768 2.18360616310963

[exp 4-1-1] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.024765876997656623 179.9999572236374 47.42829637971748 125.11141726493703

end :  2026-03-26 08:34:37.755



start :  2026-03-26 08:34:37.755

[chosen value]
tau= 3.624609913306148 , K= 0.002775150466264329
mag_gain= 0.05168160656620317
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-2] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.00043224609575303604 3.1415919070014238 0.8277799304377768 2.18360616310963

[exp 4-1-2] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.024765876997656623 179.9999572236374 47.42829637971748 125.11141726493703

end :  2026-03-26 08:55:36.754



start :  2026-03-26 08:55:36.754

[chosen value]
tau= 3.624609913306148 , K= 0.002775150466264329
mag_gain= 0.05168160656620317
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=inf

[exp 4-1-3] Gyro+Acc+Mag angle error in rad — min/max/mean/p90
0.00043224609575303604 3.1415919070014238 0.8277799304377768 2.18360616310963

[exp 4-1-3] Gyro+Acc+Mag angle error in deg — min/max/mean/p90
0.024765876997656623 179.9999572236374 47.42829637971748 125.11141726493703

end :  2026-03-26 09:22:08.581




[END] 2026-03-26 09:22:08.581
```

</details>

<br>

##### [exp 4-2]

|              |     exp 3-2    |    exp 4-2-1   |    exp 4-2-2   |    exp 4-2-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   01:10:02.746 |   09:22:08.597 |   09:34:24.569 |   09:53:16.613 |
|   end time   |   02:02:03.748 |   09:34:24.569 |   09:53:16.613 |   10:20:32.398 |
| running time |   00:52:01.002 |   00:12:15.972 |   00:18:52.044 |   00:27:15.785 |
|    speedup   |              - | 4.25× (−39m45s)| 2.76× (-33m08s)| 1.91× (-24m45s)|
|      tau     |           2.44 |           3.97 |           3.97 |           3.97 |
|       K      |    0.004115562 |    0.002530634 |    0.002530634 |    0.002530634 |
|   mag_gain   |       0.081290 |       0.054196 |       0.054196 |       0.054196 |
|     σ_acc    |            inf |            inf |            inf |            inf |
|    σ_gyro    |            inf |            inf |            inf |            inf |
|     σ_mag    |            inf |            inf |            inf |            inf |
|   σ_mag_err  |      0.1141336 |      0.0467602 |      0.0467602 |      0.0467602 |
|  Mean error  | <ul><li>0.63786 rad</li><li>36.54686 deg</li></ul> | <ul><li>0.77167 rad</li><li>44.21363 deg</li></ul> | <ul><li>0.77167 rad</li><li>44.21363 deg</li></ul> | <ul><li>0.77167 rad</li><li>44.21363 deg</li></ul> |
|   p90 error  | <ul><li>1.75871 rad</li><li>100.76688 deg</li></ul> | <ul><li>1.96249 rad</li><li>112.44226 deg</li></ul> | <ul><li>1.96249 rad</li><li>112.44226 deg</li></ul> | <ul><li>1.96249 rad</li><li>112.44226 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 09:22:08.597

start :  2026-03-26 09:22:08.597

[chosen value]
tau= 3.9748284475595734 , K= 0.002530634472316332
mag_gain= 0.05419565268459507
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.0467602

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0011414317447457762 3.1414828275914535 0.771673501127199 1.9624876326374276

[exp 4-2-1] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.06539922157618686 179.99370743381434 44.21363477667228 112.44225869674496

end :  2026-03-26 09:34:24.569



start :  2026-03-26 09:34:24.569

[chosen value]
tau= 3.9748284475595734 , K= 0.002530634472316332
mag_gain= 0.05419565268459507
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.0467602

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0011414317447457762 3.1414828275914535 0.771673501127199 1.9624876326374276

[exp 4-2-2] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.06539922157618686 179.99370743381434 44.21363477667228 112.44225869674496

end :  2026-03-26 09:53:16.613



start :  2026-03-26 09:53:16.613

[chosen value]
tau= 3.9748284475595734 , K= 0.002530634472316332
mag_gain= 0.05419565268459507
acc_gate_sigma=inf
gyro_gate_sigma=inf
mag_gate_sigma=inf
mag_err_sigma=0.0467602

[exp 4-2-3] Gyro+Acc+Mag+Gating(Mag_innov) angle error in rad — min/max/mean/p90
0.0011414317447457762 3.1414828275914535 0.771673501127199 1.9624876326374276

[exp 4-2-3] Gyro+Acc+Mag+Gating(Mag_innov) angle error in deg — min/max/mean/p90
0.06539922157618686 179.99370743381434 44.21363477667228 112.44225869674496

end :  2026-03-26 10:20:32.398




[END] 2026-03-26 10:20:32.398
```

</details>

<br>

##### [exp 4-3]

|              |     exp 3-3    |    exp 4-3-1   |    exp 4-3-2   |    exp 4-3-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   02:02:03.760 |   10:20:32.415 |   10:40:48.389 |   11:12:48.359 |
|   end time   |   03:33:28.681 |   10:40:48.389 |   11:12:48.359 |   12:00:02.990 |
| running time |   01:31:24.921 |   00:20:15.974 |   00:31:59.970 |   00:47:14.631 |
|    speedup   |              - | 4.51× (−71m08s)| 2.86× (-59m24s)| 1.94× (-44m10s)|
|      tau     |           2.61 |           4.29 |           4.32 |           4.32 |
|       K      |    0.003856468 |    0.002344903 |    0.002327978 |    0.002327978 |
|   mag_gain   |       0.090535 |       0.038243 |       0.061560 |       0.061560 |
|     σ_acc    |      0.9858854 |      1.8111151 |      1.6591989 |      1.6591989 |
|    σ_gyro    |      0.5807959 |      0.7862228 |      0.2607787 |      0.2607787 |
|     σ_mag    |     54.0217667 |   1357.3048162 |      9.0199464 |      9.0199464 |
|   σ_mag_err  |            inf |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.62563 rad</li><li>35.84594 deg</li></ul> | <ul><li>0.73346 rad</li><li>42.02402 deg</li></ul> | <ul><li>0.63779 rad</li><li>36.54272 deg</li></ul> | <ul><li>0.63779 rad</li><li>36.54272 deg</li></ul> |
|   p90 error  | <ul><li>1.51352 rad</li><li>86.71842 deg</li></ul> | <ul><li>1.88300 rad</li><li>107.88798 deg</li></ul> | <ul><li>1.83274 rad</li><li>105.00823 deg</li></ul> | <ul><li>1.83274 rad</li><li>105.00823 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 10:20:32.415

start :  2026-03-26 10:20:32.415

[chosen value]
tau= 4.289660570600159 , K= 0.0023449029883337723
mag_gain= 0.03824255524468722
acc_gate_sigma=1.8111151
gyro_gate_sigma=0.7862228
mag_gate_sigma=1357.3048162
mag_err_sigma=inf

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.0005571061990019701 3.1413396517557333 0.7334574849815911 1.8830005072765463

[exp 4-3-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.031919833943388244 179.9855040626993 42.02401834172513 107.88798188793916

end :  2026-03-26 10:40:48.389



start :  2026-03-26 10:40:48.389

[chosen value]
tau= 4.32084775905666 , K= 0.002327977853386332
mag_gain= 0.061560352062797724
acc_gate_sigma=1.6591989
gyro_gate_sigma=0.2607787
mag_gate_sigma=9.0199464
mag_err_sigma=inf

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.002286355460023382 3.1388806839633165 0.6377908216240241 1.832739422488426

[exp 4-3-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.1309985183260316 179.84461558623522 36.5427222912377 105.00823385583068

end :  2026-03-26 11:12:48.359



start :  2026-03-26 11:12:48.359

[chosen value]
tau= 4.32084775905666 , K= 0.002327977853386332
mag_gain= 0.061560352062797724
acc_gate_sigma=1.6591989
gyro_gate_sigma=0.2607787
mag_gate_sigma=9.0199464
mag_err_sigma=inf

[exp 4-3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in rad — min/max/mean/p90
0.002286355460023382 3.1388806839633165 0.6377908216240241 1.832739422488426

[exp 4-3-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm) angle error in deg — min/max/mean/p90
0.1309985183260316 179.84461558623522 36.5427222912377 105.00823385583068

end :  2026-03-26 12:00:02.990




[END] 2026-03-26 12:00:02.990
```

</details>

<br>

##### [exp 4-4]

|              |     exp 3-4    |    exp 4-4-1   |    exp 4-4-2   |    exp 4-4-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   03:33:28.693 |   12:00:03.006 |   12:14:42.995 |   12:37:25.963 |
|   end time   |   04:35:02.755 |   12:14:42.995 |   12:37:25.963 |   13:10:25.812 |
| running time |   01:01:34.062 |   00:14:39.989 |   00:22:42.968 |   00:32:59.849 |
|    speedup   |              - | 4.20× (−46m54s)| 2.71× (-38m51s)| 1.87× (-28m34s)|
|      tau     |           2.46 |           3.94 |           4.06 |           4.60 |
|       K      |    0.004088848 |    0.002553681 |    0.002480581 |    0.002186936 |
|   mag_gain   |       0.103103 |       0.075086 |       0.061593 |       0.062231 |
|     σ_acc    |      1.0180438 |      1.7598608 |      2.0511352 |      1.9938993 |
|    σ_gyro    |      0.6821457 |      0.2933352 |      0.2442438 |      0.3132505 |
|     σ_mag    |     64.1990739 |      6.4253651 |      6.3724831 |     10.6196586 |
|   σ_mag_err  |      0.1663950 |      0.0687333 |      0.0699812 |      0.0557789 |
|  Mean error  | <ul><li>0.66606 rad</li><li>38.16220 deg</li></ul> | <ul><li>0.80620 rad</li><li>46.19160 deg</li></ul> | <ul><li>0.81961 rad</li><li>46.96016 deg</li></ul> | <ul><li>0.83928 rad</li><li>48.08735 deg</li></ul> |
|   p90 error  | <ul><li>1.70196 rad</li><li>97.51531 deg</li></ul> | <ul><li>1.98956 rad</li><li>113.99285 deg</li></ul> | <ul><li>1.99982 rad</li><li>114.58126 deg</li></ul> | <ul><li>2.02167 rad</li><li>115.83293 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 12:00:03.006

start :  2026-03-26 12:00:03.006

[chosen value]
tau= 3.93895707224597 , K= 0.0025536805063992173
mag_gain= 0.07508551153333974
acc_gate_sigma=1.7598608
gyro_gate_sigma=0.2933352
mag_gate_sigma=6.4253651
mag_err_sigma=0.0687333

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0025529911720431103 3.1403162744857256 0.8061955158623374 1.989550639376818

[exp 4-4-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.14627561929222768 179.92686886427825 46.191600521284144 113.99285476384613

end :  2026-03-26 12:14:42.995



start :  2026-03-26 12:14:42.995

[chosen value]
tau= 4.055033464733176 , K= 0.002480580734640064
mag_gain= 0.06159326824007658
acc_gate_sigma=2.0511352
gyro_gate_sigma=0.2442438
mag_gate_sigma=6.3724831
mag_err_sigma=0.0699812

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.0013532906250457238 3.1379089898671007 0.819609486213152 1.999820203835809

[exp 4-4-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.07753784126974116 179.78894161554427 46.960164408899445 114.58125746478386

end :  2026-03-26 12:37:25.963



start :  2026-03-26 12:37:25.963

[chosen value]
tau= 4.599511373139761 , K= 0.002186936192761582
mag_gain= 0.06223108284167501
acc_gate_sigma=1.9938993
gyro_gate_sigma=0.3132505
mag_gate_sigma=10.6196586
mag_err_sigma=0.0557789

[exp 4-4-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in rad — min/max/mean/p90
0.001700880885922848 3.1399218155193624 0.8392825304549183 2.0216660303504703

[exp 4-4-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—fixed_norm+Mag_innov) angle error in deg — min/max/mean/p90
0.09745329621785162 179.90426803031454 48.087346814126796 115.83293112404894

end :  2026-03-26 13:10:25.812




[END] 2026-03-26 13:10:25.812
```

</details>

<br>

##### [exp 4-5]

|              |     exp 3-5    |    exp 4-5-1   |    exp 4-5-2   |    exp 4-5-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   04:35:02.770 |   13:10:25.831 |   13:37:33.896 |   14:20:02.692 |
|   end time   |   06:35:06.344 |   13:37:33.896 |   14:20:02.692 |   15:22:27.178 |
| running time |   02:00:03.574 |   00:27:08.065 |   00:42:28.796 |   01:02:24.486 |
|    speedup   |              - | 4.42× (−92m55s)| 2.83× (-77m34s)| 1.92× (-57m39s)|
|      tau     |           2.23 |           4.52 |           4.23 |           4.62 |
|       K      |    0.004508710 |    0.002225248 |    0.002378750 |    0.002175047 |
|   mag_gain   |       0.128425 |       0.056995 |       0.049781 |       0.048686 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |   time-varying |
|       p      |             58 |             66 |             68 |             75 |
|     win_s    |       8.312611 |       9.976591 |       8.169525 |       9.773986 |
| update_ratio |       0.224684 |       0.200027 |       0.130646 |       0.308748 |
|   ema_alpha  |       0.113612 |       0.120144 |       0.094614 |       0.071261 |
|   σ_mag_err  |            inf |            inf |            inf |            inf |
|  Mean error  | <ul><li>0.54571 rad</li><li>31.26704 deg</li></ul> | <ul><li>0.61175 rad</li><li>35.05096 deg</li></ul> | <ul><li>0.61923 rad</li><li>35.47905 deg</li></ul> | <ul><li>0.61268 rad</li><li>35.10404 deg</li></ul> |
|   p90 error  | <ul><li>1.57967 rad</li><li>90.50838 deg</li></ul> | <ul><li>1.84428 rad</li><li>105.66965 deg</li></ul> | <ul><li>1.86994 rad</li><li>107.13981 deg</li></ul> | <ul><li>1.84001 rad</li><li>105.42508 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 13:10:25.831

start :  2026-03-26 13:10:25.831

[chosen value]
tau= 4.520322285312156 , K= 0.0022252479482761573
mag_gain= 0.056995105315943505
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 66 , best_win_s= 9.976591110804682
best_update_ratio= 0.20002660533489514 , best_ema_alpha= 0.12014417974580677
mag_err_sigma=inf

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.00410455660052187 3.140341317876471 0.6117545823544913 1.844283285050746

[exp 4-5-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.23517376998246778 179.92830374487266 35.050955666700695 105.6696484599307

end :  2026-03-26 13:37:33.896



start :  2026-03-26 13:37:33.896

[chosen value]
tau= 4.228623311317688 , K= 0.002378750044728722
mag_gain= 0.049781156730359676
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 68 , best_win_s= 8.169524618926353
best_update_ratio= 0.13064555708444062 , best_ema_alpha= 0.09461405699687192
mag_err_sigma=inf

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.00443980017951733 3.1415041015771505 0.6192262005346935 1.8699425000198193

[exp 4-5-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.25438181216776823 179.9949263434082 35.4790478545595 107.13981318327751

end :  2026-03-26 14:20:02.692



start :  2026-03-26 14:20:02.692

[chosen value]
tau= 4.624653577798774 , K= 0.002175046783877299
mag_gain= 0.048685968424018504
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 75 , best_win_s= 9.773986162007244
best_update_ratio= 0.30874845915243115 , best_ema_alpha= 0.07126063738348828
mag_err_sigma=inf

[exp 4-5-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in rad — min/max/mean/p90
0.0027042381090563623 3.1412575347174987 0.6126810570407563 1.8400147240573352

[exp 4-5-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm)  angle error in deg — min/max/mean/p90
0.154941430447368 179.98079910298233 35.104038756049384 105.42507793041409

end :  2026-03-26 15:22:27.178




[END] 2026-03-26 15:22:27.178
```

</details>

<br>

##### [exp 4-6]

|              |     exp 3-6    |    exp 4-6-1   |    exp 4-6-2   |    exp 4-6-3   |
|:------------:|---------------:|---------------:|---------------:|---------------:|
|  tuning data |           full |          seg_1 |          seg_2 |          seg_3 |
|  start time  |   06:35:06.360 |   15:22:27.200 |   15:50:32.827 |   16:34:31.119 |
|   end time   |   08:42:48.769 |   15:50:32.827 |   16:34:31.119 |   17:45:01.124 |
| running time |   02:07:42.409 |   00:28:05.547 |   00:43:58.292 |   01:10:30.005 |
|    speedup   |              - | 4.55× (−99m36s)| 2.90× (-83m44s)| 1.81× (-57m12s)|
|      tau     |           2.13 |           4.40 |           4.37 |           4.82 |
|       K      |    0.004730251 |    0.002285143 |    0.002302736 |    0.002085775 |
|   mag_gain   |       0.145923 |       0.036417 |       0.040780 |       0.041199 |
|     σ_acc    |   time-varying |   time-varying |   time-varying |   time-varying |
|    σ_gyro    |   time-varying |   time-varying |   time-varying |   time-varying |
|     σ_mag    |   time-varying |   time-varying |   time-varying |   time-varying |
|       p      |             54 |             71 |             68 |             74 |
|     win_s    |       9.118412 |       8.180015 |       7.958232 |       9.724363 |
| update_ratio |       0.251041 |       0.424806 |       0.231253 |       0.192439 |
|   ema_alpha  |       0.169355 |       0.053551 |       0.034836 |       0.058035 |
|   σ_mag_err  |      1.7665532 |      0.1166076 |      0.1323406 |      0.1521823 |
|  Mean error  | <ul><li>0.54640 rad</li><li>31.30636 deg</li></ul> | <ul><li>0.70361 rad</li><li>40.31377 deg</li></ul> | <ul><li>0.68749 rad</li><li>39.39036 deg</li></ul> | <ul><li>0.67410 rad</li><li>38.62323 deg</li></ul> |
|   p90 error  | <ul><li>1.60174 rad</li><li>91.77311 deg</li></ul> | <ul><li>1.95238 rad</li><li>111.86303 deg</li></ul> | <ul><li>1.95519 rad</li><li>112.02397 deg</li></ul> | <ul><li>1.94582 rad</li><li>111.48701 deg</li></ul> |

<br>

** `σ = inf` means gating not applied<br>
** time format: hh:mm:ss.ms<br>
** `speedup`: multiplicative speedup relative to full-data tuning. the value in parentheses denotes absolute time reduction<br>

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
[START] 2026-03-26 15:22:27.199

start :  2026-03-26 15:22:27.200

[chosen value]
tau= 4.40184246363846 , K= 0.002285142636073229
mag_gain= 0.03641717202079333
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 71 , best_win_s= 8.180014660662678
best_update_ratio= 0.4248060499474774 , best_ema_alpha= 0.05355105461742209
mag_err_sigma=0.1166076430796149

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.004493125176493342 3.1415830034983414 0.7036080474694799 1.9523782089558916

[exp 4-6-1] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.2574371094370416 179.9994470904879 40.31377155144168 111.86303138648333

end :  2026-03-26 15:50:32.827



start :  2026-03-26 15:50:32.827

[chosen value]
tau= 4.368210886576224 , K= 0.002302736326638735
mag_gain= 0.04078049156753014
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 68 , best_win_s= 7.958232083026185
best_update_ratio= 0.23125252858661066 , best_ema_alpha= 0.034836383042531234
mag_err_sigma=0.13234060542793807

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.004462051286859954 3.1384129875446414 0.687491506714047 1.9551871273558858

[exp 4-6-2] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.2556567067079932 179.81781855535175 39.39036178580479 112.02397055579964

end :  2026-03-26 16:34:31.119



start :  2026-03-26 16:34:31.119

[chosen value]
tau= 4.822589880401419 , K= 0.00208577510018343
mag_gain= 0.0411993461458936
acc/gyro/mag_gate_sigma = time-varying gate sigma
best_p= 74 , best_win_s= 9.724363195965493
best_update_ratio= 0.19243868119080831 , best_ema_alpha= 0.05803519228943914
mag_err_sigma=0.15218234845223808

[exp 4-6-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in rad — min/max/mean/p90
0.0028630505105614696 3.1384805142608396 0.6741025299840645 1.945815344755684

[exp 4-6-3] Gyro+Acc+Mag+Gating(Gyro/Acc/Mag—time-varying_norm+Mag_innov)  angle error in deg — min/max/mean/p90
0.16404071078794774 179.8216875511943 38.623229927177924 111.48700696629395

end :  2026-03-26 17:45:01.124




[END] 2026-03-26 17:45:01.124
```

</details>

<br>

##### [best]

|              |       exp 3     | exp 4 (`seg_3`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-6      |     exp4-5      |
|  total_best  |     exp3-2      |     exp4-3      |

<br>

<details>
<summary><b><ins>Logs exp 3</ins></b></summary>

```
[START] 2026-03-26 08:43:33.245

best: exp3-6

[END] 2026-03-26 08:43:33.399
```

</details>

<details>
<summary><b><ins>Logs exp 4</ins></b></summary>

```
[START] 2026-03-26 17:45:01.137

exp 3-1: total_score=5.3734524 | ori=1.5663607, g=2.8164360, a=0.9906556
exp 3-2: total_score=5.2296440 | ori=1.4402342, g=2.8166692, a=0.9727405
exp 3-3: total_score=5.5445234 | ori=1.4084937, g=2.8092348, a=1.3267949
exp 3-4: total_score=5.4069555 | ori=1.3442309, g=2.8104827, a=1.2522419
exp 3-5: total_score=5.5345874 | ori=1.3089293, g=2.8040225, a=1.4216357
exp 3-6: total_score=5.5245646 | ori=1.3004890, g=2.8044867, a=1.4195890

ori_best: exp3-6
total_best: exp3-2

[END] 2026-03-26 18:01:28.418




[START] 2026-03-26 18:01:28.438

[seg_3]
exp 4-1: total_score=2.6393331 | ori=1.7325959, g=0.0518524, a=0.8548848
exp 4-2: total_score=2.4715143 | ori=1.5972535, g=0.0503878, a=0.8238731
exp 4-3: total_score=2.2134157 | ori=1.4460966, g=0.0537378, a=0.7135813
exp 4-4: total_score=2.2618031 | ori=1.6470711, g=0.0441323, a=0.5705996
exp 4-5: total_score=3.0178949 | ori=1.4447698, g=0.1011854, a=1.4719397
exp 4-6: total_score=3.0924498 | ori=1.5322361, g=0.0997948, a=1.4604189

best: exp4-3

[END] 2026-03-26 18:01:39.775
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
      <td>1.14</td>
      <td>2.48</td>
      <td>15.40</td>
      <td>34.82</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><b><ins>Logs</ins></b></summary>

```
[START] 2026-03-26 18:01:53.974

[Gravity]
RMSE norm: 0.29415665754269

Gravity est/ref angle error in rad — min/max/mean/p90
2.650200319338574e-05 0.16928381341158324 0.01985237786506237 0.0433162564471935

Gravity est/ref angle error in deg — min/max/mean/p90
0.001518452931623233 9.699248048363842 1.1374574649670095 2.4818386787305293


[Linear accel]
RMSE norm: 0.6285299177917264

Linear accel est/ref angle error in rad — min/max/mean/p90
5.19675043157872e-05 3.114404714478231 0.268836462740289 0.6077097984126015

Linear accel est/ref angle error in deg — min/max/mean/p90
0.0029775186691224976 178.44224583524883 15.403194694244569 34.819206617788126
. . .
[END] 2026-03-26 18:02:00.474
```

</details>

<br>

#### [Observation]

- Unlike Datasets 01–03, Dataset 04 does not preserve the same overall tendency under segment-based tuning as under full-data tuning
- All Experiment 4 configurations remain worse than their Experiment 3 counterparts in terms of full-sequence orientation error, despite substantial runtime reduction
- This suggests that, on this long and highly variable sequence, the segment-based proxy objective does not reproduce the same parameter preference as the full-data objective
- Dataset 04 already appears atypical in Experiment 3, with an unusually large suggested magnetometer scale and persistently large orientation error across configurations
- This indicates that the sequence is substantially more difficult and likely more non-stationary than the other datasets
- Under the revised total criterion, the best Experiment 4 result on `seg_3` is a fixed norm-gating configuration (exp 4-3)
- Although its orientation error is worse than the best Experiment 3 orientation result, its gravity and linear-acceleration estimates are much better in secondary validation
- This suggests that the Experiment 4 selection may be more useful for downstream estimation on this dataset
- Introducing `seg_3` preserves more long-horizon structure than `seg_1` and `seg_2`, and often yields similar or slightly improved results relative to `seg_2`. However, the improvement remains modest compared with the increased runtime
- As a result, `seg_2` may still be a reasonable default trade-off between computational cost and performance for later experiments

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-4-data-sum"></a>

For datasets 01–03:<br>

- Segment-based tuning preserves the main performance tendencies observed in Experiment 3
-The relative ranking among the strongest configurations remains broadly similar
- Full-sequence orientation accuracy is maintained with only minor variations
- In some cases, segment-based tuning yields slightly improved results
- Runtime reduction is consistently substantial, typically between 2× and 5×

<br>

For dataset 04:<br>

- Segment-based tuning does not reproduce the same behavior as full-data tuning
- Full-sequence orientation error increases relative to full-data tuning across the evaluated Experiment 4 configurations
- However, under the revised selection criterion, the selected Experiment 4 result provides better gravity and linear-acceleration estimates in secondary validation
- This suggests that model selection on long and highly non-stationary data depends strongly on which downstream target is emphasized

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-4-conclusion"></a>

Experiment 4 evaluates segment-based tuning as a proxy for full-sequence optimization.<br>

<br>

The results indicate that:<br>

1. For datasets with relatively stable characteristics (datasets 01–03), segment-based tuning preserves broadly comparable full-sequence performance while substantially reducing computational cost
2. The relative ranking among the strongest configurations remains broadly consistent with full-data tuning, suggesting that the proxy objective captures much of the main performance structure
3. On the evaluated stable datasets, segment-based tuning does not introduce clear systematic degradation and, in some cases, yields slightly improved results
4. On long and highly non-stationary data (dataset 04), segment-based tuning behaves differently and does not reproduce the same optimal configuration as full-data tuning
5. However, secondary validation suggests that the segment-based selection may produce more reliable gravity and linear-acceleration estimates in such cases

<br>

Overall:<br>

- Full-data tuning remains the most direct optimization approach, but its computational cost becomes prohibitive for long sequences
- Segment-based tuning provides a practical alternative, offering substantial runtime reduction while maintaining comparable performance on most evaluated datasets
- The discrepancy observed on dataset 04 indicates that the choice of optimization objective remains critical, particularly for long and non-stationary sequences
- These results suggest that segment-based tuning is a useful approximation for model selection in many cases, although its behavior should be examined further under more challenging conditions

<br>

Next steps:<br>
- Experiment 5: improve gyro integration robustness on long uncontrolled sequences

<br>
<br>
<br>
<br>
