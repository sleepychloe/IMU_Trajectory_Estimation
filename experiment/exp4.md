
 * [Experiment 4](#exp-4) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-4-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why Segment-based Tuning](#exp-4-why-seg) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Segment-based Proxy Objective](#exp-4-why-seg-proxy) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Segment Policies](#exp-4-why-seg-policy) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Method](#exp-4-method) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Redefine the Best-Model Criterion](#exp-4-method-best) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Experiment Structure](#exp-4-method-structure) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Evaluation and Measurements](#exp-4-method-eval) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Why Segment-based Tuning Sometimes Works Better](#exp-4-method-why-seg-better)
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



<br>
<br>
<br>

### Dataset 02 — 9 min <a name="exp-4-res-data-02"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data02_exp4.png" width="952" height="311">



<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-4-res-data-03"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data03_exp4.png" width="952" height="311">



<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-4-res-data-04"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data04_exp4.png" width="952" height="311">



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

