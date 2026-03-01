
 * [Experiment 2](#exp-2) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Goal](#exp-2-goal) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Method](#exp-2-method) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Accelerometer Correction](#exp-2-method-acc-correction) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Detect Quasi-static](#exp-2-method-quasi-static) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Suggest Tau and K](#exp-2-method-tau-k) <br>
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

I compared four runs (same dataset, same trimmed start):
- [exp 1-2] Gyro-only (baseline, after stabilization trimming)
- [exp 2-1] Gyro + Acc, wighout gating
- [exp 2-2] Gyro + Acc with Acc gating
- [exp 2-3] Gyro + Acc with Gyro/Acc gating

<br>

Key hypothesis:<br>
The correction term counteracts accumulated gyro bias<br>
so the gap `|err(gyro-only) − err(gyro+acc)|` tends to increase over time.<br>

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

<br>
<br>
<br>

### Detect Quasi-static <a name="exp-2-method-quasi-static"></a>

A quasi-static segment is automatically detected to evaluate gravity stability without being polluted by motion.<br>

- Compute `||w||` and the accelerometer magnitude residual `| ||a|| - g0 |`
- `quasi_static = (|w| < w_thr) & (| ||a|| - g0 | < a_thr)`, if threasholds are not provided, they are set to a low percentile
- Apply a majority-vote smoothing window to debounce short spikes
- Keep the longest continuous True run, and reject if it is shorter than `min_duration_s`

<br>

##### [Implementation]

```py
#autotune.py

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
                                 tau_candidates: tuple[float, ...] = (0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1),
                                 runner_kwargs: dict[str, Any] = None,
                                 ) -> tuple[list[dict[str, Any]], float, float]:
        . . .
        for tau in tau_candidates:
                K = float(dt_medean / tau)
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
        tau_table.sort(key=lambda d: d["quasi_static_score_mean_angle(rad)"])
        . . .
        return tau_table, best_tau, best_K
```

<br>
<br>
<br>

### Suggest Gate Sigma <a name="exp-2-method-sigma"></a>

Gate sigmas are suggested from robust percentiles of the data.<br>
If quasi-static exists, statistics are computed on that segment, otherwise fallback to the full sequence.<br>

<br>

```
For gyro gating
	gyro_sigma = max(sigma_floor, percentile(||w||, p_gyro)) (or inf if disabled)

For accel gating:
	acc_sigma = max(sigma_floor, percentile(| ||a|| - g0 |, p_acc)) (or inf if disabled)
```
<br>

These sigma values set the scale for how aggressively measurement can be trusted/ignored when motion or acceleration residuals are high.<br>

<br>
<br>
<br>
<br>

### Result <a name="exp-2-res"></a>

Each plot compares:<br>

- blue: exp 1-2
- orange: exp 2-1
- green: exp 2-2
- red: exp 2-3

<br>
<br>

### Dataset 01 — 5 min <a name="exp-2-res-data-01"></a>

<br>
<br>
<br>

### Dataset 02 — 9 min <a name="exp-2-res-data-02"></a>

<br>
<br>
<br>

### Dataset 03 — 13 min <a name="exp-2-res-data-03"></a>

<br>
<br>
<br>

### Dataset 04 — 96 min <a name="exp-2-res-data-04"></a>

<br>
<br>
<br>
<br>

### Cross-dataset Summary <a name="exp-2-data-sum"></a>

<br>
<br>
<br>
<br>

### Conclusion <a name="exp-2-conclusion"></a>

<br>
<br>
<br>
<br>
