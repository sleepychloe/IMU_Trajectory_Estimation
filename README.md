Currently in progress


## Lists

 * [Project IMU Orientation Estimation](#project) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Introduction](#project-intro) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why a reference is needed — Sensor Logger orientation as REF)](#project-ref) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Why trimming the initial seconds matters — Fair Comparison](#project-fair-comparison) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Experimental roadmap (progressive complexity)](#project-exp) <br>

 * [Experiment Result Shortcut](#exp) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Experiment 1 (Dataset 03) — Gyro-only propagation](#exp-1) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Experiment 2 (Dataset 03) — Gyro + Accelerometer](#exp-2) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Experiment 3 (Dataset 03) — Gyro + Accelerometer + Magnetometer](#exp-3) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Experiment 4 (Dataset 03) — Segment-based Proxy Objective](#exp-4) <br>

 * [Understanding Coordinate Systems and Sensors](#orientation) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Coordinate Frame](#orientation-coordinate) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [World Frame (Inertial Frame)](#orientation-coordinate-world) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Body Frame (Sensor Frame)](#orientation-coordinate-body) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Sensor Model](#orientation-sensor) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Gyroscope](#orientation-sensor-gyro) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Accelerometer](#orientation-sensor-acc) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Magnetometer](#orientation-sensor-mag) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Gravity vs Magnetic Field](#orientation-grav-mag) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Initial Magnetic Reference (Yaw Anchor)](#orientation-grav-mag-init-mag-ref) <br>

 * [Implementation – IMU Orientation Estimation](#implementation) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 1] Gyroscope Propagation](#implementation-gyro) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 2] Accelerometer Correction (Roll/Pitch)](#implementation-acc) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 3] Linear Acceleration Estimation](#implementation-acc-linear) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [[Step 4] Magnetometer Correction (Yaw)](#implementation-mag) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Gating Observation After Implementation](#implementation-gating-observation) <br>

 * [Quaternion](#quaternion) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Quaternion](#quaternion-quaternion) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Axis-Angle to Quaternion](#quaternion-axis-angle-to-quaternion) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Euler Angle vs Quaternion](#quaternion-euler-angle-vs-quaternion) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Applications of Quaternion](#quaternion-applications-of-quaternion) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [IMU Orientation Update](#quaternion-applications-imu) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Rotating a Vector with Quaternion](#quaternion-applications-vector) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Rotation](#quaternion-rotation) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [2D Rotation](#quaternion-rotation-2d) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [3D Rotation](#quaternion-rotation-3d) <br>


<br>
<br>

## Project IMU Orientation Estimation <a name="project"></a>

### Introduction <a name="project-intro"></a>

This project focuses on building a custom orientation estimation pipeline from IMU data (gyroscope, accelerometer, magnetometer).<br>

<br>

The goal is not to use a black-box sensor fusion library, but to implement orientation estimation from first principles and understand the full chain of reasoning:<br>
how each sensor behaves, how errors accumulate, and how different correction strategies affect stability and drift.<br>

<br>
<br>

The project emphasizes:<br>
- Low-level quaternion-based orientation modeling (frame conventions, quaternion dynamics, vector rotations)
- Drift analysis of gyro-only integration and why correction is necessary
- Real-world evaluation methodology using recorded logs and quantitative error metrics
- Structured debugging through controlled experiments (ablation-style comparison of gating/correction options)

<br>
<br>

The long-term goal is to build a robust and fully controlled orientation estimation framework that remains stable in real-world dynamic environments (indoor/outdoor transitions, transportation, varying motion patterns).<br>

<br>
<br>
<br>

### Why a reference is needed — Sensor Logger orientation as REF <a name="project-ref"></a>

To debug and evaluate the estimator, this project uses the orientation provided by the Sensor Logger application as a reference (REF).<br>

<br>

REF is not treated as perfect ground truth, it is a practical baseline that helps validate implementation correctness and identify failure modes.<br>

<br>

The workflow is intentionally comparative:<br>

- Extract IMU streams (gyro/acc/mag) from recorded logs
- Extract Sensor Logger orientation as REF
- Produce a custom orientation estimate from IMU integration + corrections
- Compute angular error between the estimate and REF
- Analyze drift, transients, and long-horizon behavior
- Iterate on filtering, gating, and calibration logic

<br>
<br>
<br>

### Why trimming the initial seconds matters — Fair Comparison <a name="project-fair-comparison"></a>

A key observation is that the first seconds of a log often contain transient effects — sensor warm-up, bias settling, initial user motion, and convergence of the reference filter.<br>

<br>

These effects can inflate error metrics even when the estimator is correct, and can mislead tuning decisions.<br>

<br>

For this reason, evaluation is performed after trimming an initial stabilization window.<br>
This provides a more fair comparison focused on the behavior that matters for long-term tracking (gyro drift + correction performance), rather than start-up artifacts.<br>


<br>
<br>
<br>

### Experimental roadmap (progressive complexity) <a name="project-exp"></a>

The estimator is developed and validated through a sequence of experiments,<br>
starting from the simplest model and progressively adding corrections and trust control (gating).<br>
Each stage is evaluated using the same error pipeline against REF.<br>

<br>

This progression makes it possible to separate and understand each error source,<br>
such as gyro bias/drift, gravity misinterpretation during translation, magnetic disturbances indoors, and the interaction between tilt estimation and yaw correction.<br>

<br>
<br>
<br>
<br>

## Experiment Result Shortcut <a name="exp"></a>

Full experimental process and results:<br>

- [Experiment 1](https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/experiment/exp1.md)
- [Experiment 2](https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/experiment/exp2.md)
- [Experiment 3](https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/experiment/exp3.md)
- [Experiment 4](https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/experiment/exp4.md)

<br>

##### [Datasets]

4 datasets are used for the experiment recorded by Sensor Logger application.<br>

| Dataset | Duration | Measured by | Posture  | Notes                        |
|:--------|---------:|:-----------:|:--------:|:-----------------------------|
| data 01 | 5 min    | A           | known    | <ul><li>controlled environment</li><li>outdoor setting</li><li>human gait variations only</li></ul> |
| data 02 | 9 min    | A           | known    | <ul><li>controlled environment</li><li>outdoor setting</li><li>human gait variations only</li></ul> |
| data 03 | 13 min   | A           | known    | <ul><li>controlled environment</li><li>outdoor setting</li><li>human gait variations only</li></ul> |
| data 04 | 96 min   | B           | unknown  | <ul><li>uncontrolled environment</li><li>indoor ↔ outdoor transition</li><li>pedestrian + public transport(metro/tram)</li></ul> | 

<br>
<br>

### Experiment 1 (Dataset 03) — Gyro-only propagation <a name="exp-1"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp1/data03_exp1.png" width="952" height="311">

This dataset clearly illustrates why initial stabilization trimming is needed,<br>
and how gyro-only orientation estimation drifts over time.<br>

<br>

##### [Trim decision]

- Stabilization satisfied early `t = 1 s`, but this is considered too early
- Policy enforced (min cut = 10s)

<br>

##### [Metrics]

| exp |  Mean error  |  p90 error   |
|:---:|-------------:|-------------:|
| 1-1 | <ul><li>2.98634 rad</li><li>171.10485 deg</li></ul> | <ul><li>3.12675 rad</li><li>179.14982 deg</li></ul> |
| 1-2 | <ul><li>0.53778 rad</li><li>30.81266 deg</li></ul>  | <ul><li>0.81277 rad</li><li>46.56837 deg</li></ul>  |

<br>

##### [Observation]

- The no-cut run collapses near π radians (≈180° flip), indicating severe divergence
- After trimming, error behaves as expected: steady drift with occasional spikes

<br>
<br>

##### [Conclusion across all datasets]

Experiment 1 shows that:<br>

1. Gyro-only integration exhibits drift over time due to bias accumulation
2. The initial stabilization period should be trimmed for fair evaluation

<br>
<br>
<br>

### Experiment 2 (Dataset 03) — Gyro + Accelerometer <a name="exp-2"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data03_exp2_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data03_exp2_02.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp2/data03_exp2_03.png" width="952" height="471">

This dataset is a representative case where gating improves performance.<br>
Accelerometer correction reduces error overall, but ungated accel updates become less reliable during stronger linear motion.<br>
In this dataset, gating improves the trade-off between average and tail error by suppressing less reliable accel corrections.<br>

<br>

##### [Chosen parameters]

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

##### [Metrics]

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

##### [Secondary validation — Gravity & Linear Accel]

Gravity direction error remains low (mean/p90 ≈ 2.75° / 4.50°), indicating a stable tilt estimate.<br>
Linear-accel direction error is moderate with dynamic spikes (mean/p90 ≈ 11.88° / 23.93°).<br>

<br>

|      |  Mean error  |  p90 error   |
|:----:|-------------:|-------------:|
| grav | <ul><li>0.04878 rad</li><li>2.79477 deg</li></ul> | <ul><li>0.07998 rad</li><li>4.58227 deg</li></ul> |
| acc  | <ul><li>0.20927 rad</li><li>11.99003 deg</li></ul> | <ul><li>0.42003 rad</li><li>24.06610 deg</li></ul> |

<br>

##### [Observation]

- This dataset is another clear case where gating helps substantially relative to ungated gyro+acc correction
- Both accel-only and joint fixed gating reduce error strongly compared with exp 2-1
- The best result is exp 2-3 under the selected ranking criterion, indicating that jointly tuned fixed gyro/acc gating provides the best overall trade-off between mean and tail error on this sequence
- Although exp 2-2 achieves a slightly lower mean error, exp 2-3 yields the best combined result once p90 error is also taken into account
- The time-varying schedule remains competitive, but it does not outperform the best fixed-gating configuration here

<br>
<br>

##### [Conclusion across all datasets]

Experiment 2 shows that:<br>

1. Accelerometer correction improves roll/pitch estimation across the evaluated datasets
2. Gating can provide an additional benefit in certain motion regimes, but its usefulness depends strongly on the dataset and the motion regime
3. The benefit of gating is not universal. On the longest and most difficult sequence evaluated, ungated gyro+acc correction still performs best.
4. The best configuration is therefore dataset-dependent, which reinforces the need to evaluate robustness across multiple motion patterns rather than drawing conclusions from a single sequence

<br>
<br>
<br>

### Experiment 3 (Dataset 03) — Gyro + Accelerometer + Magnetometer <a name="exp-3"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data03_exp3_01.png" width="952" height="311">

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp3/data03_exp3_02.png" width="952" height="631">

This dataset is a representative case where magnetometer correction provides a clear additional benefit.<br>
With magnetic heading correction, orientation error decreases substantially relative to both the gyro-only baseline and the best result from experiment 2.<br>
In this dataset, fixed norm-based gating provides the best trade-off among the tested magnetometer-gating variants.<br>

<br>

##### [Chosen parameters]

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

##### [Metrics]

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

** `exp 1-2` refers to the gyro-only baseline from experiment 1, evaluated on the same trimmed segment.<br>
** `b2` refers to the best experiment 2 result which makes minimum error (calculated by 0.8 * mean error + 0.2 * p90 error), evaluated on the same trimmed segment.<br>

<br>

##### [Secondary validation — Gravity & Linear Accel from Best Exp3]

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

##### [Observation]

- Magnetometer correction substantially improves orientation accuracy relative to the best experiment 2 result on this dataset as well
- The best result is exp 3-3, indicating that fixed norm-based gating provides the best overall result under the selected ranking criterion
- Innovation-only gating (exp 3-2) improves slightly over ungated magnetometer correction, but does not match the best fixed norm-gated configuration
- Adding innovation gating on top of norm-based gating (exp 3-4) remains competitive, but does not surpass exp 3-3 on this sequence
- Time-varying sigma variants again perform worse than the best fixed-gating configurations

<br>
<br>

##### [Conclusion across all datasets]

1. In the evaluated datasets, adding magnetometer correction improves orientation angle error relative to the best gyro+accelerometer configuration from experiment 2 
2. For the relatively consistent datasets in this experiment, fixed-gating configurations perform best among the evaluated variants, with norm-based gating present in each case
3. For the long non-stationary dataset, a time-varying gating configuration performs best under the orientation-only criterion, suggesting that adaptive gating may become more useful under changing conditions, although it does not resolve all failure modes.
4. These results suggest that the next step should focus on improving robustness during gyro integration itself, especially by detecting and suppressing abnormal integrated behavior before it propagates
5. The effect of innovation gating is dataset-dependent: it is sometimes beneficial, especially when combined with norm-based gating, but it is not uniformly dominant across all sequences
6. Secondary validation reveals an important limitation. On dataset 04, better orientation error does not necessarily lead to better gravity and linear-acceleration estimation, indicating that orientation agreement alone is not always a sufficient criterion for model selection.
7. These results suggest that future improvements should be evaluated not only by orientation error, but also by their effect on gravity / linear-accel decomposition, especially for long uncontrolled sequences

<br>
<br>
<br>

### Experiment 4 (Dataset 03) — Segment-based Proxy Objective <a name="exp-4"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/exp4/data03_exp4.png" width="952" height="311">

This dataset shows that segment-based tuning can approximate full-sequence optimization while significantly reducing computational cost.<br>
By optimizing on selected segments instead of the entire sequence, it is possible to retain similar full-sequence performance at much lower cost, and in some cases even obtain slightly better results.<br>

<br>

##### [Chosen parameters]

- quasi-static: (41487, 43669, 2182)
- suggested σ_gyro: 0.4822681
- suggested σ_acc : 0.6755996
- suggested σ_mag : 5.0989239
- `seg_1` head: 5, tail: 5, stride: 15, win: 3
- `seg_2` head: 10, tail: 10, stride: 15, win: 5

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

##### [best]

|              |       exp 3     | exp 4 (`seg_2`) |
|:------------:|:---------------:|:---------------:|
|   ori_best   |     exp3-3      |     exp4-4      |
|  total_best  |     exp3-4      |     exp4-4      |

<br>

##### [Secondary validation — Gravity & Linear Accel from Best Exp4]

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

##### [Observation]

- Segment-based tuning preserves broadly comparable full-sequence performance on this dataset
- In some configurations (notably exp 4-3 and exp 4-4), segment-based tuning yields slightly improved mean and p90 errors relative to full-data tuning
- The best-performing configuration differs slightly from Experiment 3, although the gap between the top candidates remains small
- This suggests that model selection can be sensitive when several configurations perform similarly
- Runtime reduction remains substantial, with speedups of approximately 2.5× to 4×
- `seg_1` and `seg_2` produce similar results on this dataset, without a strong consistent advantage for either policy

<br>

##### [Conclusion across all datasets]

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
<br>
<br>
<br>

## Understanding Coordinate Systems and Sensors <a name="orientation"></a>

The physical modeling and implementation logic behind orientation estimation using:<br>

- Gyroscope
- Accelerometer
- Magnetometer
<br>

The orientation is represented using a quaternion `q` that maps:<br>

```
	q : body → world
```

<br>
<br>

### Coordinate Frame <a name="orientation-coordinate"></a>

#### World Frame (Inertial Frame) <a name="orientation-coordinate-world"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/00_coord_world.png" width="610" height="255">

The world frame is fixed.<br>
<br>

Gravity is assumed constant:<br>

```
	g_world = (0, 0, -g0),
	|| g_world|| = g0

	g0 ≈ 9.81 m/s²
```
<br>
<br>


#### Body Frame (Sensor Frame) <a name="orientation-coordinate-body"></a>

<img src="https://github.com/sleepychloe/IMU_Orientation_Estimation/blob/main/img/01_coord_body.png" width="610" height="255">

The body frame is attached to the device (smartphone).<br>

All sensors measure in the body frame.<br>

When the device rotates:<br>

```
	g_world = (0, 0, -g0),
	g_body = (g𝑥, g𝑦, g𝑧),
	||g_body|| = g0
```
<br>

Quaternion definition:<body>

```
	v_world = R(q) ⋅ v_body
	v_body = R(q)ᵀ ⋅ v_world
```
<br>
<br>
<br>

## Sensor Model <a name="orientation-sensor"></a>

### Gyroscope <a name="orientation-sensor-gyro"></a>

Measurement model:<br>

```
	ω_meas(t) = ω_true(t) + b_g + n_gyro(t)

	b_g: gyro bias
	n_gyro: measurement noise
```

Bias can be estimated during stationary periods.<br>

<br>
<br>

### Accelerometer <a name="orientation-sensor-acc"></a>

The accelerometer measures proper acceleration.<br>
<br>

World frame:<br>

```
	a_proper_world = a_linear_world - g_world
```
<br>

Body frame:<br>

```
	a_apparent_body = R(q)ᵀ⋅(a_linear_world - g_world)
```
<br>

Measurement model:<br>

```
	a_meas(t) = R(q)ᵀ⋅(a_linear_world - g_world) + b_a + n_acc(t)

	b_a: acc bias
	n_acc: measurement noise
```
<br>

At rest (a_linear_world = 0):<br>

```
	a_meas ≈ - R(q)ᵀ⋅g_world,

	a_meas / ||a_meas|| ≈ - g_body / ||g_body|| = - g_body_unit
```

<br>
<br>

### Magnetometer <a name="orientation-sensor-mag"></a>

Measurement model:<br>
```
	m_meas(t) = A * m_true(t) + b_hi n_mag(t)

	A: soft-iron 3x3 matrix
	b_hi: hard-iron bias
	n_mag: measurement noise
```
<br>

Ideal case:<br>

```
	m_body = R(q)ᵀ⋅ m_world
```
<br>
Distortions include:<br>

- Hard-iron offset
- Soft-iron scaling (3x3 matrix)

<br>
Thus magnetometer reliability requires:<br>

- Norm gate
- Innovation gate
- Calibration

<br>
<br>
<br>

### Gravity vs Magnetic Field <a name="orientation-grav-mag"></a>

Gravity:<br>

- Nearly constant magnitude
- Direction fixed in world frame
- ||g_world|| = g0
<br>

Magnetic field:<br>

- Magnitude varies by location
- Affected by indoor environment
- Yaw reference is not physically absolute
<br>

Two possible heading references in magnetic field:<br>

1. Absolute heading (true north, declination corrected)
2. Relative heading (initial direction = yaw 0, often more stable indoors)

<br>
<br>
<br>

#### Initial Magnetic Reference (Yaw Anchor) <a name="orientation-grav-mag-init-mag-ref"></a>

Goal: `m_ref_world_h`, used to correct yaw drift.<br>
<br>

1. Stationary Detection

```
	For reliable bias estimation:

	1. | ||a_meas|| - g0| ≈ 0
	2. |ω| ≈ 0
```

<br>

2. Horizontal Projection

```
	m̂ = m_meas / ||m_meas||
	m_body_h = m̂ - (m̂ ⋅ g_body_unit) * g_body_unit
	m̂_body_h = m_body_h / ||m_body_h||
```

<br>

3. Transform to world frame

```
	m̂_world_h = R(q_pred)⋅ m̂_body_h
	                    Σ_{t ∈ T} weight(t) * m̂_world_h(t)
	m_ref_world_h = ───────────────────────────────────────
	                 || Σ_{t ∈ T} weight(t) * m̂_world_h(t) ||

	T: initial stable window
	weight(t): weighting (stationary + norm gate)
``` 

<br>

This defines a stable yaw reference without requiring absolute north.<br>

<br>
<br>
<br>
<br>

## Implementation – IMU Orientation Estimation <a name="implementation"></a>

Implementation logic of quaternion-based orientation estimation using:<br>

- Gyroscope (Propagation)
- Accelerometer (Roll/Pitch correction)
- Magnetometer (Yaw correction)
<br>

The quaternion `q` is defined as:<br>

```
	q : body → world
```

<br>
<br>

### [Step 1] Gyroscope Propagation <a name="implementation-gyro"></a>

1. Continuous quaternion dynamics (body → world):<br>

```
	q̇(t) = 1/2 ⋅ q(t) ⊗ Ω(ω𝑚𝑒𝑎𝑠(t))

	Ω(ω) : [0, ω𝑥, ω𝑦, ω𝑧]
```
<br>

2. Discrete integration over dt:<br>

```
	θ = ||ω||⋅dt
	u = ω / ||ω|| (if ||ω|| > 0)
```
<br>

3. Small rotation quaternion ∆q𝑔𝑦𝑟𝑜:<br>

```
	        ┏              ┓
	        ┃   cos(θ/2)   ┃
	∆q𝑔𝑦𝑟𝑜 = ┃  u𝑥⋅sin(θ/2) ┃
	        ┃  u𝑦⋅sin(θ/2) ┃
	        ┃  u𝑧⋅sin(θ/2) ┃
	        ┗              ┛
```
<br>

4. Prediction update:<br>

```
	q𝑝𝑟𝑒𝑑 = normalize(q ⊗ ∆q𝑔𝑦𝑟𝑜)
```

<br>
<br>

### [Step 2] Accelerometer Correction (Roll/Pitch) <a name="implementation-acc"></a>

1. World gravity direction:<br>

```
	g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡 = g𝑤𝑜𝑟𝑙𝑑 / ||g𝑤𝑜𝑟𝑙𝑑|| = (0, 0, -1)
```
<br>

2.  Predicted gravity direction in body frame:<br>

```
	g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 = R(q𝑝𝑟𝑒𝑑)ᵀ⋅g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡
```
<br>

3. Error axis:<br>

```
	e_axis𝑎𝑐𝑐 = g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 × (-a𝑚𝑒𝑎𝑠_𝑢𝑛𝑖𝑡)

	||e_axis𝑎𝑐𝑐|| = sinφ
```
<br>

4. Accel gating:<br>

```
	dev𝑎𝑐𝑐 = | ||a𝑚𝑒𝑎𝑠|| - g0 |
	weight𝑎𝑐𝑐 = exp( -1/2 * (dev𝑎𝑐𝑐 / σ𝑎𝑐𝑐)²)

	σ𝑎𝑐𝑐: accel gating sigma
```
<br>

(optional) Gyro gating:<br>

```
	weight𝑔𝑦𝑟𝑜 = exp( -1/2 * ( ω𝑛𝑜𝑟𝑚 / σ𝑔𝑦𝑟𝑜)²)
	weight𝑎𝑐𝑐 = weight𝑎𝑐𝑐 * weight𝑔𝑦𝑟𝑜

	σ𝑔𝑦𝑟𝑜: gyro gating sigma
```
<br>

5. Correction quaternion ∆q𝑐𝑜𝑟𝑟:<br>

```
	        ┏                                     ┓
	        ┃                  1                  ┃
	∆q𝑐𝑜𝑟𝑟 = ┃  1/2 ⋅ K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐_𝑥 ┃
	        ┃  1/2 ⋅ K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐_𝑦 ┃
	        ┃  1/2 ⋅ K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐_𝑧 ┃
	        ┗                                     ┛
```
<br>

6. Update:<br>

```
	q̂ = normalize(q𝑝𝑟𝑒𝑑 ⊗ ∆q𝑐𝑜𝑟𝑟)
```


<br>
<br>

### [Step 3] Linear Acceleration Estimation <a name="implementation-acc-linear"></a>

1. Predicted gravity vector in body frame:<br>

```
	g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 = g0 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡
```
<br>

2. Predicted linear acceleration in body frame:<br>
```
	a𝑒𝑠𝑡_𝑙𝑖𝑛𝑒𝑎𝑟_𝑏𝑜𝑑𝑦 ≈ a𝑚𝑒𝑎𝑠 + g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦
```
<br>
<br>

### [Step 4] Magnetometer Correction (Yaw) <a name="implementation-mag"></a>

Yaw does not change gravity direction:<br>

```
	R𝑧(ψ)ᵀ ⋅ g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡 = g𝑤𝑜𝑟𝑙𝑑_𝑢𝑛𝑖𝑡
```
<br>
Thus magnetometer is required for heading correction.<br>
<br>

1. Tilt compensation(Projecting mag onto the horizontal plane — remove gravity component):<br>

```
	Normalize magnetic measurement:
		m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 = m𝑚𝑒𝑎𝑠 / ||m𝑚𝑒𝑎𝑠||

	Remove gravity component:
		m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ = m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 - (m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡) * g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡

	Normalize:
		m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡 = m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ / ||m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ||
```
<br>

2. Predicted magnetic direction:<br>

```
	Predicted magnetic field in body frame:
		m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 = R(q𝑝𝑟𝑒𝑑)ᵀ⋅m𝑟𝑒𝑓_𝑤𝑜𝑟𝑙𝑑_ℎ

	Remove gravity component:
		m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ = m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 - (m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡) * g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡

	Normalize:
		m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡 = m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ / ||m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ||
```
<br>

3. Error axis:<br>

```
	e_axis𝑚𝑎𝑔 = m𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡 × m𝑚𝑒𝑎𝑠_𝑏𝑜𝑑𝑦_ℎ_𝑢𝑛𝑖𝑡

	||e_axis𝑚𝑎𝑔|| = sinφ
```
<br>

4. Magnetometer gating<br>

4-1. Norm gate:<br>

```
	m₀ = median( ||m𝑚𝑒𝑎𝑠|| )
	dev𝑚𝑎𝑔 = | ||m𝑚𝑒𝑎𝑠|| - m₀ |
	weight𝑚𝑎𝑔 = exp ( -1/2 * (dev𝑚𝑎𝑔 / σ𝑚𝑎𝑔)²)

	σ𝑚𝑎𝑔: mag gating sigma
```
<br>

Use `median` because it is robust to outlier.<br>

<br>

(optional) Gyro gating:<br>

```
	weight𝑔𝑦𝑟𝑜 = exp( -1/2 * ( ω𝑛𝑜𝑟𝑚 / σ𝑔𝑦𝑟𝑜)²)
	weight𝑚𝑎𝑔 = weight𝑚𝑎𝑔 * weight𝑔𝑦𝑟𝑜

	σ𝑔𝑦𝑟𝑜: gyro gating sigma
```
<br>

4-2. Innovation gate:<br>

```
	weight𝑚𝑎𝑔 = weight𝑚𝑎𝑔 * exp( -1/2 * (||e_axis𝑚𝑎𝑔|| / σ𝑒_𝑚𝑎𝑔)² )

	σ𝑒_𝑚𝑎𝑔: mag error sigma (σ𝑒_𝑚𝑎𝑔 and σ𝑚𝑎𝑔 above is independent value)
```
<br>

Reduce impact of mag if ||e_axis𝑚𝑎𝑔|| changes abruptly.<br>

<br>
<br>

5. Correction quaternion ∆q𝑐𝑜𝑟𝑟:<br>

Total correction vector:<br>

```
	e𝑡𝑜𝑡𝑎𝑙 = K𝑎𝑐𝑐 ⋅ weight𝑎𝑐𝑐 ⋅ e_axis𝑎𝑐𝑐 + K𝑚𝑎𝑔 ⋅ weight𝑚𝑎𝑔 ⋅ e_axis𝑚𝑎𝑔

	       ┏                    ┓
	       ┃          1         ┃
	∆𝑐𝑜𝑟𝑟 = ┃  1/2 ⋅ e_axis𝑡𝑜𝑡𝑎𝑙_𝑥 ┃
	       ┃  1/2 ⋅ e_axis𝑡𝑜𝑡𝑎𝑙_𝑦 ┃
	       ┃  1/2 ⋅ e_axis𝑡𝑜𝑡𝑎𝑙_𝑧 ┃
	       ┗                    ┛
```
<br>

6. Final update:<br>

```
	q̂ = normalize(q𝑝𝑟𝑒𝑑 ⊗ ∆q𝑐𝑜𝑟𝑟)
```

<br>
<br>

### Gating Observation After Implementation <a name="implementation-gating-observation"></a>

I observed that gyro-based stationary gating improves Gyro+Acc performance, but can degrade Gyro+Acc+Mag performance.<br>

<br>

In Gyro+Acc, a gyro-based stationary gate (small ||ω||) helps because it reduces the probability of applying accelerometer-based gravity correction during motion.<br>

<br>

During translation, ||a𝑚𝑒𝑎𝑠|| often deviates from g0,<br>
meaning the accelerometer direction is no longer a clean gravity measurement and can inject incorrect tilt corrections.<br>

<br>
<br>

However, in Gyro+Acc+Mag, magnetometer yaw correction depends on a stable tilt estimate (roll/pitch),<br>
because tilt compensation projects the measured magnetic vector onto the gravity-defined horizontal plane:<br>

```
	m_ℎ = m_𝑢𝑛𝑖𝑡 - (m_𝑢𝑛𝑖𝑡 ⋅ g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡) * g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡
```
<br>

If gyro gating suppresses accel correction too aggressively (especially during rotation or dynamic motion), roll/pitch estimation can become noisier (gyro drift + less accel correction).<br>
The destabilizes g𝑒𝑠𝑡_𝑏𝑜𝑑𝑦_𝑢𝑛𝑖𝑡, which makes tilt compensation less reliable, and can corrupt the magnetometer error axis,<br>
worsening the yaw correction even when mag gating is enabled.<br>

<br>
<br>

Conclution:<br>

1. Use gyro-based gating primarily to control accelerometer trust (gravity correction)

2. For magnetometer updates, prefer magnitude/innovation gates, and avoid fully disabling mag updates during rotation

<br>
<br>
<br>
<br>


## Quaternion <a name="quaternion"></a>

### Quaternion <a name="quaternion-quaternion"></a>

In 3D, a rotation must represent both:<br>
- rotation axis
- rotation angle
<br>

Quaternion is a 4D vector representation for expressing 3D rotations.<br>
<br>

A unit quaternion is written as:<br>

```
	q = (w, x, y, z),

	where w is a scalar component,
	(x, y, z) are the vector components.

	and unit-norm condition:
	||q|| = 1
```

or equivalently:<br>

```
	q = w + xi + yj + zk,

	where i, j, k are imaginary basis elements satisfying:
	i² = j² = k² = ijk = -1,
	ij = k, jk = i, ki = j,
	ji = -k, kj = -i, ik = -j,

	and unit-norm condition:
	w² + x² + y² + z² = 1
```

Because of constraint `w² + x² + y² + z² = 1`, the actual degrees of freedom are 3,<br>
when a quaternion is 4D(4 values).<br>

Also, note that `q` and `-q` represent the same physical 3D rotation<br>
(double-cover property of unit quaternions over SO(3)).<br>
<br>
<br>

### Axis-Angle to Quaternion <a name="quaternion-axis-angle-to-quaternion"></a>

A quaternion expresses "rotation by θ around some axis".<br>

```
	q = (cos(θ/2), sin(θ/2)⋅u𝑥, sin(θ/2)⋅u𝑦, sin(θ/2)⋅u𝑧)

	θ: rotation angle
	u(u𝑥, u𝑦, u𝑧): rotation axis unit vector

	in other words:

	w = cos(θ/2),
	(x, y, z) = u⋅sin(θ/2), where ||u|| = 1
```
<br>
<br>

### Euler Angle vs Quaternion <a name="quaternion-euler-angle-vs-quaternion"></a>

- Euler Angle
	- Represents orientation with 3 sequential rotations (roll/pitch/yaw)
	- Parameterization by sequential axis rotations (stores its process — dependent)
	- Intuitive, but can suffer from gimbal lock(loss of one rotational DOF when axes align)

- Quaternion
	- Unit quaternion representation of 3D orientation, maps to SO(3) up to sign (stores its result — independent)
	- Rotation can be composed via quaternion multiplication
	- Stable in continuous rotation
	- No gimbal lock
	- Numerically robust for continuous IMU integration
	- `q` and `-q` encode the same orientation (same rotation in 3D space)

<br>
<br>

### Applications of Quaternion <a name="quaternion-applications-of-quaternion"></a>

#### IMU Orientation Update <a name="quaternion-applications-imu"></a>

Using gyroscope angular velocity `ω = (ω𝑥, ω𝑦, ω𝑧)`(integrated over time),<br>
orientation is propagated by:<br>

```
	qₜ₊Δₜ = qₜ ⊗ Δq

	⊗: quaternion multiplication,
	Δq: incremental rotation over Δt.
```

This is the core orientation prediction step in inertial navigation.<br>

<br>
<br>

#### Rotating a Vector with Quaternion <a name="quaternion-applications-vector"></a>

To rotate a vector v into world frame:<br>

```
	v𝑤𝑜𝑟𝑙𝑑 = q ⊗ v ⊗ q⁻¹

	v: embed as pure quaternion (0, v)

	(Depending on active/passive rotation convention and frame definition,
	equivalent forms such as `q ⊗ v ⊗ q⁻¹` or `q⁻¹ ⊗ v ⊗ q` may appear)
```

This is used to separate gravity, or convert sensor-frame quantities into global coordinates.<br>

<br>
<br>
<br>

### Rotation <a name="quaternion-rotation"></a>

#### 2D Rotation <a name="quaternion-rotation-2d"></a>

In 2D (complex plane), Euler's formula gives:<br>

```
	eⁱᶿ = cosθ + i⋅sinθ

	(A point on the unit circle in the complex plane)
```

Taylor expansion of exponential function:<br>

```
	1. by Taylor series, any smooth function can be expanded like:

	   f(x) = f(0) + f′(0)⋅x + f″(0)/2!⋅x² + f′″(0)/3!⋅x³ + ⋅⋅⋅.


	                             d
	2. on exponential function, ── eˣ = eˣ, so all derivatives are 1 at x=0.
	                            dx


	3. eⁱᶿ = 1 + iθ + (iθ)²/2! + (iθ)³/3! + (iθ)⁴/4! + ⋅⋅⋅,

	   since i² = -1, when we divide the even/odd degree term,

	   [even degree term] 1 + (iθ)²/2! + (iθ)⁴/4! + ⋅⋅⋅
	                       = 1 - θ²/2! + θ⁴/4! - θ⁶/6! + ⋅⋅⋅
			       = cosθ
	   [odd degree term] iθ + (iθ)³/3! + (iθ)⁵/5! + ⋅⋅⋅
	                      = i⋅(θ - θ³/3! + θ⁵/5! - θ⁷/7! + ⋅⋅⋅)
			      = i⋅sinθ
	
	∴ eⁱᶿ = cosθ + i⋅sinθ
```

2D Rotation is a movement of θ on a unit circle.<br>
Also, complex number multiplication is angle addition.<br>

```
	eⁱᵃ ⋅ eⁱᵇ = eⁱ⁽ᵃ⁺ᵇ⁾ = cos(a+b) + i⋅sin(a+b)
```

<br>
<br>

#### 3D Rotation <a name="quaternion-rotation-3d"></a>

Whereas 2D has only one rotation axis (z-axis), 3D has infinitely many axes of rotation.<br>
That's why 2D rotations can be represented by complex numbers,<br>
but general 3D rotations require quaternions.<br>

In quaternion q = (w, u𝑥, u𝑦, u𝑧),<br>
u is u𝑥⋅i + u𝑦⋅j + u𝑧⋅k,<br>
when u is unit axis, u² = -1.<br>

Taylor expansion of exponential function:<br>

```
	eᵘᶿ = 1 + uθ + (uθ)²/2! + (uθ)³/3! + (uθ)⁴/4! + ⋅⋅⋅,

	   since u² = -1, when we divide the even/odd degree term,

	   [even degree term] 1 + (uθ)²/2! + (uθ)⁴/4! + ⋅⋅⋅
	                       = 1 - θ²/2! + θ⁴/4! - θ⁶/6! + ⋅⋅⋅
			       = cosθ
	   [odd degree term] uθ + (uθ)³/3! + (uθ)⁵/5! + ⋅⋅⋅
	                      = u⋅(θ - θ³/3! + θ⁵/5! - θ⁷/7! + ⋅⋅⋅)
			      = u⋅sinθ

	∴ eᵘᶿ = cosθ + u⋅sinθ
	      = cosθ + sinθ⋅u𝑥⋅i + sinθ⋅u𝑦⋅j + sinθ⋅u𝑧⋅k
```

How we rotate vector in 3D is `v' = q ⊗ v ⊗ q⁻¹`, which applies quaternion multiplication on both sides.<br>
So when we define `q = cos(θ/2) + u⋅sin(θ/2)`, the result of `v' = q ⊗ v ⊗ q⁻¹` rotates exactly θ.<br>

Since the space where 3d vector(R³) and quaternion(R⁴) exist are different,<br>
If it is simply multiplied from the left like when 2D rotation is applied,<br>
a scalar component that should not be there is created.<br>

That's why 3D rotation is applied by the "sandwich" form,<br>
which is interpreted:<br>

1. lift v into quaternion space (pure quaternion)
2. rotate by quaternion multiplication
3. project back to 3D vector

<br>
<br>
<br>
<br>
