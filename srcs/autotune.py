import numpy as np
from typing import Any

from my_types import Vec3, Quat, ScalarBatch, Vec3Batch, BoolBatch, Int8Batch, Int32Batch, Int64Batch
from my_types import as_vec3, as_scalar_batch, as_bool_batch, as_int8_batch
from pipelines import integrate_gyro_acc_no_gate, integrate_gyro_acc_with_gate_acc, integrate_gyro_acc_with_gate_gyro_acc

EPS: float = 1e-6
DELTA: float = 1e-12

def stats_basic(x: float) -> tuple[float, ...]:
        """
        Returns:
                min, max, mean, p50, p90, p99
        """
        return (np.min(x), np.max(x), np.mean(x),
                np.percentile(x, 50), np.percentile(x, 90), np.percentile(x, 99))

def largest_true_run(mask: BoolBatch) -> tuple[int, int, int] | None:
        """
        Returns:
                start, end, length
        """
        if mask.size == 0 or not mask.any():
                return None

        m: Int8Batch = as_int8_batch(mask.astype(np.int8, copy=False))

        # diff == 1: 0(False) → 1(True): start point
        # diff == -1: 1(True) → 0(False): end point
        diff: Int8Batch = as_int8_batch(np.diff(m).astype(np.int8, copy=False))
        
        start: Int64Batch = np.where(diff == 1)[0] + 1
        end: Int64Batch = np.where(diff == -1)[0] + 1

        if bool(mask[0]):
                start = np.r_[np.array([0], dtype=start.dtype), start]
        if bool(mask[-1]):
                end = np.r_[end, np.array([mask.size], dtype=end.dtype)]

        length: Int64Batch = end - start

        i: int = int(np.argmax(length))
        return int(start[i]), int(end[i]), int(length[i])

def smooth_bool(mask: BoolBatch, win: int = 5) -> BoolBatch:
        """
        Simple debouncing: True if majority True in a window.
        """
        if win <= 1:
                return mask

        # BoolBatch → Int8Batch for 0/1 majority vote
        m: Int8Batch = as_int8_batch(mask.astype(np.int8, copy=False))
        kernel: Int32Batch = np.ones(win, dtype=np.int32)

        s: Int64Batch = np.convolve(m, kernel, mode="same")
        res: BoolBatch = (s >= (win//2 + 1))
        return as_bool_batch(res)

def detect_still_segments(w: Vec3Batch, a: Vec3Batch, dt: ScalarBatch, g0: float,
                          w_thr: float, a_thr: float,
                          min_duration_s: float, smooth_win: int,
                          ) -> tuple[BoolBatch, dict[str, Any], tuple[int, int, int]]:
        """
        Returns:
                still_mask: (N,) BoolBatch
                still_info: dict[str, Any], threshold and stats
                best_run: tuple[int, int, int], longest still segment
        """
        w_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(w, axis=1))
        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a, axis=1))
        acc_resid: float = np.abs(a_norm - g0)

        # If thresholds not provided, choose them from distribution
        # - w_thr: "very low rotation" => use a low percentile
        # - a_thr: "close to gravity magnitude"
        if w_thr is None:
                w_thr = float(np.percentile(w_norm, 30))
        if a_thr is None:
                a_thr = float(np.percentile(acc_resid, 30))

        raw_mask: BoolBatch = (w_norm < w_thr) & (acc_resid < a_thr)
        still_mask: BoolBatch = smooth_bool(raw_mask, win=smooth_win)

        # enforce min duration
        # convert duration to samples roughly using median dt
        dt_medean: float = np.median(dt)
        min_len: int = int(np.ceil(min_duration_s / max(dt_medean, EPS)))

        best_run: tuple[int, int, int] = largest_true_run(still_mask)
        if best_run is not None and best_run[2] < min_len:
                best_run = None

        still_info: dict[str, Any] = {
                "w_thr": w_thr,
                "a_thr": a_thr,
                "min_duration_s": min_duration_s,
                "smooth_win": smooth_win,
                "dt_median": dt_medean,
                "gyro_norm_stats": stats_basic(w_norm),
                "acc_resid_stats": stats_basic(acc_resid),
                "still_fraction": float(np.mean(still_mask))
        }
        return still_mask, still_info, best_run

#def suggest_acc_sigma(a: Vec3Batch, g0: float,
#                  p_acc: int = 90, sigma_floor: float = 1e-3
#                  ) -> tuple[float, float, dict[str, Any]]:
#        """
#        Returns:
#                acc_sigma: float
#                sigma_info: dict[str, Any]
#        """
#        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a, axis=1))
#        acc_resid: float = np.abs(a_norm - g0)

#        acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))

#        sigma_info: dict[str, Any] = {
#                "p_acc": int(p_acc),
#                "acc_resid_p50_p90_p99": tuple(np.percentile(acc_resid, [50, 90, 99]))
#        }
#        return acc_sigma, sigma_info

def suggest_acc_sigma(a: Vec3Batch, g0: float,
                  p_acc: int = 90, sigma_floor: float = 1e-3,
                  segment: tuple[int, int] = None,
                  ) -> tuple[float, float, dict[str, Any]]:
        """
        Returns:
                acc_sigma: float
                sigma_info: dict[str, Any]
        """
        if segment is not None:
                s, e = segment
                a_use = a[s:e]
        else:
                a_use = a
        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a_use, axis=1))
        acc_resid: float = np.abs(a_norm - g0)

        acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))

        sigma_info: dict[str, Any] = {
                "p_acc": int(p_acc),
                "acc_resid_p50_p90_p99": tuple(np.percentile(acc_resid, [50, 90, 99]))
        }
        return acc_sigma, sigma_info

#def suggest_gyro_acc_sigma(w: Vec3Batch, a: Vec3Batch, g0: float,
#                  p_gyro: int, p_acc: int, sigma_floor: float,
#                  ) -> tuple[float, float, dict[str, Any]]:
#        """
#        Returns:
#                gyro_sigma: float
#                acc_sigma: float
#                sigma_info: dict[str, Any]
#        """
#        w_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(w, axis=1))
#        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a, axis=1))
#        acc_resid: float = np.abs(a_norm - g0)

#        gyro_sigma: float = max(sigma_floor, float(np.percentile(w_norm, p_gyro)))
#        acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))

#        sigma_info: dict[str, Any] = {
#                "p_gyro": int(p_gyro),
#                "p_acc": int(p_acc),
#                "gyro_norm_p50_p90_p99": tuple(np.percentile(w_norm,[50, 90, 99])),
#                "acc_resid_p50_p90_p99": tuple(np.percentile(acc_resid, [50, 90, 99]))
#        }
#        return gyro_sigma, acc_sigma, sigma_info

def suggest_gyro_acc_sigma(w: Vec3Batch, a: Vec3Batch, g0: float,
                  p_gyro: int, p_acc: int, sigma_floor: float,
                  segment: tuple[int, int] = None,
                  ) -> tuple[float, float, dict[str, Any]]:
        """
        Returns:
                gyro_sigma: float
                acc_sigma: float
                sigma_info: dict[str, Any]
        """
        if segment is not None:
                s, e = segment
                a_use = a[s:e]
        else:
                a_use = a

        w_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(w, axis=1))
        a_norm: ScalarBatch = as_scalar_batch(np.linalg.norm(a_use, axis=1))
        acc_resid: float = np.abs(a_norm - g0)

        gyro_sigma: float = max(sigma_floor, float(np.percentile(w_norm, p_gyro)))
        acc_sigma: float = max(sigma_floor, float(np.percentile(acc_resid, p_acc)))

        sigma_info: dict[str, Any] = {
                "p_gyro": int(p_gyro),
                "p_acc": int(p_acc),
                "gyro_norm_p50_p90_p99": tuple(np.percentile(w_norm,[50, 90, 99])),
                "acc_resid_p50_p90_p99": tuple(np.percentile(acc_resid, [50, 90, 99]))
        }
        return gyro_sigma, acc_sigma, sigma_info

def choose_tau_from_still_segment(q0: Quat, w: Vec3Batch, dt: ScalarBatch, a: Vec3Batch,
                                  g0: float, g_world_unit: Vec3,
                                  acc_gate_sigma: float, gyro_gate_sigma: float,
                                  tau_candidates: tuple[float, ...],
                                  still_run: tuple[int, int, int] | None = None
                                  ) -> tuple[float, list[dict[str, Any]]]:
        """
        Returns:
                best_tau: float
                tau_table: dict[str, Any]
        """
        if still_run is None:
                return 0.25, [{"tau": 0.25, "score": None, "note": "no still segment found"}]

        s, e, _ = still_run
        dt_medean: float = float(np.median(dt).reshape(-1))

        tau_table: list[dict[str, Any]] = []
        for tau in tau_candidates:
                K = float(dt_medean / tau)

                if not np.isinf(gyro_gate_sigma) and not np.isinf(acc_gate_sigma):
                        q, g_body_est, a_lin_est, weight_acc, weight_gyro = integrate_gyro_acc_with_gate_gyro_acc(
                                                                q0.copy(), w, dt,
                                                                K, g0, g_world_unit,
                                                                acc_gate_sigma, gyro_gate_sigma, a)
                elif np.isinf(gyro_gate_sigma) and not np.isinf(acc_gate_sigma):
                        q, g_body_est, a_lin_est, weight_acc, weight_gyro = integrate_gyro_acc_with_gate_acc(
                                                                q0.copy(), w, dt,
                                                                K, g0, g_world_unit,
                                                                acc_gate_sigma, a)
                        weight_gyro: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))
                else:
                        q, g_body_est, a_lin_est = integrate_gyro_acc_no_gate(
                                                                q0.copy(), w, dt,
                                                                K, g0, g_world_unit,
                                                                a)
                        weight_acc: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))
                        weight_gyro: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))

                # In still segment, gravity in body frame should be stable.
                # Use direction variance as a stability proxy.
                gb = g_body_est[s:e]
                gb_unit = gb / (np.linalg.norm(gb, axis=1, keepdims=True) + DELTA)

                # score: average angular deviation from mean direction
                mean_dir: Vec3 = as_vec3(np.mean(gb_unit, axis=0))
                mean_dir = mean_dir / (np.linalg.norm(mean_dir) + DELTA)
                dot: ScalarBatch = np.clip(gb_unit @ mean_dir, -1, 1)
                ang: ScalarBatch = np.arccos(dot)
                score: float = float(np.mean(ang)) # lower is better

                tau_table.append({
                        "tau": float(tau),
                        "K": K,
                        "still_score_mean_angle(rad)": score,
                        "mean_weight_acc": np.mean(weight_acc[s:e]),
                        "mean_weight_gyro": np.mean(weight_gyro[s:e]),
                })
        tau_table.sort(key=lambda d: d["still_score_mean_angle(rad)"])
        best_tau: float = tau_table[0]["tau"]
        return best_tau, tau_table

def auto_param_exp2_1(q0: Quat, w: Vec3Batch, dt: ScalarBatch, a: Vec3Batch,
                g0: float, g_world_unit: Vec3,
                min_duration_s: float, smooth_win: int,
                w_thr: float, a_thr: float,
                tau_candidates: tuple[float, ...] = (0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1)
                ) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int, int]]:
        """
        Returns:
                param: dict[str, Any]
                diag: dict[str, Any]
                still_mask: tuple[int, int, int]
        """
        still_mask, still_info, best_run = detect_still_segments(w, a, dt, g0,
                                                                 w_thr, a_thr,
                                                                 min_duration_s, smooth_win)

        gyro_sigma: float = np.inf
        acc_sigma: float = np.inf

        best_tau, tau_table = choose_tau_from_still_segment(q0, w, dt, a,
                                                            g0, g_world_unit,
                                                            gyro_sigma, acc_sigma,
                                                            tau_candidates, best_run)

        dt_medean: float = np.median(dt)
        K: float = dt_medean / best_tau

        param: dict[str, Any] = {
                "K": K,
                "acc_gate_sigma": acc_sigma,
                "gyro_gate_sigma": gyro_sigma,
                "tau": best_tau,
                "best_run": best_run
        }
        
        diag: dict[str, Any] = {
                "still_info": still_info,
                "sigma_info": None,
                "tau_table_top3": tau_table[:3]
        }
        return param, diag, still_mask

def auto_param_exp2_2(q0: Quat, w: Vec3Batch, dt: ScalarBatch, a: Vec3Batch,
                g0: float, g_world_unit: Vec3,
                p_acc: int, sigma_floor: float,
                min_duration_s: float, smooth_win: int,
                w_thr: float, a_thr: float,
                sigma_scale_candidates: tuple[float, ...] = None,
                tau_candidates: tuple[float, ...] = (0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1)
                ) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int, int]]:
        """
        Returns:
                param: dict[str, Any]
                diag: dict[str, Any]
                still_mask: tuple[int, int, int]
        """
        still_mask, still_info, best_run = detect_still_segments(w, a, dt, g0,
                                                                 w_thr, a_thr,
                                                                 min_duration_s, smooth_win)

        gyro_sigma: float = np.inf

        if best_run is not None:
                s, e, _ = best_run
                acc_sigma, sigma_info = suggest_acc_sigma(a, g0,
                                                          p_acc, sigma_floor,
                                                          segment=(s,e))
        else:
                acc_sigma, sigma_info = suggest_acc_sigma(a, g0,
                                                          p_acc, sigma_floor)

        best_tau, tau_table = choose_tau_from_still_segment(q0, w, dt, a,
                                                            g0, g_world_unit,
                                                            gyro_sigma, acc_sigma,
                                                            tau_candidates, best_run)

        dt_medean: float = np.median(dt)
        K: float = dt_medean / best_tau

        param: dict[str, Any] = {
                "K": K,
                "acc_gate_sigma": acc_sigma,
                "gyro_gate_sigma": gyro_sigma,
                "tau": best_tau,
                "best_run": best_run
        }
        
        diag: dict[str, Any] = {
                "still_info": still_info,
                "sigma_info": sigma_info,
                "tau_table_top3": tau_table[:3]
        }
        return param, diag, still_mask

def auto_param_exp2_3(q0: Quat, w: Vec3Batch, dt: ScalarBatch, a: Vec3Batch,
                g0: float, g_world_unit: Vec3,
                p_gyro: int, p_acc: int, sigma_floor: float,
                min_duration_s: float, smooth_win: int,
                w_thr: float, a_thr: float,
                tau_candidates: tuple[float, ...] = (0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1)
                ) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int, int]]:
        """
        Returns:
                param: dict[str, Any]
                diag: dict[str, Any]
                still_mask: tuple[int, int, int]
        """
        still_mask, still_info, best_run = detect_still_segments(w, a, dt, g0,
                                                                 w_thr, a_thr,
                                                                 min_duration_s, smooth_win)

        if best_run is not None:
                s, e, _ = best_run
                gyro_sigma, acc_sigma, sigma_info = suggest_gyro_acc_sigma(w, a, g0,
                                                                           p_gyro, p_acc, sigma_floor,
                                                                           segment=(s,e))
        else:
                gyro_sigma, acc_sigma, sigma_info = suggest_gyro_acc_sigma(w, a, g0,
                                                                           p_gyro, p_acc, sigma_floor)

        best_tau, tau_table = choose_tau_from_still_segment(q0, w, dt, a,
                                                            g0, g_world_unit,
                                                            gyro_sigma, acc_sigma,
                                                            tau_candidates, best_run)

        dt_medean: float = np.median(dt)
        K: float = dt_medean / best_tau

        param: dict[str, Any] = {
                "K": K,
                "acc_gate_sigma": acc_sigma,
                "gyro_gate_sigma": gyro_sigma,
                "tau": best_tau,
                "best_run": best_run
        }
        
        diag: dict[str, Any] = {
                "still_info": still_info,
                "sigma_info": sigma_info,
                "tau_table_top3": tau_table[:3]
        }
        return param, diag, still_mask
