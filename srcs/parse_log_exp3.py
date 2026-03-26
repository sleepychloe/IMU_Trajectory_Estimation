from typing import Any
import re

from my_types import Quat, ScalarBatch, Vec3, Vec3Batch, QuatBatch
from autotune import suggest_timevarying_gate_sigma
from pipelines import integrate_gyro_acc_mag
from evaluation import calc_angle_err, score_angle_err, calc_vec3_direction_angle_err

def parse_scalar(s: str) -> float:
        s: str = s.strip().lower()
        if s == "inf":
                return float("inf")
        return float(s)

def parse_exp3_param(text: str) -> dict[str, dict[str, Any]]:
        """
        return:
                "3-1": {
                        "tau": ...,
                        "K": ...,
                        ...
                        "mode": "fixed"},
                ...
                "3-5": {
                        "tau": ...,
                        "K": ...,
                        ...
                        "best_p": ...,
                        ...
                        "mode": "time_varying"},
                ...
        """
        pattern = re.compile(r"\[chosen value\]\s*(.*?)\s*\[exp (3-[1-6])\]", re.DOTALL)
        res: dict[str, dict[str, Any]] = {}

        for block, exp in pattern.findall(text):
                param: dict[str, Any] = {}

                # tau, K
                m = re.search(r"tau=\s*([^,\n]+)\s*,\s*K=\s*([^\n]+)", block)
                if not m:
                        raise ValueError(f"tau / K not found for exp {exp}")
                param["tau"] = parse_scalar(m.group(1))
                param["K"] = parse_scalar(m.group(2))

                # mag_gain
                m = re.search(r"mag_gain=\s*([^\n]+)", block)
                if not m:
                        raise ValueError(f"mag_gain not found for exp {exp}")
                param["mag_gain"] = parse_scalar(m.group(1))

                # mag_err_sigma
                m = re.search(r"mag_err_sigma=\s*([^\n]+)", block)
                if not m:
                        raise ValueError(f"mag_err_sigma not found for exp {exp}")
                param["mag_err_sigma"] = parse_scalar(m.group(1))

                # 3-1 ~ 3-4: fixed sigma
                if exp in {"3-1", "3-2", "3-3", "3-4"}:
                        param["mode"] = "fixed"

                        # acc_gate_sigma, gyro_gate_sigma, mag_gate_sigma
                        for key in ["acc_gate_sigma", "gyro_gate_sigma", "mag_gate_sigma"]:
                                m = re.search(rf"{key}=\s*([^\n]+)", block)
                                if not m:
                                        raise ValueError(f"{key} not found for exp {exp}")
                                param[key] = parse_scalar(m.group(1))
                        
                # 3-5, 3-6: time-varying sigma
                elif exp in {"3-5", "3-6"}:
                        param["mode"] = "time_varying"

                        # best_p, best_win_s
                        m = re.search(r"best_p=\s*([^,\n]+)\s*,\s*best_win_s=\s*([^\n]+)", block)
                        if not m:
                                raise ValueError(f"best_p / best_win_s not found for exp {exp}")
                        param["best_p"] = parse_scalar(m.group(1))
                        param["best_win_s"] = parse_scalar(m.group(2))

                        # best_update_ratio, best_ema_alpha
                        m = re.search(r"best_update_ratio=\s*([^,\n]+)\s*,\s*best_ema_alpha=\s*([^\n]+)", block)
                        if not m:
                                raise ValueError(f"best_update_ratio / best_ema_alpha not found for exp {exp}")
                        param["best_update_ratio"] = parse_scalar(m.group(1))
                        param["best_ema_alpha"] = parse_scalar(m.group(2))
                res[exp] = param
        return res

def run_exp3_from_param(exp: str, param: dict[str, Any],
                        q0: Quat, w: Vec3Batch, dt: ScalarBatch,
                        g0: float, g_world_unit: Vec3,
                        m0: float, m_ref_world_h_unit: Vec3,
                        a: Vec3Batch, m: Vec3Batch, sigma_floor: float,
                        q_ref: QuatBatch, g_ref: Vec3Batch, a_lin_ref: Vec3Batch) -> float:
        """
        returns:
                total_score
        """
        K_exp3: float = float(param["K"])
        mag_gain_exp3: float = float(param["mag_gain"])
        mag_err_sigma_exp3: float = float(param["mag_err_sigma"])

        if param["mode"] == "fixed":
                acc_gate_sigma_exp3: float = float(param["acc_gate_sigma"])
                gyro_gate_sigma_exp3: float = float(param["gyro_gate_sigma"])
                mag_gate_sigma_exp3: float = float(param["mag_gate_sigma"])

                [q_est_exp3, g_body_est_exp3, a_lin_est_exp3,
                 _, _, _] = integrate_gyro_acc_mag(q0, w, dt,
                                                   K_exp3, g0, g_world_unit,
                                                   m0, m_ref_world_h_unit, mag_gain_exp3,
                                                   acc_gate_sigma_exp3, gyro_gate_sigma_exp3,
                                                   mag_gate_sigma_exp3, mag_err_sigma_exp3,
                                                   a, m)

        elif param["mode"] == "time_varying":
                best_p_exp3: int = int(param["best_p"])
                best_win_s_exp3: float = float(param["best_win_s"])
                best_update_ratio_exp3: float = float(param["best_update_ratio"])
                best_ema_alpha_exp3: float = float(param["best_ema_alpha"])

                [timevarying_gyro_sigma_exp3,
                 timevarying_acc_sigma_exp3,
                 timevarying_mag_sigma_exp3] = suggest_timevarying_gate_sigma(
                                                w, a, m,
                                                dt, g0,
                                                p_gyro=best_p_exp3-10, p_acc=best_p_exp3, p_mag=best_p_exp3,
                                                sigma_floor=sigma_floor,
                                                win_s=best_win_s_exp3,
                                                update_s=best_win_s_exp3 * best_update_ratio_exp3,
                                                ema_alpha=best_ema_alpha_exp3)
                [q_est_exp3, g_body_est_exp3, a_lin_est_exp3,
                 _, _, _] = integrate_gyro_acc_mag(q0, w, dt,
                                                   K_exp3, g0, g_world_unit,
                                                   m0, m_ref_world_h_unit, mag_gain_exp3,
                                                   timevarying_acc_sigma_exp3, timevarying_gyro_sigma_exp3,
                                                   timevarying_mag_sigma_exp3, mag_err_sigma_exp3,
                                                   a, m)
        angle_err_exp3: ScalarBatch = calc_angle_err(q_est_exp3, q_ref)

        ori_score: float = score_angle_err(angle_err_exp3)
        g_score: float = score_angle_err(calc_vec3_direction_angle_err(g_body_est_exp3, g_ref))
        a_score: float = score_angle_err(calc_vec3_direction_angle_err(a_lin_est_exp3, a_lin_ref))
        total_score: float = ori_score + g_score + a_score
        print(f"exp {exp}: total_score={total_score:.7f} | ori={ori_score:.7f}, g={g_score:.7f}, a={a_score:.7f}")
        return total_score
