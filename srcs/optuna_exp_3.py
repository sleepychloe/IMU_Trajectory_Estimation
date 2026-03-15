import numpy as np
from typing import Any, Callable
import optuna

from my_types import ScalarBatch, Vec3Batch, QuatBatch
from autotune import calc_score_quasi, calc_score_quasi_ori, suggest_timevarying_gate_sigma

def exp_3_1(dt: ScalarBatch, q_ref: QuatBatch,
            best_quasi_static: tuple[int, int, int],
            tau_candidate: tuple[float, float],
            mag_gain_candidate: tuple[float, float],
            runner_func: Callable[[float], tuple[Any, ...]],
            n_trials: int
            ) -> tuple[float, float, float]:
        """
        returns:
                best_tau, best_K, best_mag_gain
        """
        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                mag_gain: float = trial.suggest_float("mag_gain", mag_gain_candidate[0], mag_gain_candidate[1])

                score: float = calc_score_quasi_ori(tau, dt, q_ref,
                                                    runner_func,
                                                    best_quasi_static,
                                                    runner_kwargs={"mag_gain": mag_gain,
                                                                   "acc_gate_sigma": np.inf,
                                                                   "gyro_gate_sigma": np.inf,
                                                                   "mag_gate_sigma": np.inf,
                                                                   "mag_err_sigma": np.inf})
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params

        best_tau: float = best["tau"]
        dt_median: float = float(np.median(dt))
        best_K: float = float(dt_median / best_tau)
        best_mag_gain: float = best["mag_gain"]
        return best_tau, best_K, best_mag_gain

def exp_3_2(dt: ScalarBatch, q_ref: QuatBatch,
            best_quasi_static: tuple[int, int, int],
            tau_candidate: tuple[float, float],
            mag_gain_candidate: tuple[float, float],
            mag_err_sigma_candidate: tuple[float, float],
            runner_func: Callable[[float], tuple[Any, ...]],
            n_trials: int
            ) -> tuple[float, ...]:
        """
        returns:
                best_tau, best_K, best_mag_gain, best_mag_err_sigma
        """
        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                mag_gain: float = trial.suggest_float("mag_gain", mag_gain_candidate[0], mag_gain_candidate[1])
                mag_err_sigma: float = trial.suggest_float("mag_err_sigma", mag_err_sigma_candidate[0], mag_err_sigma_candidate[1])

                score: float = calc_score_quasi_ori(tau, dt, q_ref,
                                                    runner_func,
                                                    best_quasi_static,
                                                    runner_kwargs={"mag_gain": mag_gain,
                                                                   "acc_gate_sigma": np.inf,
                                                                   "gyro_gate_sigma": np.inf,
                                                                   "mag_gate_sigma": np.inf,
                                                                   "mag_err_sigma": mag_err_sigma})
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params

        best_tau: float = best["tau"]
        dt_median: float = float(np.median(dt))
        best_K: float = float(dt_median / best_tau)
        best_mag_gain: float = best["mag_gain"]
        best_mag_err_sigma: float = best["mag_err_sigma"]
        return best_tau, best_K, best_mag_gain, best_mag_err_sigma

def exp_3_3(dt: ScalarBatch, q_ref: QuatBatch,
            best_quasi_static: tuple[int, int, int],
            tau_candidate: tuple[float, float],
            mag_gain_candidate: tuple[float, float],
            acc_gate_sigma_candidate: tuple[float, float],
            gyro_gate_sigma_candidate: tuple[float, float],
            mag_gate_sigma_candidate: tuple[float, float],
            runner_func: Callable[[float], tuple[Any, ...]],
            n_trials: int
            ) -> tuple[float, ...]:
        """
        returns:
                best_tau, best_K, best_mag_gain,
                best_acc_sigma, best_gyro_sigma, best_mag_sigma
        """
        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                mag_gain: float = trial.suggest_float("mag_gain", mag_gain_candidate[0], mag_gain_candidate[1])
                acc_gate_sigma: float = trial.suggest_float("acc_gate_sigma", acc_gate_sigma_candidate[0], acc_gate_sigma_candidate[1])
                gyro_gate_sigma: float = trial.suggest_float("gyro_gate_sigma", gyro_gate_sigma_candidate[0], gyro_gate_sigma_candidate[1])
                mag_gate_sigma: float = trial.suggest_float("mag_gate_sigma", mag_gate_sigma_candidate[0], mag_gate_sigma_candidate[1])

                score: float = calc_score_quasi_ori(tau, dt, q_ref,
                                                    runner_func,
                                                    best_quasi_static,
                                                    runner_kwargs={"mag_gain": mag_gain,
                                                                   "acc_gate_sigma": acc_gate_sigma,
                                                                   "gyro_gate_sigma": gyro_gate_sigma,
                                                                   "mag_gate_sigma": mag_gate_sigma,
                                                                   "mag_err_sigma": np.inf})
                
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params

        best_tau: float = best["tau"]
        dt_median: float = float(np.median(dt))
        best_K: float = float(dt_median / best_tau)
        best_mag_gain: float = best["mag_gain"]
        best_acc_sigma: float = best["acc_gate_sigma"]
        best_gyro_sigma: float = best["gyro_gate_sigma"]
        best_mag_sigma: float = best["mag_gate_sigma"]
        return best_tau, best_K, best_mag_gain, best_acc_sigma, best_gyro_sigma, best_mag_sigma

def exp_3_4(dt: ScalarBatch, q_ref: QuatBatch,
            best_quasi_static: tuple[int, int, int],
            tau_candidate: tuple[float, float],
            mag_gain_candidate: tuple[float, float],
            acc_gate_sigma_candidate: tuple[float, float],
            gyro_gate_sigma_candidate: tuple[float, float],
            mag_gate_sigma_candidate: tuple[float, float],
            mag_err_sigma_candidate: tuple[float, float],
            runner_func: Callable[[float], tuple[Any, ...]],
            n_trials: int
            ) -> tuple[float, ...]:
        """
        returns:
                best_tau, best_K, best_mag_gain,
                best_acc_sigma, best_gyro_sigma, best_mag_sigma, best_mag_err_sigma
        """
        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                mag_gain: float = trial.suggest_float("mag_gain", mag_gain_candidate[0], mag_gain_candidate[1])
                acc_gate_sigma: float = trial.suggest_float("acc_gate_sigma", acc_gate_sigma_candidate[0], acc_gate_sigma_candidate[1])
                gyro_gate_sigma: float = trial.suggest_float("gyro_gate_sigma", gyro_gate_sigma_candidate[0], gyro_gate_sigma_candidate[1])
                mag_gate_sigma: float = trial.suggest_float("mag_gate_sigma", mag_gate_sigma_candidate[0], mag_gate_sigma_candidate[1])
                mag_err_sigma: float = trial.suggest_float("mag_err_sigma", mag_err_sigma_candidate[0], mag_err_sigma_candidate[1])

                score: float = calc_score_quasi_ori(tau, dt, q_ref,
                                                    runner_func,
                                                    best_quasi_static,
                                                    runner_kwargs={"mag_gain": mag_gain,
                                                                   "acc_gate_sigma": acc_gate_sigma,
                                                                   "gyro_gate_sigma": gyro_gate_sigma,
                                                                   "mag_gate_sigma": mag_gate_sigma,
                                                                   "mag_err_sigma": mag_err_sigma})
                
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params

        best_tau: float = best["tau"]
        dt_median: float = float(np.median(dt))
        best_K: float = float(dt_median / best_tau)
        best_mag_gain: float = best["mag_gain"]
        best_acc_sigma: float = best["acc_gate_sigma"]
        best_gyro_sigma: float = best["gyro_gate_sigma"]
        best_mag_sigma: float = best["mag_gate_sigma"]
        best_mag_err_sigma: float = best["mag_err_sigma"]
        return best_tau, best_K, best_mag_gain, best_acc_sigma, best_gyro_sigma, best_mag_sigma, best_mag_err_sigma

def exp_3_5(dt: ScalarBatch, q_ref: QuatBatch,
            g0: float, sigma_floor: float,
            w: Vec3Batch, a: Vec3Batch, m: Vec3Batch,
            best_quasi_static: tuple[int, int, int],
            tau_candidate: tuple[float, float],
            mag_gain_candidate: tuple[float, float],
            p_candidate: list[int],
            win_s_candidate: list[float],
            update_ratio_candidate: list[float],
            ema_candidate: list[float],
            runner_func: Callable[[float], tuple[Any, ...]],
            n_trials: int
            ) -> tuple[Any, ...]:
        """
        returns:
                best_tau, best_K, best_mag_gain,
                timevarying_acc_sigma, timevarying_gyro_sigma, timevarying_mag_sigma
        """
        dt_median: float = float(np.median(dt))

        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                mag_gain: float = trial.suggest_float("mag_gain", mag_gain_candidate[0], mag_gain_candidate[1])
                p: int = trial.suggest_int("p", p_candidate[0], p_candidate[1])
                win: float = trial.suggest_float("win_s", win_s_candidate[0], win_s_candidate[1])
                update_ratio: float = trial.suggest_float("update_ratio", update_ratio_candidate[0], update_ratio_candidate[1])
                update: float = win * update_ratio
                ema: float = trial.suggest_float("ema_alpha", ema_candidate[0], ema_candidate[1])

                [timevarying_gyro_sigma,
                 timevarying_acc_sigma,
                 timevarying_mag_sigma] = suggest_timevarying_gate_sigma(
                                                w=w, a=a, m=m,
                                                dt=dt, g0=g0,
                                                p_gyro=p-10, p_acc=p, p_mag=p, sigma_floor=sigma_floor,
                                                win_s=win, update_s=update, ema_alpha=ema)

                score: float = calc_score_quasi_ori(tau, dt, q_ref,
                                                    runner_func,
                                                    best_quasi_static,
                                                    runner_kwargs={"mag_gain": mag_gain,
                                                                   "acc_gate_sigma": timevarying_acc_sigma,
                                                                   "gyro_gate_sigma": timevarying_gyro_sigma,
                                                                   "mag_gate_sigma": timevarying_mag_sigma,
                                                                   "mag_err_sigma": np.inf})
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params

        best_tau: float = best["tau"]
        best_K: float = float(dt_median / best_tau)
        best_mag_gain: float = best["mag_gain"]
        best_p: int = best["p"]
        best_win: float = best["win_s"]
        best_update: float = best_win * best["update_ratio"]
        best_ema: float = best["ema_alpha"]
        [timevarying_gyro_sigma,
         timevarying_acc_sigma,
         timevarying_mag_sigma] = suggest_timevarying_gate_sigma(
                                        w=w, a=a, m=m,
                                        dt=dt, g0=g0,
                                        p_gyro=best_p-10, p_acc=best_p, p_mag=best_p, sigma_floor=sigma_floor,
                                        win_s=best_win, update_s=best_update, ema_alpha=best_ema)
        return [best_tau, best_K, best_mag_gain,
                timevarying_acc_sigma, timevarying_gyro_sigma, timevarying_mag_sigma]

def exp_3_6(dt: ScalarBatch, q_ref: QuatBatch,
            g0: float, sigma_floor: float,
            w: Vec3Batch, a: Vec3Batch, m: Vec3Batch,
            best_quasi_static: tuple[int, int, int],
            tau_candidate: tuple[float, float],
            mag_gain_candidate: tuple[float, float],
            p_candidate: list[int],
            win_s_candidate: list[float],
            update_ratio_candidate: list[float],
            ema_candidate: list[float],
            mag_err_sigma_candidate: tuple[float, float],
            runner_func: Callable[[float], tuple[Any, ...]],
            n_trials: int
            ) -> tuple[Any, ...]:
        """
        returns:
                best_tau, best_K, best_mag_gain,
                timevarying_acc_sigma, timevarying_gyro_sigma, timevarying_mag_sigma, best_mag_err_sigma
        """
        dt_median: float = float(np.median(dt))

        def objective(trial):
                tau: float = trial.suggest_float("tau", tau_candidate[0], tau_candidate[1])
                mag_gain: float = trial.suggest_float("mag_gain", mag_gain_candidate[0], mag_gain_candidate[1])
                mag_err_sigma: float = trial.suggest_float("mag_err_sigma", mag_err_sigma_candidate[0], mag_err_sigma_candidate[1])
                p: int = trial.suggest_int("p", p_candidate[0], p_candidate[1])
                win: float = trial.suggest_float("win_s", win_s_candidate[0], win_s_candidate[1])
                update_ratio: float = trial.suggest_float("update_ratio", update_ratio_candidate[0], update_ratio_candidate[1])
                update: float = win * update_ratio
                ema: float = trial.suggest_float("ema_alpha", ema_candidate[0], ema_candidate[1])

                [timevarying_gyro_sigma,
                 timevarying_acc_sigma,
                 timevarying_mag_sigma] = suggest_timevarying_gate_sigma(
                                                w=w, a=a, m=m,
                                                dt=dt, g0=g0,
                                                p_gyro=p-10, p_acc=p, p_mag=p, sigma_floor=sigma_floor,
                                                win_s=win, update_s=update, ema_alpha=ema)

                score: float = calc_score_quasi_ori(tau, dt, q_ref,
                                                    runner_func,
                                                    best_quasi_static,
                                                    runner_kwargs={"mag_gain": mag_gain,
                                                                   "acc_gate_sigma": timevarying_acc_sigma,
                                                                   "gyro_gate_sigma": timevarying_gyro_sigma,
                                                                   "mag_gate_sigma": timevarying_mag_sigma,
                                                                   "mag_err_sigma": mag_err_sigma})
                return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params

        best_tau: float = best["tau"]
        best_K: float = float(dt_median / best_tau)
        best_mag_gain: float = best["mag_gain"]
        best_p: int = best["p"]
        best_win: float = best["win_s"]
        best_update: float = best_win * best["update_ratio"]
        best_ema: float = best["ema_alpha"]
        [timevarying_gyro_sigma,
         timevarying_acc_sigma,
         timevarying_mag_sigma] = suggest_timevarying_gate_sigma(
                                        w=w, a=a, m=m,
                                        dt=dt, g0=g0,
                                        p_gyro=best_p-10, p_acc=best_p, p_mag=best_p, sigma_floor=sigma_floor,
                                        win_s=best_win, update_s=best_update, ema_alpha=best_ema)
        best_mag_err_sigma: float = best["mag_err_sigma"]
        return [best_tau, best_K, best_mag_gain,
                timevarying_acc_sigma, timevarying_gyro_sigma, timevarying_mag_sigma, best_mag_err_sigma]
