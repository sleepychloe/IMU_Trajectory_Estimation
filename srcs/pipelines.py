import numpy as np

from my_types import Vec3, Quat, ScalarBatch, Vec3Batch, QuatBatch
from my_types import as_quat, as_vec3_batch, as_quat_batch
import lib_quat as libq

EPS: float = 1e-9

def gyro_predict(q: Quat, w_avg: Vec3, dt: float) -> Quat:
        dq: Quat = libq.delta_quat_from_omega(w_avg, dt)
        q_pred: Quat = libq.quat_norm(libq.quat_mul(q, dq))
        return q_pred

def integrate_gyro(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch) -> QuatBatch:
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        
        for i in range(len(dt)):
                q: Quat = gyro_predict(q, w_avg[i], dt[i])
                res[i] = q
        return res

def safe_unit(v: Vec3) -> Vec3:
        norm: float = float(np.linalg.norm(v))
        return v / max(norm, EPS)

def predict_gravity_body_frame(q_pred: Quat, g_world_unit: Vec3) -> Vec3:
       g_pred: Vec3 = libq.rotate_world_to_body(q_pred, g_world_unit)
       return safe_unit(g_pred)

def small_angle_correction_quat(K_eff: float, e_axis: Vec3) -> Quat:
        dq_corr: Quat = as_quat(np.array([
                1,
                0.5 * K_eff * e_axis[0],
                0.5 * K_eff * e_axis[1],
                0.5 * K_eff * e_axis[2]]))
        return libq.quat_norm(dq_corr)

def integrate_gyro_grav(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                        K: float, g_world_unit: Vec3, g_meas_body: Vec3Batch,) -> QuatBatch:
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                g_meas: Vec3 = safe_unit(g_meas_body[i].copy())

                e_axis: Vec3 = np.cross(g_pred, g_meas)
                dq_corr: Quat = small_angle_correction_quat(K, e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q
        return res

def select_acc_measurement(a_src: Vec3, g_pred: Vec3) -> Vec3:
        a_meas_norm: float = float(np.linalg.norm(a_src))
        if a_meas_norm < EPS:
                return g_pred
        return a_src

def calc_acc_gating(g0: float, gate_sigma: float, a_meas: Vec3) -> float:
        # accel trust gating: if |a| deviates from g0, trust less
        dev: float = abs(float(np.linalg.norm(a_meas)) - g0)
        # w in [0,1]. 1 near static, 0 high linear acceleration
        weight_acc : float = np.exp(-0.5 * (dev / gate_sigma) ** 2)
        return weight_acc

def integrate_gyro_acc(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                       K: float, g0: float, g_world_unit: Vec3, gate_sigma: float, a_src: Vec3Batch
                       ) -> tuple[QuatBatch, Vec3Batch, Vec3Batch]:
        """
        Returns:
                res (q_gyro_acc): (N,4) QuatBatch
                g_body_est: (N,3) Vec3Batch
                a_lin_est: (N,3) Vec3Batch
        """
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        g_body_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        a_lin_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                a_meas: Vec3 = select_acc_measurement(a_src[i].copy(), g_pred.copy())
                a_unit: Vec3 = safe_unit(a_meas)

                weight_acc: float = calc_acc_gating(g0, gate_sigma, a_meas)
                e_axis: Vec3 = np.cross(g_pred, a_unit)
                dq_corr: Quat = small_angle_correction_quat(K * weight_acc, e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q

                g_body_est[i] = libq.rotate_world_to_body(q, g_world_unit) * g0
                a_lin_est[i] = a_src[i] + g_body_est[i]
        return res, g_body_est, a_lin_est