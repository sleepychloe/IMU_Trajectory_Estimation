import numpy as np

from my_types import Vec3, Quat, ScalarBatch, Vec3Batch, QuatBatch
from my_types import as_vec3, as_quat, as_scalar_batch, as_vec3_batch, as_quat_batch
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

def safe_unit_vec3(v: Vec3) -> Vec3:
        norm: float = float(np.linalg.norm(v))
        return v / max(norm, EPS)

def predict_gravity_body_frame(q_pred: Quat, g_world_unit: Vec3) -> Vec3:
       g_pred: Vec3 = libq.rotate_world_to_body(q_pred, g_world_unit)
       return safe_unit_vec3(g_pred)

def small_angle_correction_quat(K_eff: float, e_axis: Vec3) -> Quat:
        dq_corr: Quat = as_quat(np.array([
                1,
                0.5 * K_eff * e_axis[0],
                0.5 * K_eff * e_axis[1],
                0.5 * K_eff * e_axis[2]]))
        return libq.quat_norm(dq_corr)

def select_acc_measurement(a_src: Vec3, g_pred: Vec3) -> Vec3:
        a_meas_norm: float = float(np.linalg.norm(a_src))
        if a_meas_norm < EPS:
                return g_pred
        return a_src

def calc_acc_gating(g0: float, acc_sigma: float, a_meas: Vec3) -> float:
        if not np.isfinite(acc_sigma) or acc_sigma <= 0:
                return 1
        # accel trust gating: if |a| deviates from g0, trust less
        dev: float = abs(float(np.linalg.norm(a_meas)) - g0)
        # w in [0,1]. 1 near static, 0 high linear acceleration
        weight_acc : float = np.exp(-0.5 * (dev / acc_sigma) ** 2)
        return weight_acc

def calc_gyro_gating(gyro_sigma: float, w: Vec3) -> float:
        if not np.isfinite(gyro_sigma) or gyro_sigma <= 0:
                return 1
        w_norm: float = float(np.linalg.norm(w))
        weight_gyro: float = np.exp(-0.5 * (w_norm / gyro_sigma) ** 2)
        return weight_gyro

def integrate_gyro_acc(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                       K: float, g0: float, g_world_unit: Vec3,
                       acc_gate_sigma: float | ScalarBatch, gyro_gate_sigma: float | ScalarBatch, a_src: Vec3Batch
                       ) -> tuple[QuatBatch, Vec3Batch, Vec3Batch, ScalarBatch, ScalarBatch]:
        """
        Returns:
                res (q_gyro_acc): (N,4) QuatBatch
                g_body_est: (N,3) Vec3Batch
                a_lin_est: (N,3) Vec3Batch
                weight_acc: (N,) ScalarBatch
                weight_gyro: (N,) ScalarBatch
        """
        q: Quat = q0.copy()

        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        g_body_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        a_lin_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        weight_acc: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))
        weight_gyro: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))

        if not isinstance(acc_gate_sigma, np.ndarray):
                acc_gate_sigma: ScalarBatch = as_scalar_batch(np.full((len(dt),), acc_gate_sigma))
        if not isinstance(gyro_gate_sigma, np.ndarray):
                gyro_gate_sigma: ScalarBatch = as_scalar_batch(np.full((len(dt),), gyro_gate_sigma))

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                a_meas: Vec3 = select_acc_measurement(a_src[i].copy(), g_pred.copy())
                a_unit: Vec3 = safe_unit_vec3(a_meas)

                weight_acc[i] = calc_acc_gating(g0, acc_gate_sigma[i], a_meas)
                weight_gyro[i] = calc_gyro_gating(gyro_gate_sigma[i], w_avg[i])

                e_axis: Vec3 = np.cross(g_pred, a_unit)
                dq_corr: Quat = small_angle_correction_quat(K*(weight_acc[i]*weight_gyro[i]), e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q

                g_body_est[i] = libq.rotate_world_to_body(q, g_world_unit) * g0
                a_lin_est[i] = a_src[i] + g_body_est[i]
        return res, g_body_est, a_lin_est, weight_acc, weight_gyro

def generate_m_ref_world_h_unit(g0: float, g_world_unit: Vec3, sample_length: int,
                             q_ref: QuatBatch, a_src: Vec3Batch, m_src: Vec3Batch) -> Vec3:
        mask: ScalarBatch = as_scalar_batch(
                        (np.abs(np.linalg.norm(a_src, axis=1) - g0 < EPS)) & (np.linalg.norm(m_src, axis=1) > EPS))
        mask = mask[:sample_length]
        res: Vec3 = as_vec3(np.zeros(3))
        for i in range(sample_length):
                m_unit: Vec3 = safe_unit_vec3(m_src[i])
                m_world: Vec3 = libq.rotate_body_to_world(q_ref[i], m_unit)
                m_world_h: Vec3 = m_world - np.dot(m_world, g_world_unit) * g_world_unit
                n: float = np.linalg.norm(m_world_h)
                if n > EPS:
                        res += (m_world_h / n)
        return safe_unit_vec3(res)

def calc_mag_gating(m0: float, mag_sigma: float, m_meas: Vec3) -> float:
        if not np.isfinite(mag_sigma) or mag_sigma <= 0:
                return 1
        dev: float = abs(float(np.linalg.norm(m_meas)) - m0)
        weight_mag : float = np.exp(-0.5 * (dev / mag_sigma) ** 2)
        return weight_mag

def calc_mag_innovation_gating(e_axis_mag: Vec3, mag_err_sigma: float, threshold: float = 0.4) -> float:
        if not np.isfinite(mag_err_sigma) or mag_err_sigma <= 0:
                return 1
        e_axis_norm: float = float(np.linalg.norm(e_axis_mag))
        if e_axis_norm > threshold:
                return 0
        weight_mag_innov: float = np.exp(-0.5 * (e_axis_norm / mag_err_sigma) ** 2)
        return weight_mag_innov

def calc_mag_err_axis(q_pred: Quat, g_pred: Vec3, m_unit: Vec3, m_world_h_unit: Vec3) -> Vec3:
        m_body_h: Vec3 = m_unit - np.dot(m_unit, g_pred) * g_pred
        if np.linalg.norm(m_body_h) < 0.2:
                return as_vec3(np.array([0, 0, 0]))
        m_body_h_unit: Vec3 = safe_unit_vec3(m_body_h)

        m_pred: Vec3 = libq.rotate_world_to_body(q_pred, m_world_h_unit)
        m_pred_h_unit: Vec3 = safe_unit_vec3(m_pred - np.dot(m_pred, g_pred) * g_pred)

        mag_err_axis: Vec3 = np.cross(m_body_h_unit, m_pred_h_unit)
        mag_err_axis = np.dot(mag_err_axis, g_pred) * g_pred
        return mag_err_axis

def integrate_gyro_acc_mag(q0: Quat, w_avg: Vec3Batch, dt: ScalarBatch,
                       K: float, g0: float, g_world_unit: Vec3,
                       m0: float, m_world_h_unit: Vec3, mag_gain: float,
                       acc_gate_sigma: float | ScalarBatch, gyro_gate_sigma: float | ScalarBatch,
                       mag_gate_sigma: float | ScalarBatch, mag_err_sigma: float,
                       a_src: Vec3Batch, m_src: Vec3Batch
                       ) -> tuple[QuatBatch, Vec3Batch, Vec3Batch, ScalarBatch, ScalarBatch, ScalarBatch]:
        """
        Returns:
                res (q_gyro_acc_mag): (N,4) QuatBatch
                g_body_est: (N,3) Vec3Batch
                a_lin_est: (N,3) Vec3Batch
                weight_acc: (N,) ScalarBatch
                weight_gyro: (N,) ScalarBatch
                weight_mag: (N,) ScalarBatch
        """
        q: Quat = q0.copy()
        res: QuatBatch = as_quat_batch(np.zeros((len(dt), 4)))
        g_body_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        a_lin_est: Vec3Batch = as_vec3_batch(np.zeros((len(dt), 3)))
        weight_acc: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))
        weight_gyro: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))
        weight_mag: ScalarBatch = as_scalar_batch(np.zeros((len(dt),)))

        if not isinstance(acc_gate_sigma, np.ndarray):
                acc_gate_sigma: ScalarBatch = as_scalar_batch(np.full((len(dt),), acc_gate_sigma))
        if not isinstance(gyro_gate_sigma, np.ndarray):
                gyro_gate_sigma: ScalarBatch = as_scalar_batch(np.full((len(dt),), gyro_gate_sigma))
        if not isinstance(mag_gate_sigma, np.ndarray):
                mag_gate_sigma: ScalarBatch = as_scalar_batch(np.full((len(dt), ), mag_gate_sigma))

        for i in range(len(dt)):
                q_pred: Quat = gyro_predict(q, w_avg[i], dt[i])

                g_pred: Vec3 = predict_gravity_body_frame(q_pred, g_world_unit)
                a_meas: Vec3 = select_acc_measurement(a_src[i].copy(), g_pred.copy())
                a_unit: Vec3 = safe_unit_vec3(a_meas)
                m_meas: Vec3 = m_src[i].copy()
                m_unit: Vec3 = safe_unit_vec3(m_meas)

                weight_acc[i] = calc_acc_gating(g0, acc_gate_sigma[i], a_meas)
                weight_gyro[i] = calc_gyro_gating(gyro_gate_sigma[i], w_avg[i])
                weight_mag[i] = calc_mag_gating(m0, mag_gate_sigma[i], m_meas)

                e_axis_acc: Vec3 = np.cross(g_pred, a_unit)
                e_axis_mag: Vec3 = calc_mag_err_axis(q_pred, g_pred, m_unit, m_world_h_unit)

                weight_mag_innov: float = calc_mag_innovation_gating(e_axis_mag, mag_err_sigma)
                weight_mag[i] = weight_mag[i] * weight_mag_innov

                e_axis: Vec3 = weight_acc[i] * e_axis_acc + mag_gain * weight_mag[i] * e_axis_mag
                dq_corr: Quat = small_angle_correction_quat(K * weight_gyro[i], e_axis)

                q = libq.quat_norm(libq.quat_mul(q_pred, dq_corr))
                res[i] = q

                g_body_est[i] = libq.rotate_world_to_body(q, g_world_unit) * g0
                a_lin_est[i] = a_src[i] + g_body_est[i]
        return res, g_body_est, a_lin_est, weight_acc, weight_gyro, weight_mag
