import os
os.environ['GEOMSTATS_BACKEND'] = 'numpy'  # NOQA
import sys
import matplotlib.pyplot as plt
import numpy as np

from functools import reduce, wraps

import geomstats.backend as gs
from geomstats import algebra_utils
from geomstats.geometry.special_euclidean import SpecialEuclidean

from scipy.linalg import block_diag

def _block_diag(arrs):
    return block_diag(*arrs)

# vectorized_block_diag = np.vectorize(block_diag, signature='(n,m),(p,q)->(r,l)')
# vectorized_block_diag = np.vectorize(_block_diag, signature='(n,m,p,q)->(n,r,l)')
def vectorized_block_diag(*arrs):
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_3d(a) for a in arrs]

    tensor_depth = arrs[0].shape[0]
    bad_args = [k for k in range(len(arrs)) if (
            arrs[k].ndim > 3 or arrs[k].shape[0] != tensor_depth)]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 3: %s" % bad_args)

    block_shapes = np.array([a.shape[1:] for a in arrs])
    out_matrix_shape = np.sum(block_shapes, axis=0)
    out_tensor_shape = np.hstack((tensor_depth, out_matrix_shape))
    out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
    out = np.zeros(out_tensor_shape, dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(block_shapes):
        out[:, r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def vectorize(otypes=None, signature=None):
    """Numpy vectorization wrapper that works with instance methods."""
    def decorator(fn):
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)
        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)
        return wrapper
    return decorator


class Localization:
    group = SpecialEuclidean(2, 'vector')

    @staticmethod
    def split_state(state):
        if gs.ndim(state) == 1:
            return state
        return state[:, 0], state[:, 1], state[:, 2]

    @staticmethod
    def split_input(input):
        if gs.ndim(input) == 1:
            return input[0], input[1:3], input[3]
        return input[:, 0], input[:, 1:3], input[:, 3]

    @staticmethod
    def rotation_matrix(theta):
        if gs.ndim(gs.array(theta)) <= 1:
            theta = gs.array([theta])
        return Localization.group.rotations.matrix_from_rotation_vector(theta.T)

    @staticmethod
    def regularize_angle(theta):
        if gs.ndim(gs.array(theta)) <= 1:
            theta = gs.array([theta])
        return Localization.group.rotations.log_from_identity(theta.T)

    @staticmethod
    def angle_difference(theta_1, theta_2):
        ''' Distance function used to compute heading RMSE '''
        if gs.ndim(gs.array(theta_2)) < gs.ndim(gs.array(theta_1)):
            theta_2 = gs.tile(theta_2, theta_1.shape[0])
        angle_concat = gs.array([theta_1, theta_2])
        if gs.ndim(angle_concat) > 1:
            theta = np.max(angle_concat, axis=0)
            alpha = np.min(angle_concat, axis=0)
            return np.min([theta - alpha, alpha + 2 * gs.pi - theta], axis=0)

        theta = np.max(angle_concat)
        alpha = np.min(angle_concat)

        return Localization.regularize_angle(np.min([theta - alpha, alpha + 2 * gs.pi - theta]))

    @staticmethod
    @vectorize(signature='(n),()->(n)')
    def angle_difference_vectorized(theta_1, theta_2):
        return Localization.angle_difference(theta_1, theta_2)

    def __init__(self, nb_gps=1):
        self.group = SpecialEuclidean(2, 'vector')
        self.dim = self.group.dim
        self.dim_noise = 3
        self.nb_gps = nb_gps
        self.dim_obs = 2 * nb_gps
        self.obs_corruption_mode = 'None'
        self.input_corruption_mode = 'None'

    def ad_chi(self, state):
        ''' Returns the tangent map associated to Ad_X : g |-> XgX^-1'''
        theta, x, y = self.split_state(state)
        J = gs.array([[0, -1],
                      [1, 0]])
        ad = gs.eye(3)
        if gs.ndim(state) > 1:
            n_states = state.shape[0]
            J = gs.array([J] * n_states)
            ad = gs.array([ad] * n_states)
            rot = self.rotation_matrix(theta)
            ad[:, 1:, 1:] = rot
            ad[:, 1:, 0] = -gs.einsum('ijk, ik -> ij', J, state[:, 1:])
            return ad
        ad[1:, 1:] = self.rotation_matrix(theta)
        ad[1:, 0] = -J.dot([x, y])

        return ad

    def propagate(self, state, input):
        """ Standard constant velocity motion model on SE(2),
        with a bidimensional velocity. """
        dt, linear_speed, angular_speed = self.split_input(input)
        theta, x, y = self.split_state(state)
        if gs.ndim(state) > 1:
            dt = dt[0]
            rot = self.rotation_matrix(theta)
            pos_incr = gs.einsum('ijk, ik -> ij', rot, linear_speed)
            new_pos = state[:, 1:] + dt * pos_incr
            theta = theta + dt * angular_speed
            theta = self.regularize_angle(theta)
            return np.column_stack((theta, new_pos))
        x, y = state[1:] + dt * self.rotation_matrix(theta).dot(
            linear_speed)
        theta = theta + dt * angular_speed
        theta = self.regularize_angle(theta)
        return gs.concatenate((theta, [x, y]))

    def update(self, state, vector):
        new_state = self.group.exp(vector, state)
        if gs.ndim(state) == 1 and gs.ndim(new_state) > 1:
            return gs.squeeze(new_state)
        return new_state

    def propagation_jacobian(self, state, input):
        ''' Returns the jacobian associated to the variable i for the propagation factor
            f(X_i)^{-1} X_{i+1}, where f depends on the input u.
            More precisely, it is the jacobian of the tangent map at the identity
            of g(X) = f(I)^{-1} f(X) '''
        dt, linear_speed, angular_speed = self.split_input(input)
        if gs.ndim(state) > 1:
            dt = dt[0]
        input_vector_form = dt*gs.hstack((gs.array([angular_speed]).T, linear_speed))
        input_inv = self.group.inverse(input_vector_form)

        return self.ad_chi(input_inv)

    def noise_jacobian(self, state, input):
        dt, _, _ = self.split_input(input)
        if gs.ndim(input) > 1:
            n_inputs = input.shape[0]
            return gs.sqrt(dt[0]) * gs.array([gs.eye(self.dim_noise)] * n_inputs)
        return gs.sqrt(dt) * gs.eye(self.dim_noise)
        # return dt * gs.eye(self.dim_noise)

    def observation_jacobian(self, state, observation):
        jac = gs.tile(gs.eye(2, 3, 1), (nb_gps, 1))
        if gs.ndim(state) > 1:
            n_states = state.shape[0]
            return gs.array([jac] * n_states)
        return jac
        # return self.rotation_matrix(state[0]).dot(gs.eye(2, 3, 1))

    def get_measurement_noise_cov(self, state, observation_cov):
        if self.nb_gps > 1:
            return self._get_expanded_measurement_noise_cov(
                state, observation_cov)

        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        if gs.ndim(state) > 1:
            return gs.einsum('ijk, ikl, ilm -> ijm', rot, observation_cov,
                             rot.transpose(0, 2, 1))
        return rot.T.dot(observation_cov).dot(rot)
        # return observation_cov

    def _get_expanded_measurement_noise_cov(self, state, observation_cov):
        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        if gs.ndim(state) > 1:
            rot_expanded = vectorized_block_diag(*[rot for _ in range(self.nb_gps)])
            # rot_expanded = vectorized_block_diag([rot for _ in range(self.nb_gps)])
            return gs.einsum('ijk, ikl, ilm -> ijm', rot_expanded, observation_cov,
                             rot_expanded.transpose(0, 2, 1))
        rot_expanded = block_diag(*[rot for _ in range(self.nb_gps)])
        return rot_expanded.T.dot(observation_cov).dot(rot_expanded)
        # return observation_cov

    def innovation(self, state, observation):
        if self.nb_gps > 1:
            return self._expanded_innovation(state, observation)

        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        nb_measurement = self.dim_obs // 2
        if gs.ndim(state) > 1:
            expected = state[:, 1:]
            return gs.einsum('ijk, ik -> ij', rot.transpose(0, 2, 1),
                             observation - expected)
            # return observation - expected
        expected = state[1:]
        return rot.T.dot(observation - expected)
        # return observation - expected

    def _expanded_innovation(self, state, observation):
        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        # nb_measurement = self.dim_obs // 2
        if gs.ndim(state) > 1:
            rot_expanded = vectorized_block_diag(
                *[rot for _ in range(self.nb_gps)])
            # rot_expanded = vectorized_block_diag(
            #     [rot for _ in range(self.nb_gps)])
            expected = gs.tile(state[:, 1:], nb_gps)
            return gs.einsum('ijk, ik -> ij', rot_expanded.transpose(0, 2, 1),
                             observation - expected)
            # return observation - expected
        rot_expanded = block_diag(*[rot for _ in range(self.nb_gps)])
        expected = gs.tile(state[1:], self.nb_gps)
        return rot_expanded.T.dot(observation - expected)
        # return observation - expected

    def get_constraint(self, state, mode=None):
        if mode is None:
            mode = self.obs_corruption_mode
        if mode == 'distance':
            if gs.ndim(state) > 1:
                n_states = state.shape[0]
                rot = self.rotation_matrix(state[:, 0])
                constraint_direction = gs.einsum('ikj, ik -> ij', rot,
                                                 state[:, 1:]).reshape(n_states, 2, 1)
                other_gps = gs.zeros((n_states, 2 * (self.nb_gps - 1), 1))
                constraint_direction = gs.concatenate((constraint_direction, other_gps), axis=1)
                constrained_value = gs.zeros((state.shape[0], self.dim, 1))
                return constraint_direction, constrained_value
            rot = self.rotation_matrix(state[0])
            constraint_direction = (rot.T.dot(state[1:])).reshape(2, 1)
            other_gps = gs.zeros((2 * (self.nb_gps - 1), 1))
            constraint_direction = gs.vstack((constraint_direction, other_gps))
            constrained_value = gs.zeros((self.dim, 1))
        elif mode == 'angle':
            tangent_base = gs.array([[0, -1],
                                     [1, 0]])
            if gs.ndim(state) > 1:
                n_states = state.shape[0]
                rot = self.rotation_matrix(state[:, 0])
                rotated_state = gs.einsum('ikj, ik -> ij', rot,
                                                 state[:, 1:]).reshape(n_states, 2, 1)
                constraint_direction = gs.matmul(tangent_base, rotated_state)
                other_gps = gs.zeros((n_states, 2 * (self.nb_gps - 1), 1))
                constraint_direction = gs.concatenate(
                    (constraint_direction, other_gps), axis=1)
                constrained_value = gs.zeros((state.shape[0], self.dim, 1))
                return constraint_direction, constrained_value
            rot = self.rotation_matrix(state[0])
            constraint_direction = (rot.T.dot(tangent_base.dot(state[1:]))).reshape(2, 1)
            other_gps = gs.zeros((2 * (self.nb_gps - 1), 1))
            constraint_direction = gs.vstack((constraint_direction, other_gps))
            # constraint_direction = tangent_base.dot(state[1:]).reshape(2, 1)
            constrained_value = gs.zeros((self.dim, 1))
        elif mode == 'both':
            distance_dir, _ = self.get_constraint(state, mode='distance')
            angle_dir, _ = self.get_constraint(state, mode='angle')
            if gs.ndim(state) > 1:
                n_states = state.shape[0]
                angle_dir = gs.concatenate((gs.zeros(
                    (n_states, 2, 1)), angle_dir[:, :2], gs.zeros((n_states, 2 * (self.nb_gps - 2), 1))), axis=1)
                constraint_direction = gs.concatenate(
                    (distance_dir, angle_dir), axis=2)
            else:
                angle_dir = gs.vstack((gs.zeros((2, 1)), angle_dir[:2], gs.zeros((2 * (self.nb_gps - 2), 1))))
                constraint_direction = gs.hstack((distance_dir, angle_dir))
            constrained_value = gs.zeros((self.dim, 2))
        else:
            raise ValueError('Constraint not implemented')

        return constraint_direction, constrained_value

    def set_corruption_modes(self, obs_mode, input_mode):
        if obs_mode in ['distance', 'angle', 'both']:
            self.obs_corruption_mode = obs_mode
        else:
            raise ValueError('Constraint not implemented')
        if input_mode in ['rotation']:
            self.input_corruption_mode = input_mode
        else:
            raise ValueError('Constraint not implemented')

    def corrupt_measure(self, observation, value, mode=None):
        corrupted_obs = 1 * observation
        if mode is None:
            mode = self.obs_corruption_mode
        if mode == 'None':
            pass
        elif mode == 'distance':
            corrupted_obs[:2] = value * corrupted_obs[:2]
        elif mode == 'angle':
            corrupted_obs[:2] = self.rotation_matrix(- value).dot(corrupted_obs[:2])
        elif mode == 'both':
            distance_part = self.corrupt_measure(observation[:2], value[0], 'distance')
            angle_part = self.corrupt_measure(observation[2:], value[1], 'angle')
            corrupted_obs = gs.hstack((distance_part, angle_part))
        else:
            raise ValueError('Constraint not implemented')
        return corrupted_obs

    def corrupt_input(self, input, value, mode=None):
        dt, linear_vel, angular_vel = self.split_input(input)
        corrupted_lin_vel = 1 * linear_vel
        corrupted_ang_vel = 1 * angular_vel
        if mode is None:
            mode = self.input_corruption_mode
        if mode == 'None':
            pass
        elif mode == 'rotation':
            corrupted_lin_vel = self.rotation_matrix(value).dot(corrupted_lin_vel)
        else:
            raise ValueError('Constraint not implemented')
        return gs.concatenate(([dt], corrupted_lin_vel, [corrupted_ang_vel]))

class LocalizationEKF(Localization):

    def update(self, state, vector):
        new_state = gs.zeros_like(state)
        if gs.ndim(new_state) > 1:
            new_rot = self.regularize_angle(state[:, 0] + vector[:, 0])
            new_pos = state[:, 1:] + vector[:, 1:]
            new_state = gs.hstack((new_rot, new_pos))
            return new_state
        new_state[0] = self.regularize_angle(state[0] + vector[0])
        new_state[1:] = state[1:] + vector[1:]
        return new_state

    def propagation_jacobian(self, state, input):
        ''' Returns the jacobian associated to the variable i for the propagation factor
            f(X_i)^{-1} X_{i+1}, where f depends on the input u.
            More precisely, it is the jacobian of the tangent map at the identity
            of g(X) = f(I)^{-1} f(X) '''
        dt, linear_speed, angular_speed = self.split_input(input)
        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        tangent_rotation = gs.array([[0, -1],
                                     [1, 0]])
        if gs.ndim(state) > 1:
            n_state = state.shape[0]
            dt = dt[0]
            ortho_linear_speed = gs.einsum('jk, ik -> ij', tangent_rotation, linear_speed)
            # ortho_linear_speed = gs.array([-linear_speed[:, 1], linear_speed[:, 0]])
            jacobian = gs.array([gs.eye(self.dim)] * n_state)
            # jacobian[:, 1:, 0] = dt*gs.matmul(rot, ortho_linear_speed)
            jacobian[:, 1:, 0] = dt*gs.einsum('ijk, ik -> ij', rot, ortho_linear_speed)
            return jacobian
        # ortho_linear_speed = [-linear_speed[1], linear_speed[0]]
        ortho_linear_speed = tangent_rotation.dot(linear_speed)
        jacobian = gs.eye(self.dim)
        jacobian[1:, 0] = dt*rot.dot(ortho_linear_speed)

        return jacobian

    def noise_jacobian(self, state, input):
        dt, _, _ = self.split_input(input)
        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        # rot = gs.eye(2)
        if gs.ndim(state) > 1:
            dt = dt[0]
            n_state = state.shape[0]
            speed_jacobian = vectorized_block_diag(
                gs.ones((n_state, 1, 1)), rot)
            return gs.sqrt(dt) * speed_jacobian
        speed_jacobian = block_diag(1, rot)
        return gs.sqrt(dt) * speed_jacobian

    def get_measurement_noise_cov(self, state, observation_cov):
        return observation_cov

    def innovation(self, state, observation):
        if gs.ndim(state) > 1:
            expected = gs.tile(state[:, 1:], self.nb_gps)
            return observation - expected
        expected = gs.tile(state[1:], self.nb_gps)
        return observation - expected

    def get_constraint(self, state, mode=None):
        if mode is None:
            mode = self.obs_corruption_mode
        if mode == 'distance':
            if gs.ndim(state) > 1:
                n_states = state.shape[0]
                constraint_direction = state[:, 1:].reshape(
                    n_states, 2, 1)
                other_gps = gs.zeros((n_states, 2 * (self.nb_gps - 1), 1))
                constraint_direction = gs.concatenate(
                    (constraint_direction, other_gps), axis=1)
                constrained_value = gs.zeros((state.shape[0], self.dim, 1))
                return constraint_direction, constrained_value
            constraint_direction = state[1:].reshape(2, 1)
            other_gps = gs.zeros((2 * (self.nb_gps - 1), 1))
            constraint_direction = gs.vstack((constraint_direction, other_gps))
            constrained_value = gs.zeros((self.dim, 1))
        elif mode == 'angle':
            tangent_base = gs.array([[0, -1],
                                       [1, 0]])
            if gs.ndim(state) > 1:
                n_states = state.shape[0]
                constraint_direction = gs.matmul(tangent_base, state[:, 1:].reshape(n_states, 2, 1))
                other_gps = gs.zeros((n_states, 2 * (self.nb_gps - 1), 1))
                constraint_direction = gs.concatenate(
                    (constraint_direction, other_gps), axis=1)
                constrained_value = gs.zeros((state.shape[0], self.dim, 1))
                return constraint_direction, constrained_value
            constraint_direction = (
                tangent_base.dot(state[1:])).reshape(2, 1)
            other_gps = gs.zeros((2 * (self.nb_gps - 1), 1))
            constraint_direction = gs.vstack((constraint_direction, other_gps))
            constrained_value = gs.zeros((self.dim, 1))
        elif mode == 'both':
            distance_dir, _ = self.get_constraint(state, mode='distance')
            angle_dir, _ = self.get_constraint(state, mode='angle')
            if gs.ndim(state) > 1:
                n_states = state.shape[0]
                angle_dir = gs.concatenate((gs.zeros(
                    (n_states, 2, 1)), angle_dir[:, :2], gs.zeros(
                    (n_states, 2 * (self.nb_gps - 2), 1))), axis=1)
                constraint_direction = gs.concatenate(
                    (distance_dir, angle_dir), axis=2)
            else:
                angle_dir = gs.vstack((gs.zeros((2, 1)), angle_dir[:2],
                                       gs.zeros((2 * (self.nb_gps - 2), 1))))
                constraint_direction = gs.hstack((distance_dir, angle_dir))
            constrained_value = gs.zeros((self.dim, 2))
        else:
            raise ValueError('Constraint not implemented')

        return constraint_direction, constrained_value


class KalmanFilter:

    def __init__(self, model):
        self.model = model
        self.state = model.group.get_identity()
        self.covariance = gs.zeros((self.model.dim, self.model.dim))
        self.process_noise = gs.zeros(
            (self.model.dim_noise, self.model.dim_noise))
        self.measurement_noise = gs.zeros(
            (self.model.dim_obs, self.model.dim_obs))

    def _vectorize_state(self, n_states):
        self.state = gs.array([self.state] * n_states)

    def _vectorize_covs(self, n_states):
        self.covariance = gs.array([self.covariance] * n_states)
        self.process_noise = gs.array([self.process_noise] * n_states)
        self.measurement_noise = gs.array([self.measurement_noise] * n_states)

    def vectorize(self, n_states):
        self._vectorize_state(n_states)
        self._vectorize_covs(n_states)

    def initialise_covariances(self, prior_values, process_values, obs_values):
        values = [prior_values, process_values, obs_values]
        attributes = ['covariance', 'process_noise', 'measurement_noise']
        for (index, val) in enumerate(values):
            if gs.ndim(val) == 1:
                setattr(self, attributes[index],
                        algebra_utils.from_vector_to_diagonal_matrix(val))
            else:
                setattr(self, attributes[index], val)
        if gs.ndim(self.state) > 1:
            self._vectorize_covs(self.state.shape[0])

    def set_rotation(self, new_rot):
        if gs.ndim(self.state) > 1:
            # self.state[:, 0] = new_rot
            if gs.ndim(gs.array(new_rot)) == 0:
                n_states = self.state.shape[0]
                new_rot = [new_rot] * n_states
            self.state = gs.hstack((new_rot, self.state[:, 1:]))
        else:
            self.state = gs.concatenate(([new_rot], self.state[1:]))

    def set_position(self, new_pos):
        if gs.ndim(self.state) > 1:
            # self.state[:, 0] = new_rot
            if gs.ndim(gs.array(new_pos)) == 1:
                n_states = self.state.shape[0]
                new_pos = [new_pos] * n_states
            self.state = gs.hstack((self.state[:, :1], new_pos))
        else:
            self.state = gs.concatenate((self.state[:1], new_pos))

    def propagate(self, input):
        Q = self.process_noise
        F = self.model.propagation_jacobian(self.state, input)
        # G = self.model.noise_jacobian(self.model.propagate(self.state, input), input)
        G = self.model.noise_jacobian(self.state, input)
        P = self.covariance
        if gs.ndim(self.state) > 1:
            prop_cov = gs.einsum('ijk, ikl, ilm -> ijm', F, P,
                                 F.transpose(0, 2, 1))
            noise_cov = gs.einsum('ijk, ikl, ilm -> ijm', G, Q,
                                  G.transpose(0, 2, 1))
            self.covariance = prop_cov + noise_cov
        else:
            self.covariance = F.dot(P).dot(F.T) + G.dot(Q).dot(G.T)
        self.state = self.model.propagate(self.state, input)

    def compute_gain(self, observation):
        N = self.model.get_measurement_noise_cov(
            self.state, self.measurement_noise)
        H = self.model.observation_jacobian(self.state, observation)
        if gs.ndim(self.state) > 1:
            estimate_cov = gs.einsum(
                'ijk, ikl, ilm -> ijm', H, self.covariance, H.transpose(0, 2, 1))
            innovation_info = gs.linalg.inv(estimate_cov + N)
            return gs.einsum('ijk, ikl, ilm -> ijm', self.covariance,
                             H.transpose(0, 2, 1), innovation_info)
        innovation_cov = H.dot(self.covariance).dot(H.T) + N
        return self.covariance.dot(H.T).dot(gs.linalg.inv(innovation_cov))

    def update(self, observation):
        innovation = self.model.innovation(self.state, observation)
        gain = self.compute_gain(observation)
        H = self.model.observation_jacobian(self.state, observation)
        if gs.ndim(self.state) > 1:
            n_states = self.state.shape[0]
            gain_factor = gs.einsum('ijk, ikl -> ijl', gain, H)
            id_factor = gs.array([gs.eye(self.model.dim)] * n_states)
            self.covariance = gs.einsum(
                'ijk, ikl -> ijl', id_factor - gain_factor, self.covariance)
            state_upd = gs.einsum('ijk, ik -> ij', gain, innovation)
            self.state = self.model.update(self.state, state_upd)
        else:
            self.covariance = (gs.eye(self.model.dim) - gain.dot(H)).dot(
                self.covariance)
            self.state = self.model.update(self.state, gain.dot(innovation))

class KalmanFilterConstraints(KalmanFilter):

    def compute_gain(self, observation, state):
        N = self.model.get_measurement_noise_cov(state, self.measurement_noise)
        H = self.model.observation_jacobian(state, observation)
        constraint_dir, constrained_val = self.model.get_constraint(self.state)

        if gs.ndim(self.state) > 1:
            n_states = self.state.shape[0]
            estimate_cov = gs.einsum(
                'ijk, ikl, ilm -> ijm', H, self.covariance, H.transpose(0, 2, 1))
            innovation_info = gs.linalg.inv(estimate_cov + N)
            kalman_gain = gs.einsum(
                'ijk, ikl, ilm -> ijm', self.covariance, H.transpose(0, 2, 1),
                innovation_info)
            scale = gs.einsum(
                'ijk, ikl, ilm -> ijm', constraint_dir.transpose(0, 2, 1), innovation_info, constraint_dir)
            projection_term = gs.einsum(
                'ijk, ikl, ilm -> ijm', constraint_dir, gs.linalg.inv(scale),
                constraint_dir.transpose(0, 2, 1))
            projection_factor = gs.einsum('ijk, ikl -> ijl', projection_term, innovation_info)
            id_factor = gs.array([gs.eye(self.model.dim_obs)] * n_states)
            state_gain = gs.einsum('ijk, ikl -> ijl', kalman_gain, id_factor - projection_factor)

        else:
            innovation_cov = H.dot(self.covariance).dot(H.T) + N
            innovation_info = gs.linalg.inv(innovation_cov)

            kalman_gain = self.covariance.dot(H.T).dot(gs.linalg.inv(innovation_cov))
            scale = constraint_dir.T.dot(innovation_info).dot(constraint_dir)
            projection_term = constraint_dir.dot(gs.linalg.inv(scale)).dot(constraint_dir.T)
            state_gain = kalman_gain.dot(
                gs.eye(self.model.dim_obs) - projection_term.dot(innovation_info))
        return kalman_gain, state_gain, projection_term


    def update(self, observation):
        innovation = self.model.innovation(self.state, observation)
        kalman_gain, state_gain, projection_term = self.compute_gain(
            observation, self.state)
        H = self.model.observation_jacobian(self.state, observation)
        if gs.ndim(self.state) > 1:
            n_states = self.state.shape[0]
            gain_factor = gs.einsum('ijk, ikl -> ijl', kalman_gain, H)
            id_factor = gs.array([gs.eye(self.model.dim)] * n_states)
            kalman_cov_upd = gs.einsum(
                'ijk, ikl -> ijl', id_factor - gain_factor, self.covariance)
            projection_upd = gs.einsum(
                'ijk, ikl, ilm -> ijm', kalman_gain, projection_term,
                kalman_gain.transpose(0, 2, 1))
            self.covariance = kalman_cov_upd + projection_upd
            state_upd = gs.einsum('ijk, ik -> ij', state_gain, innovation)
            self.state = self.model.update(self.state, state_upd)
            # covariance_correction = self.model.ad_chi(state_upd)
            # self.covariance = gs.einsum('ijk, ikl, ilm -> ijm', covariance_correction, self.covariance,
            #                      covariance_correction.transpose(0, 2, 1))
        else:
            kalman_cov_upd = (gs.eye(self.model.dim) - kalman_gain.dot(H)).dot(
                self.covariance)
            self.covariance = kalman_cov_upd + kalman_gain.dot(projection_term).dot(
                kalman_gain.T)
            self.state = self.model.update(self.state, state_gain.dot(innovation))
            # covariance_correction = self.model.ad_chi(state_gain.dot(innovation))
            # self.covariance = covariance_correction.dot(self.covariance).dot(covariance_correction.T)


nb_gps = 2
obs_corruption_mode = 'both'
model = Localization(nb_gps=nb_gps)
# model = LocalizationEKF(nb_gps=nb_gps)
filter = KalmanFilter(model)
filter_cons = KalmanFilterConstraints(model)

n_traj = 4000
obs_freq = 50
dt = 0.1
P0 = gs.array([1., 10., 10.])
P0 = np.diag(P0)
# Q = np.diag([1e-4, 1e-4, 1e-6])
Q = 0.001 * gs.eye(3)
N = 0.01 * gs.eye(2 * nb_gps)
obs_corruption_values = {'distance' : 1.2,
                     'angle' : -0.1}
obs_corruption_values['both'] = [obs_corruption_values['distance'], obs_corruption_values['angle']]
model.set_obs_corruption_mode(obs_corruption_mode)
obs_corruption_value = obs_corruption_values[obs_corruption_mode]
initial_covs = (P0, Q, N)

true_state = gs.array([0, 0, 0])
# true_state = gs.array([0.02, -30., 30.])
# initial_state = true_state + 2*np.diag(P0)
initial_state = true_state + np.random.multivariate_normal([0,0,0], P0)

true_inputs = [gs.array([dt, 0.5, 0.5, 0.1]) for _ in range(n_traj)]
# true_inputs = [gs.array([dt, .2, 0., 0.]) for _ in range(n_traj)]
# true_inputs = [gs.array([dt, .2, 0., 0.]) for _ in range(n_traj//2)] + [gs.array([0.1, 0., -.2, 0.]) for _ in range(n_traj//2, n_traj)]


true_traj = [1*true_state]
for input in true_inputs:
    true_traj.append(model.propagate(true_traj[-1], input))
true_traj = gs.array(true_traj)

true_obs = [pose[1:] for pose in true_traj[obs_freq::obs_freq]]


plt.figure()
plt.plot(true_traj[:,1], true_traj[:,2], label='GT')

np.random.seed(12345)
inputs = [gs.concatenate((incr[:1], np.random.multivariate_normal(incr[1:], Q))) for incr in true_inputs]
obs = [model.corrupt_measure(gs.tile(meas, nb_gps), obs_corruption_value) for meas in true_obs]
obs = [np.random.multivariate_normal(meas, N) for meas in obs]


def estimation(observer, initial_covs, initial_state, inputs, obs):
    observer.initialise_covariances(*initial_covs)
    observer.set_rotation(initial_state[0])
    observer.set_position(initial_state[1:])

    traj = [1 * observer.state]
    for i in range(n_traj):
        observer.propagate(inputs[i])
        if i > 0 and i % obs_freq == obs_freq - 1:
            observer.update(obs[(i // obs_freq)])
        traj.append(1 * observer.state)
    traj = gs.array(traj)

    return traj

def estimation_vectorized(observer, initial_covs, initial_state, inputs, obs):
    observer.initialise_covariances(*initial_covs)
    observer.vectorize(initial_state.shape[0])
    observer.set_rotation(initial_state[:, :1])
    observer.set_position(initial_state[:, 1:])

    traj = [1 * observer.state]
    for i in range(n_traj):
        observer.propagate(inputs[:, i])
        if i > 0 and i % obs_freq == obs_freq - 1:
            observer.update(obs[:, (i // obs_freq)])
        traj.append(1 * observer.state)
    traj = gs.array(traj)

    return traj

for observer in [filter, filter_cons]:
    traj = estimation(observer, initial_covs, initial_state, inputs, obs)
    plt.plot(traj[:, 1], traj[:, 2], label=observer.__class__.__name__)

    obs = gs.array(obs)
for gps_index in range(nb_gps):
    plt.scatter(obs[:, 2 * gps_index], obs[:, 2 * gps_index + 1], s=2, c='k', label='Observation'+str(gps_index))
plt.legend()
plt.axis('equal')

from time import time


# model = Localization()
# # model = LocalizationEKF()
# filter = KalmanFilter(model)
# filter_cons = KalmanFilterConstraints(model)
n_MC = 50
start = time()
initial_state_vectorized = gs.array([
    np.random.multivariate_normal(true_state, P0) for _ in range(n_MC)])
inputs_vectorized = gs.array([[gs.concatenate((incr[:1], np.random.multivariate_normal(incr[1:], Q))) for incr in true_inputs] for _ in range(n_MC)])
obs_vectorized = [
    [model.corrupt_measure(
        np.random.multivariate_normal(gs.tile(pos, nb_gps), N), obs_corruption_value)
        for pos in true_obs] for _ in range(n_MC)]
obs_vectorized = gs.array(obs_vectorized)

traj_test = estimation(filter, initial_covs, initial_state_vectorized[0], inputs_vectorized[0], obs_vectorized[0])

# trajs_kf = estimation_vectorized(filter, initial_covs, initial_state_vectorized[:1], inputs_vectorized[:1], obs_vectorized[:1])
trajs_kf = estimation_vectorized(filter, initial_covs, initial_state_vectorized, inputs_vectorized, obs_vectorized)
trajs_cons = estimation_vectorized(filter_cons, initial_covs, initial_state_vectorized, inputs_vectorized, obs_vectorized)

error_heading_kf = filter.model.angle_difference_vectorized(trajs_kf[:, :, 0], true_traj[:, 0])
error_heading_constraints = filter.model.angle_difference_vectorized(trajs_cons[:, :, 0], true_traj[:, 0])
error_position_kf = trajs_kf[:, :, 1:] - true_traj[:, np.newaxis, 1:]
error_position_constraints = trajs_cons[:, :, 1:] - true_traj[:, np.newaxis, 1:]
print(time() - start)

MSE_position_both_axis_kf = gs.sum(error_position_kf ** 2, axis=1)/n_MC
MSE_position_norm_kf = gs.sum(MSE_position_both_axis_kf, axis=1)
RMSE_position_kf = gs.sqrt(MSE_position_norm_kf)
MSE_heading_kf = gs.sum(error_heading_kf ** 2, axis=1)/n_MC
RMSE_heading_kf = gs.sqrt(MSE_heading_kf)

MSE_position_both_axis_constraints = gs.sum(error_position_constraints ** 2, axis=1)/n_MC
MSE_position_norm_constraints = gs.sum(MSE_position_both_axis_constraints, axis=1)
RMSE_position_constraints = gs.sqrt(MSE_position_norm_constraints)
MSE_heading_constraints = gs.sum(error_heading_constraints ** 2, axis=1)/n_MC
RMSE_heading_constraints = gs.sqrt(MSE_heading_constraints)

plt.figure()
plt.plot(RMSE_position_kf, label='IEKF')
plt.plot(RMSE_position_constraints, label='IELCKF')
plt.title('Position RMSE')
plt.legend()

plt.figure()
plt.plot(np.std(error_position_kf, axis=1)[:, 0], label='IEKF x std')
plt.plot(np.std(error_position_kf, axis=1)[:, 1], label='IEKF y std')
plt.plot(np.std(error_position_constraints, axis=1)[:, 0], label='IELCKF x std')
plt.plot(np.std(error_position_constraints, axis=1)[:, 1], label='IELCKF y std')
plt.legend()
plt.title('Position std per coordinate')

plt.figure()
plt.plot(np.std(error_heading_kf, axis=1), label='IEKF heading std')
plt.plot(np.std(error_heading_constraints, axis=1), label='IELCKF heading std')
plt.legend()
plt.title('Heading std')

plt.figure()
plt.plot(RMSE_heading_kf, label='IEKF')
plt.plot(RMSE_heading_constraints, label='IELCKF')
plt.title('Heading RMSE')
plt.legend()
