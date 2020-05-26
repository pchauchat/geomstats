import os
os.environ['GEOMSTATS_BACKEND'] = 'numpy'  # NOGA
import matplotlib.pyplot as plt
import numpy as np


import geomstats.backend as gs
from geomstats import algebra_utils
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.special_euclidean import SpecialEuclidean

from scipy.linalg import block_diag

class LocalizationLinear:

    def __init__(self):
        self.group = Euclidean(2)
        self.dim = self.group.dim
        self.dim_noise = 3
        self.dim_obs = 1

    def propagate(self, state, sensor_input):
        """ Standard constant velocity motion model on SE(2),
        with a bidimensional velocity. """
        dt, acc = sensor_input
        pos, speed = state
        pos = pos + dt * speed
        speed = speed + dt * acc
        return gs.array([pos, speed])

    def update(self, state, vector):
        new_state = self.group.exp(vector, state)
        if gs.ndim(state) == 1 and gs.ndim(new_state) > 1:
            return gs.squeeze(new_state)
        return new_state

    def propagation_jacobian(self, state, sensor_input):
        ''' Returns the jacobian associated to the variable i for the propagation factor
            f(X_i)^{-1} X_{i+1}, where f depends on the input u.
            More precisely, it is the jacobian of the tangent map at the identity
            of g(X) = f(I)^{-1} f(X) '''
        dt, _ = sensor_input
        jac = gs.eye(self.dim)
        jac[0, 1] = dt
        return jac

    def noise_jacobian(self, state, sensor_input):
        dt, _ = sensor_input
        return gs.sqrt(dt) * gs.eye(self.dim, self.dim_noise)

    def observation_jacobian(self, state, observation):
        return gs.eye(1, 2, 1)

    def get_measurement_noise_cov(self, state, observation_cov):
        return observation_cov

    def innovation(self, state, observation):
        return observation - state[1:]


class Localization:
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

    def __init__(self):
        self.group = SpecialEuclidean(2, 'vector')
        self.dim = self.group.dim
        self.dim_noise = 3
        self.dim_obs = 2

    def ad_chi(self, state):
        ''' Returns the tangent map associated to Ad_X : g |-> XgX^-1'''
        theta, x, y = self.split_state(state)
        tangent_base = gs.array([[0, -1],
                                 [1, 0]])
        ad = gs.eye(3)
        ad[1:, 1:] = self.rotation_matrix(theta)
        ad[1:, 0] = -tangent_base.dot([x, y])

        return ad

    def propagate(self, state, input):
        """ Standard constant velocity motion model on SE(2),
        with a bidimensional velocity. """
        dt, linear_speed, angular_speed = self.split_input(input)
        theta, x, y = self.split_state(state)
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
        input_vector_form = dt*gs.hstack((gs.array([angular_speed]).T, linear_speed))
        input_inv = self.group.inverse(input_vector_form)

        return self.ad_chi(input_inv)

    def noise_jacobian(self, state, input):
        dt, _, _ = self.split_input(input)
        return dt * gs.eye(self.dim_noise)

    def observation_jacobian(self, state, observation):
        return gs.eye(2, 3, 1)

    def get_measurement_noise_cov(self, state, observation_cov):
        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        if gs.ndim(state) > 1:
            return gs.einsum('ijk, ikl, ilm -> ijm', rot, observation_cov,
                             rot.transpose(0, 2, 1))
        return rot.T.dot(observation_cov).dot(rot)

    def innovation(self, state, observation):
        theta, _, _ = self.split_state(state)
        rot = self.rotation_matrix(theta)
        return rot.T.dot(observation - state[1:])


class KalmanFilter:

    def __init__(self, model):
        self.model = model
        self.state = model.group.get_identity()
        self.covariance = gs.zeros((self.model.dim, self.model.dim))
        self.process_noise = gs.zeros(
            (self.model.dim_noise, self.model.dim_noise))
        self.measurement_noise = gs.zeros(
            (self.model.dim_obs, self.model.dim_obs))

    def initialise_covariances(self, prior_values, process_values, obs_values):
        values = [prior_values, process_values, obs_values]
        attributes = ['covariance', 'process_noise', 'measurement_noise']
        for (index, val) in enumerate(values):
            if gs.ndim(val) == 1:
                setattr(self, attributes[index],
                        algebra_utils.from_vector_to_diagonal_matrix(val))
            else:
                setattr(self, attributes[index], val)

    def set_rotation(self, new_rot):
        self.state = gs.concatenate(([new_rot], self.state[1:]))

    def set_position(self, new_pos):
        self.state = gs.concatenate((self.state[:1], new_pos))

    def propagate(self, sensor_input):
        prop_noise = self.process_noise
        prop_jac = self.model.propagation_jacobian(self.state, sensor_input)
        noise_jac = self.model.noise_jacobian(self.state, sensor_input)
        cov = self.covariance
        if gs.ndim(self.state) > 1:
            prop_cov = gs.einsum('ijk, ikl, ilm -> ijm', prop_jac, cov,
                                 prop_jac.transpose(0, 2, 1))
            noise_cov = gs.einsum('ijk, ikl, ilm -> ijm', noise_jac, prop_noise,
                                  noise_jac.transpose(0, 2, 1))
            self.covariance = prop_cov + noise_cov
        else:
            self.covariance = prop_jac.dot(cov).dot(prop_jac.T) + noise_jac.dot(prop_noise).dot(noise_jac.T)
        self.state = self.model.propagate(self.state, input)

    def compute_gain(self, observation):
        N = self.model.get_measurement_noise_cov(
            self.state, self.measurement_noise)
        obs_jac = self.model.observation_jacobian(self.state, observation)
        if gs.ndim(self.state) > 1:
            estimate_cov = gs.einsum(
                'ijk, ikl, ilm -> ijm', obs_jac, self.covariance, obs_jac.transpose(0, 2, 1))
            innovation_info = gs.linalg.inv(estimate_cov + N)
            return gs.einsum('ijk, ikl, ilm -> ijm', self.covariance,
                             obs_jac.transpose(0, 2, 1), innovation_info)
        innovation_cov = obs_jac.dot(self.covariance).dot(obs_jac.T) + N
        return self.covariance.dot(obs_jac.T).dot(gs.linalg.inv(innovation_cov))

    def update(self, observation):
        innovation = self.model.innovation(self.state, observation)
        gain = self.compute_gain(observation)
        obs_jac = self.model.observation_jacobian(self.state, observation)
        if gs.ndim(self.state) > 1:
            n_states = self.state.shape[0]
            gain_factor = gs.einsum('ijk, ikl -> ijl', gain, obs_jac)
            id_factor = gs.array([gs.eye(self.model.dim)] * n_states)
            self.covariance = gs.einsum(
                'ijk, ikl -> ijl', id_factor - gain_factor, self.covariance)
            state_upd = gs.einsum('ijk, ik -> ij', gain, innovation)
            self.state = self.model.update(self.state, state_upd)
        else:
            self.covariance = (gs.eye(self.model.dim) - gain.dot(obs_jac)).dot(
                self.covariance)
            self.state = self.model.update(self.state, gain.dot(innovation))


model = Localization()
filter = KalmanFilter(model)

true_state = gs.array([0, 0, 0])

# true_state = gs.array([0.02, -30., 30.])

n_traj = 4000
obs_freq = 50
# true_inputs = [gs.array([0.1, .2, 0., 0.]) for _ in range(n_traj//2)] + [gs.array([0.1, 0., -.2, 0.]) for _ in range(n_traj//2, n_traj)]
true_inputs = [gs.array([0.1, 0.5, 0.5, 0.1]) for _ in range(n_traj)]
true_traj = [1*true_state]
for input in true_inputs:
    true_traj.append(model.propagate(true_traj[-1], input))

true_traj = gs.array(true_traj)

plt.figure()
plt.plot(true_traj[:,1], true_traj[:,2], label='GT')

np.random.seed(12345)
inputs = [gs.concatenate(([0.1], np.random.multivariate_normal(incr[1:], 0.001*gs.eye(3)))) for incr in true_inputs]
true_obs = [pose[1:] for pose in true_traj[obs_freq::obs_freq]]
obs = [np.random.multivariate_normal(pos, 0.01*gs.eye(2)) for pos in true_obs]

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


initial_covs = (gs.array([0.1, 1., 1.]), 0.001 * gs.ones(3), 0.01 * gs.ones(2))
initial_state = true_state + gs.array([0.2, 2.5, 2.5])
for observer in [filter]:
    traj = estimation(observer, initial_covs, initial_state, inputs, obs)
    plt.plot(traj[:, 1], traj[:, 2], label=observer.__class__.__name__)

    obs = gs.array(obs)
plt.scatter(obs[:, 0], obs[:, 1], s=2, c='k', label='Observation')
plt.legend()
plt.axis('equal')
