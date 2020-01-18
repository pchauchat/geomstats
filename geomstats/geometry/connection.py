"""Affine connections."""

import autograd
from scipy.optimize import minimize

import geomstats.backend as gs


N_STEPS = 10


class Connection(object):
    """TODO: Define class here."""

    def __init__(self, dimension):
        self.dimension = dimension

    def christoffels(self, base_point):
        """Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension]

        Returns
        -------
        gamma : array-like, shape=[n_samples, dimension, dimension, dimension]
        """
        raise NotImplementedError(
                'The Christoffel symbols are not implemented.')

    def connection(self, tangent_vector_a, tangent_vector_b, base_point):
        """TODO: write method description.

        Connection applied to tangent_vector_b in the direction of
        tangent_vector_a, both tangent at base_point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        tangent_vec_b: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
                'connection is not implemented.')

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS):
        """Exponential map associated to the affine connection.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic the equation (initial value problem), using the
        christoffel symbols

        Parameters
        ----------
        tangent_vec: array-like, shape=[n_samples, dimension]
                                 or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        n_steps: int
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        def geodesic_equation(x, v):
            """
            The geodesic ordinary differential equation associated
            with the connection.
            """
            v = gs.to_ndarray(v, to_ndim=2)
            gamma = self.christoffels(x)
            return - gs.einsum('lkij,li,lj->lk', gamma, v, v)

        initial_state = (base_point, tangent_vec)
        flow, _ = self.integrate(geodesic_equation, initial_state,
                                 n_steps=n_steps)
        return flow[-1]

    def log(self, point, base_point, n_steps=N_STEPS):
        """Logarithm map associated to the affine connection.

        Solve the boundary value problem associated to the geodesic equation
        using the christoffel symbols and an conjugate gradient descent

        Parameters
        ----------
        point: array-like, shape=[n_samples, dimension]
                           or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        n_steps: int
        """

        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        def objective(vel):
            vel = vel.reshape(base_point.shape)
            loss = 1 / self.dimension * gs.sum(
                (self.exp(base_point=base_point, tangent_vec=vel,
                          n_steps=n_steps) - point) ** 2, axis=1)
            return 1 / n_samples * gs.sum(loss)

        objective_grad = autograd.elementwise_grad(objective)

        def objective_with_grad(vel):
            return objective(vel), objective_grad(vel)

        n_samples = len(base_point)
        velocity = gs.zeros_like(base_point).flatten()
        res = minimize(objective_with_grad, velocity, method='CG', jac=True,
                       options={'disp': True, 'maxiter': 25})

        tangent_vec_result = res.x
        tangent_vec_result = gs.reshape(tangent_vec_result, base_point.shape)
        return tangent_vec_result

    def pole_ladder_step(self, base_point, next_point, base_shoot):
        """Pole Ladder step.

        One step of pole ladder scheme [1]_ using the geodesic to
        transport along as diagonal of the parallelogram

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        next_point: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        base_shoot: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        transported_tangent_vector: array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]

        end_point: array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]

        References
        ----------
        .. [1] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transport
        of Deformations in Time Series of Images: from Schild's to Pole Ladder.
        Journal of Mathematical Imaging and Vision, Springer Verlag, 2013,
         50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩
        """
        mid_tangent_vector_to_shoot = 1. / 2. * self.log(
                base_point=base_point,
                point=next_point)

        mid_point = self.exp(
                base_point=base_point,
                tangent_vec=mid_tangent_vector_to_shoot)

        tangent_vector_to_shoot = - self.log(
                base_point=mid_point,
                point=base_shoot)

        end_shoot = self.exp(
                base_point=mid_point,
                tangent_vec=tangent_vector_to_shoot)

        transported_tangent_vector = - self.log(
            base_point=next_point, point=end_shoot)

        end_point = self.exp(
                base_point=next_point,
                tangent_vec=transported_tangent_vector)

        return transported_tangent_vector, end_point

    def pole_ladder_parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, n_steps=1):
        """Approximate parallel transport using the pole ladder scheme.

        Approximation of Parallel transport using the pole ladder
        scheme [1][2]. The tangent vector a is transported along along
        the geodesic starting at the initial point base_point with
        initial tangent vector the tangent vector b.

        Returns a tangent vector at the point
        exp_(base_point)(tangent_vector_b).

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        tangent_vec_b: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        n_steps: int
            the number of pole ladder steps

        Returns
        -------
        transported_tangent_vector: array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]

        References
        ----------
        .. [1] Marco Lorenzi, Xavier Pennec. Efficient Parallel Transport
        of Deformations in Time Series of Images: from Schild's to Pole Ladder.
        Journal of Mathematical Imaging and Vision, Springer Verlag, 2013,
         50 (1-2), pp.5-17. ⟨10.1007/s10851-013-0470-3⟩

         .. [2] N. Guigui, Shuman Jia, Maxime Sermesant, Xavier Pennec.
         Symmetric Algorithmic Components for Shape Analysis with
          Diffeomorphisms. GSI 2019, Aug 2019, Toulouse, France. pp.10.
           ⟨hal-02148832⟩
        """
        current_point = gs.copy(base_point)
        transported_tangent_vector = gs.copy(tangent_vec_a)
        base_shoot = self.exp(base_point=current_point,
                              tangent_vec=transported_tangent_vector)
        for i_point in range(0, n_steps):
            frac_tangent_vector_b = (i_point + 1) / n_steps * tangent_vec_b
            next_point = self.exp(
                base_point=base_point,
                tangent_vec=frac_tangent_vector_b)
            transported_tangent_vector, base_shoot = self.pole_ladder_step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot)
            current_point = next_point

        return transported_tangent_vector

    def riemannian_curvature(self, base_point):
        """Riemannian curvature tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
                'The Riemannian curvature tensor is not implemented.')

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None,
                 point_type='vector'):
        """
        Geodesic curve defined by either:
        - an initial point and an initial tangent vector,
        or
        -an initial point and an end point.

        The geodesic is returned as a function parameterized by t.
        """

        point_ndim = 1
        if point_type == 'matrix':
            point_ndim = 2

        initial_point = gs.to_ndarray(initial_point,
                                      to_ndim=point_ndim+1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            end_point = gs.to_ndarray(end_point,
                                      to_ndim=point_ndim+1)
            shooting_tangent_vec = self.log(point=end_point,
                                            base_point=initial_point)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=point_ndim+1)

        def point_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_point = gs.to_ndarray(
                                          initial_point,
                                          to_ndim=point_ndim+1)
            new_initial_tangent_vec = gs.to_ndarray(
                                          initial_tangent_vec,
                                          to_ndim=point_ndim+1)

            if point_type == 'vector':
                tangent_vecs = gs.einsum('il,nk->ik',
                                         t,
                                         new_initial_tangent_vec)
            elif point_type == 'matrix':
                tangent_vecs = gs.einsum('il,nkm->ikm',
                                         t,
                                         new_initial_tangent_vec)

            point_at_time_t = self.exp(tangent_vec=tangent_vecs,
                                       base_point=new_initial_point)
            return point_at_time_t

        return point_on_geodesic

    def torsion(self, base_point):
        """Torsion tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
                'The torsion tensor is not implemented.')

    @staticmethod
    def _symplectic_euler_step(y, force, dt):
        x, v = y
        x_new = x + v * dt
        v_new = v + force(x, v) * dt
        return x_new, v_new

    def integrate(self, function, initial_state, end_time=1.0, n_steps=10):
        """ Integrator function to compute flows of vector fields
        on a regular grid between 0 and a finite time.

        Parameters
        ----------

        function: callable
            the vector field to integrate

        initial_state: tuple
            initial position and speed

        end_time: scalar

        n_steps: int

        Returns
        ----------
        a tuple of sequences of solutions every end_time / n_steps
        """
        dt = end_time / n_steps
        positions = [initial_state[0]]
        velocities = [initial_state[1]]
        current_state = (positions[0], velocities[0])
        for i in range(n_steps):
            current_state = self._symplectic_euler_step(current_state,
                                                        function, dt)
            positions.append(current_state[0])
            velocities.append(current_state[1])

        return positions, velocities


class LeviCivitaConnection(Connection):
    """Levi-Civita connection associated with a Riemannian metric."""

    def __init__(self, metric):
        self.metric = metric
        self.dimension = metric.dimension

    def metric_matrix(self, base_point):
        """Metric matrix defining the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        metric_matrix: array-like, shape=[n_samples, dimension, dimension]
                                   or shape=[1, dimension, dimension]
        """
        metric_matrix = self.metric.inner_product_matrix(base_point)
        return metric_matrix

    def cometric_matrix(self, base_point):
        """Compute cometric.

        The cometric is the inverse of the metric.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        cometric_matrix: array-like, shape=[n_samples, dimension, dimension]
                                     or shape=[1, dimension, dimension]
        """
        metric_matrix = self.metric_matrix(base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix)
        return cometric_matrix

    def metric_derivative(self, base_point):
        """Compute metric derivative at base point.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        metric_derivative = autograd.jacobian(self.metric_matrix)
        return metric_derivative(base_point)

    def christoffels(self, base_point):
        """Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        christoffels: array-like,
                             shape=[n_samples, dimension, dimension, dimension]
                             or shape=[1, dimension, dimension, dimension]
        """
        cometric_mat_at_point = self.cometric_matrix(base_point)
        metric_derivative_at_point = self.metric_derivative(base_point)
        term_1 = gs.einsum('nim,nmkl->nikl',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_2 = gs.einsum('nim,nmlk->nilk',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_3 = - gs.einsum('nim,nklm->nikl',
                             cometric_mat_at_point,
                             metric_derivative_at_point)

        christoffels = 0.5 * (term_1 + term_2 + term_3)
        return christoffels

    def torsion(self, base_point):
        """Torsion tensor associated with the Levi-Civita connection is zero.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        torsion: array-like, shape=[dimension, dimension, dimension]
        """
        torsion = gs.zeros((self.dimension,) * 3)
        return torsion

    def exp(self, tangent_vec, base_point):
        """Exponential map associated to the metric.

        Parameters
        ----------
        tangent_vec: array-like, shape=[n_samples, dimension]
                                 or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        return self.metric.exp(tangent_vec, base_point)

    def log(self, point, base_point):
        """Logarithm map associated to the metric.

        Parameters
        ----------
        point: array-like, shape=[n_samples, dimension]
                           or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        return self.metric.log(point, base_point)
