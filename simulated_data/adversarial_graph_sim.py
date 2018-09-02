import pickle
from simulated_data import *
from simulated_data.utils import *


class AdversGraphSimulator():

    def __init__(self, N=1000, n_nodes=96, T=10, lambda_const=5):
        self.N = N
        self.n_nodes = n_nodes
        self.T = T
        self.dt = T / N
        self.t = np.arange(0, T, self.dt)
        self.lambda_const = lambda_const

    def build_activity_tensor(self, interval_tensor):
        activity_mat = np.zeros((self.n_nodes, self.n_nodes, self.N + 1), dtype=np.int8)
        ind_mat_unfilt = np.searchsorted(self.t, interval_tensor, side='left') + np.arange(0, self.N)
        ind_mat_unfilt[ind_mat_unfilt > self.N] = self.N
        ind_mat_unfilt = np.concatenate(
            (ind_mat_unfilt, np.ones((self.n_nodes, self.n_nodes, 1), dtype=np.int8) * int((self.N))), axis=-1)
        X, Y = np.meshgrid(range(self.n_nodes), range(self.n_nodes))
        ind_interval_itr = ind_mat_unfilt[:, :, 0]
        while np.any(ind_interval_itr != self.N):
            activity_mat[Y, X, ind_interval_itr] = 1
            ind_interval_itr = ind_mat_unfilt[Y, X, ind_interval_itr]
        return activity_mat[:, :, :-1]

    def create_interval_matrix(self, rate_mat):
        return np.random.exponential(1, size=(self.n_nodes, self.n_nodes, self.N)) / rate_mat

    def get_homog_interval_model(self, lambda_const=None):
        """
        intervals are of a homogeneous poisson process with constant rate lambda
        :return:
        """
        rate_mat = lambda_const if lambda_const is not None else sim.lambda_const  # lambda_const * np.ones((n_nodes, n_nodes, N))
        return self.create_interval_matrix(rate_mat)

    def get_cox_process_interval_model(self):
        pass

    @staticmethod
    def validate_homog_poisson(act_mat, time_axis):
        """check if the intervals from the matrix have cv = 1"""
        bool_act_mat = act_mat
        setattr(bool_act_mat, 'dtype', bool)
        # vectorized tau calculation
        tau = np.diff(np.tile(time_axis, (96, 96, 1))[act_mat])
        tau = tau[tau > 0]
        assert np.isclose(coefficient_of_variation(tau), 1, rtol=1e-1)


if __name__ == "__main__":
    # sim_param = dict(N=1000, n_nodes=96, T=10,
    #                    lambda_const=5)
    # sim = AdversGraphSimulator(**sim_param)
    # rate_mat = sim.lambda_const  # lambda_const * np.ones((n_nodes, n_nodes, N))
    # tau_mat = np.random.exponential(1, size=(sim.n_nodes, sim.n_nodes, sim.N)) / rate_mat
    # act_tens = sim.build_activity_tensor(tau_mat)
    # sim.validate_homog_poisson(act_tens, sim.t)

    import tensorflow as tf

    tf.enable_eager_execution()

    # activity matrix creation
    with open(r"data/Reality_Mining_MIT.adjs", "rb") as f:
        source_tens = pickle.load(f)
    source_tens = source_tens[20:]
    source_shape = source_tens[0].shape

    source_nonzeros = []
    for k in range(len(source_tens)):
        x, y = source_tens[k].nonzero()
        source_nonzeros += [[x, y, k] for x, y in zip(x, y)]
    sp_source_tensor = tf.SparseTensor(indices=source_nonzeros, values=np.ones(len(source_nonzeros), dtype=np.int8),
                                       dense_shape=list(source_shape + (len(source_tens),)))

    # kernel creation
    kernel_param = dict(num_people=8, time_span=4)
    time_kernel = 1/np.exp(np.arange(0,4,4/kernel_param["time_span"]))
    sampled_interactions = np.random.randint(0,source_shape[0],kernel_param['num_people'])
    kernel_nonzeros ,kernel_values = [], []
    X, Y = np.meshgrid(sampled_interactions, np.arange(source_shape[0]))
    sp_kernel = list()
    for n in range(kernel_param["time_span"]):
        sp_kernel_itr = dok_matrix(source_shape)
        sp_kernel_itr[X,Y] = time_kernel[n]
        sp_kernel_itr[Y,X] = time_kernel[n]
        sp_kernel += [sp_kernel_itr]
        # for k, l in zip(X.flatten(), Y.flatten()):
        #     sp_kernel_itr[k,l] = time_kernel[n]
        #     sp_kernel_itr[l, k] = time_kernel[n]
        #     kernel_nonzeros += [[k, l, n],[l, k, n]]
        #     kernel_values += list(time_kernel[n].repeat(2))
    # sp_kernel_tensor = tf.SparseTensor(indices=kernel_nonzeros, values=kernel_values,
    #                                    dense_shape=list(source_shape + (kernel_param["time_span"],)))

    # intensity function creation
    tf.nn.conv3d