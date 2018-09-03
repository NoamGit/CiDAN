import pickle

from numpy import int8
from scipy.signal import correlate
from simulated_data import *
from simulated_data.utils import *
from scipy.integrate import cumtrapz


class AdversGraphSimulator():

    def __init__(self, N=1000, n_nodes=96, T=10, lambda_const=5):
        self.N = N
        self.n_nodes = n_nodes
        self.T = T
        self.dt = T / N
        self.t = np.arange(0, T, self.dt)
        self.lambda_const = lambda_const

    def build_activity_tensor_from_intervals(self, interval_tensor):
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

    def create_interval_matrix(self, rate_mat, symmetric=True, self_connection=False):
        intervals = np.random.exponential(1, size=(self.n_nodes, self.n_nodes, self.N)) / rate_mat
        # TODO: can be improved computationally be generating only `np.arange(self.n_nodes+1).sum()` rand_numbers and reshaping the triu
        if symmetric:
            for k in np.arange(0, self.N):
                intervals[:, :, k] = trans_to_symm(intervals[:, :, k]) if self_connection else trans_to_symm(
                    intervals[:, :, k], diag_value=self.T)
        return intervals

    def get_homog_interval_model(self, lambda_const=None, **kw_interval_mat):
        """
        intervals are of a homogeneous poisson process with constant rate lambda
        :return:
        """
        rate_mat = lambda_const if lambda_const is not None else self.lambda_const  # lambda_const * np.ones((n_nodes, n_nodes, N))
        return self.create_interval_matrix(rate_mat, **kw_interval_mat)

    def build_activity_tensor_time_rescaling(self, lambd_t: np.ndarray, symmetric=True):
        """

        :param lambd_t: temporal intensity function
        :return:
        """

        # TODO: generalize to list of intensities

        cum_int = np.append(cumtrapz(lambd_t, dx=self.dt),0)
        prev_index = (cum_int.shape[0]-1) *  np.ones((self.n_nodes, self.n_nodes), dtype=np.int8)
        rv_exp = np.random.exponential(1, size=(self.n_nodes, self.n_nodes, self.N))
        act_mat = np.zeros((self.n_nodes, self.n_nodes, self.N), dtype=np.int8)
        # cum_int = np.append(cumtrapz(intensity_func, dx=dt),0)
        # act_mat = np.zeros((96, 96, len(source_tens)))
        # prev_index =  (cum_int.shape[0]-1) * np.ones((96,96), dtype=np.int8)
        for k in range(self.N-1):
            accept_mask = cum_int[k] - cum_int[prev_index] >= rv_exp[:, :, k]
            # print(f"sum accept mask - {accept_mask.sum()}\t iteration - {k}")
            if symmetric:
                accept_mask = trans_to_symm(accept_mask, diag_value=False)
            prev_index[accept_mask] = k
            act_mat[:, :, k] = accept_mask
        # act_mat[:, :, -1] = (cum_int[k] + intensity_func[-1]) - cum_int[prev_index] >= rv_exp[:, :, k]
        act_mat[:, :, -1] = (cum_int[k] + lambd_t[-1]) - cum_int[prev_index] >= rv_exp[:, :, k]
        return act_mat

    @staticmethod
    def validate_homog_poisson(act_mat, time_axis, return_cv=False, symmetric=False):
        """check if the intervals from the matrix have cv = 1"""
        bool_act_mat = act_mat
        setattr(bool_act_mat, 'dtype', bool)
        if symmetric:
            idx = np.triu_indices(act_mat.shape[0])
            for k in range(act_mat.shape[-1]):
                a_itr = act_mat[:, :, k]
                a_itr[idx] = 0
                act_mat[:, :, k] = a_itr
        # vectorized tau calculation
        tau = np.diff(np.tile(time_axis, (act_mat[:, :, 0].shape[0], act_mat[:, :, 0].shape[0], 1))[act_mat])
        tau = tau[tau > 0]
        if return_cv:
            return coefficient_of_variation(tau)
        else:
            assert np.isclose(coefficient_of_variation(tau), 1, rtol=1e-1)


if __name__ == "__main__":

    # activity matrix creation
    with open(r"data/Reality_Mining_MIT.adjs", "rb") as f:
        source_tens = pickle.load(f)
    dt = 1  # day
    source_tens = source_tens[20:]
    source_shape = source_tens[0].shape
    sim = AdversGraphSimulator(N=len(source_tens), n_nodes=source_shape[0], T=1/24*len(source_tens), lambda_const=None)

    # kernel creation
    kernel_param = dict(num_people=8, time_span=4)
    time_kernel = 1 / np.exp(np.arange(0, 4, 4 / kernel_param["time_span"]))
    sampled_interactions = np.random.randint(0, source_shape[0], kernel_param['num_people'])
    kernel_nonzeros, kernel_values = [], []
    X, Y = np.meshgrid(sampled_interactions, np.arange(source_shape[0]))
    sp_kernel = list()
    for n in range(kernel_param["time_span"]):
        sp_kernel_itr = dok_matrix(source_shape)
        sp_kernel_itr[X, Y] = time_kernel[n]
        sp_kernel_itr[Y, X] = time_kernel[n]
        sp_kernel += [sp_kernel_itr]

    # creating non-negative intensity function and rescale
    non_linear_func = lambda x: (x + 1) ** 2
    intensity_func_source = sparse_tensor_convolution(source_tens, sp_kernel)

    # creating destination intensity and correcting relative phase shift
    f = 1 # 1 cycle per day
    intensity_func_dest = np.sin( 2 * np.pi * f * sim.t)
    cross_correlation = correlate(intensity_func_source, intensity_func_dest)
    shift_calculated = (cross_correlation.argmax() - (sim.N)) * sim.dt
    intensity_func_dest_phase_corr = np.sin( 2 * np.pi * f * (sim.t-shift_calculated))

    # plt.figure()
    # plt.plot(sim.t[:500],intensity_func_source[:500])
    # plt.plot(sim.t[:500], 5 * np.median(intensity_func_source) * intensity_func_dest_phase_corr[:500])
    # plt.plot(sim.t[:500],5*np.median(intensity_func_source) * intensity_func_dest[:500])
    # sp = np.fft.fft(intensity_func_source)
    # freq = np.fft.fftfreq(sim.t.shape[-1])
    # plt.plot(freq, sp.real, freq, sp.imag)

    intensity_func = (np.median(intensity_func_source) * intensity_func_dest_phase_corr) + intensity_func_source
    intensity_func = non_linear_func(intensity_func)
    intensity_func -= intensity_func.min()
    intensity_func /= intensity_func.ptp()
    intensity_func *= dt

    # building destination activity tensor
    dest_tens = sim.build_activity_tensor_time_rescaling(intensity_func)

    # save
    sim_net = []
    for k in range(dest_tens.shape[0]):
        sim_net += [dok_matrix(dest_tens[:,:,k])]
    with open(r"data/sim_dynamic_network.adjs", "wb") as f:
        pickle.dump(sim_net,f)
