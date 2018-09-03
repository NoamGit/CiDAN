from unittest import TestCase
from simulated_data import *
from simulated_data.adversarial_graph_sim import AdversGraphSimulator


class TestSynth(TestCase):

    def setUp(self):
        np.random.seed(12456)
        self.interval_tensor = np.array([0.8 * np.ones((3, 3)), 0.2 * np.ones((3, 3))
                                            , 0.6 * np.ones((3, 3)), 0.1 * np.ones((3, 3))]).T
        self.interval_tensor = np.random.permutation(self.interval_tensor.flatten()).reshape((3, 3, 4))
        self.sim = AdversGraphSimulator(N=4, n_nodes=3, T=2, lambda_const=5)

    def test_build_activity_tensor(self):
        am = self.sim.build_activity_tensor_from_intervals(self.interval_tensor)
        self.assertListEqual(list(am[0, 0, :].flatten()), list(np.array([0, 0, 1, 1]).flatten()))
        self.assertSequenceEqual(list(am[1, 0, :].flatten()), list(np.array([0, 1, 0, 1]).flatten()))
        self.assertSequenceEqual(list(am[2, 0, :].flatten()), list(np.array([0, 0, 1, 1]).flatten()))

    def test_homog_poisson(self):
        sim_param = dict(N=1000, n_nodes=96, T=1,
                           lambda_const=10)
        self.sim = AdversGraphSimulator(**sim_param)
        tau_mat = self.sim.get_homog_interval_model(lambda_const=None,symmetric=True, self_connection=False)
        act_tens = self.sim.build_activity_tensor_from_intervals(tau_mat)
        cv = self.sim.validate_homog_poisson(act_tens, self.sim.t, return_cv=True, symmetric=True)
        self.assertAlmostEqual(cv,1,0)

    def test_time_rescaling(self):
        sim_param = dict(N=24*100, T=100, n_nodes=30)
        self.sim = AdversGraphSimulator(**sim_param)
        intensity_func = np.sin(2 * np.pi * 1 * self.sim.t)
        intensity_func -= intensity_func.min()
        intensity_func /= intensity_func.ptp()
        dest_tens = self.sim.build_activity_tensor_time_rescaling(intensity_func)
        dest_tens.mean(axis=(0, 1))
        sp = np.fft.fft(dest_tens.mean(axis=(0, 1)))
        freq = np.fft.fftfreq(self.sim.t.shape[-1])
        self.assertAlmostEqual(np.abs(freq[np.argmax(sp.imag)]),self.sim.dt)