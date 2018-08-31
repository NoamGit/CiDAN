from unittest import TestCase
from synth_data import *
from synth_data.adversarial_graph_synth import AdversGraphSynth


class TestSynth(TestCase):

    def setUp(self):
        np.random.seed(12456)
        self.interval_tensor = np.array([0.8 * np.ones((3, 3)), 0.2 * np.ones((3, 3))
                                            , 0.6 * np.ones((3, 3)), 0.1 * np.ones((3, 3))]).T
        self.interval_tensor = np.random.permutation(self.interval_tensor.flatten()).reshape((3, 3, 4))
        self.synt = AdversGraphSynth(N=4, n_nodes=3, T=2, lambda_const=5)

    def test_build_activity_tensor(self):
        am = self.synt.build_activity_tensor(self.interval_tensor)
        self.assertListEqual(list(am[0, 0, :].flatten()), list(np.array([0, 0, 1, 1]).flatten()))
        self.assertSequenceEqual(list(am[1, 0, :].flatten()), list(np.array([0, 1, 0, 1]).flatten()))
        self.assertSequenceEqual(list(am[2, 0, :].flatten()), list(np.array([0, 0, 1, 1]).flatten()))
