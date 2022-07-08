#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from edd.backend import IBMQBackend
import edd.experiments as edde

# type check imports
from edd.circuit import IBMQDdCircuit

import unittest

class CircuitExperimentsTest(unittest.TestCase):
    """ Tests that experiments run smoothly. """

    def test_theta_sweep_free_default(self):
        '''test theta_sweep_free with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        num_ids = 5
        experiments = edde.circ.theta_sweep_free(num_ids, london)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        time = num_ids * id_time
        qubits = [n for n in range(london.get_number_qubits())]
        job_tag = f"theta_sweep_free_encodeqs_{qubits}_ddqs_{qubits}_"
        job_tag += f"ids_{num_ids}_T_{time}ns"
        self.assertEquals(experiments[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(experiments[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(experiments[1][0].name, exp_tag1)

    def test_theta_sweep_free_not_default(self):
        '''test theta_sweep_free with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        # this is the most complicated case with non-contiguous
        # encoding_qubits and dd_qubits
        e_qubits = [0, 3]
        dd_qubits = [1, 2, 4]
        num_ids = 5
        exps = edde.circ.theta_sweep_free(num_ids, london, e_qubits, dd_qubits)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        time = num_ids * id_time
        job_tag = f"theta_sweep_free_encodeqs_{e_qubits}_ddqs_{dd_qubits}_"
        job_tag += f"ids_{num_ids}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_theta_sweep_dd_default(self):
        '''test theta_sweep_dd with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        num_reps = 5
        id_pad = 1
        exps = edde.circ.theta_sweep_dd('xy4', num_reps, id_pad, london)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        u3_time = london.get_gate_times()['u3']
        # there are 4 u3 gates for each rep of xy4 and 5 reps
        # then there is 1 identity gate in between each pulse
        time = (4 * num_reps) * (u3_time + id_time)
        qubits = [n for n in range(london.get_number_qubits())]
        job_tag = f"theta_sweep_xy4_encodeqs_{qubits}_ddqs_{qubits}_"
        job_tag += f"reps_{num_reps}_idpad_{id_pad}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_theta_sweep_dd_not_default(self):
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        # this is the most complicated case with non-contiguous
        # encoding_qubits and dd_qubits
        e_qubits = [0, 3]
        dd_qubits = [1, 2, 4]
        num_reps = 5
        id_pad = 1
        exps = edde.circ.theta_sweep_dd('xy4', num_reps, id_pad, london,
                                        e_qubits, dd_qubits)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        u3_time = london.get_gate_times()['u3']
        # there are 4 u3 gates for each rep of xy4
        # then there is 1 identity gate in between each pulse
        time = (4 * num_reps) * (u3_time + id_time)
        job_tag = f"theta_sweep_xy4_encodeqs_{e_qubits}_ddqs_{dd_qubits}_"
        job_tag += f"reps_{num_reps}_idpad_{id_pad}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_static_haar_fid_decay_free_default(self):
        '''tests static_haar_fid_decay_free with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        num_ids = 10
        exps = edde.circ.static_haar_fid_decay_free(haar_params_list,
                                                    num_ids, london)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        time = num_ids * id_time
        qubits = [n for n in range(london.get_number_qubits())]
        job_tag = f"static_haar_fid_decay_free_encodeqs_{qubits}_"
        job_tag += f"ddqs_{qubits}_ids_{num_ids}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_static_haar_fid_decay_free_not_default(self):
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        num_ids = 10
        e_qubits = [1, 4]
        dd_qubits = [1, 3]
        exps = edde.circ.static_haar_fid_decay_free(haar_params_list,
                                                    num_ids, london,
                                                    e_qubits, dd_qubits)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        time = num_ids * id_time
        job_tag = f"static_haar_fid_decay_free_encodeqs_{e_qubits}_"
        job_tag += f"ddqs_{dd_qubits}_ids_{num_ids}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)

    def test_static_haar_fid_decay_dd_default(self):
        '''tests static_haar_fid_decay_dd with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        num_reps = 5
        id_pad = 1
        exps = edde.circ.static_haar_fid_decay_dd(haar_params_list, 'xy4',
                                                  num_reps, id_pad, london)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        u3_time = london.get_gate_times()['u3']
        time = (4 * num_reps) * (u3_time + id_time)
        qubits = [n for n in range(london.get_number_qubits())]
        job_tag = f"static_haar_fid_decay_xy4_encodeqs_{qubits}_"
        job_tag += f"ddqs_{qubits}_reps_{num_reps}_idpad_{id_pad}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_static_haar_fid_decay_dd_not_default(self):
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        num_reps = 5
        id_pad = 1
        e_qubits = [1, 4]
        dd_qubits = [1, 3]
        exps = edde.circ.static_haar_fid_decay_dd(haar_params_list, 'xy4',
                                                    num_reps, id_pad, london,
                                                    e_qubits, dd_qubits)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        u3_time = london.get_gate_times()['u3']
        time = (4 * num_reps) * (u3_time + id_time)
        job_tag = f"static_haar_fid_decay_xy4_encodeqs_{e_qubits}_"
        job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_idpad_{id_pad}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_bell_fid_decay_free_default(self):
        '''tests bell_fid_decay_free with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        qubit_pairs = [(0, 1), (3, 4)]
        num_ids = 5
        exps = edde.circ.bell_fid_decay_free(qubit_pairs, num_ids, london)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        time = num_ids * id_time
        qubits = [0, 1, 3, 4]
        job_tag = f"bell_fid_decay_free_pairs_{qubit_pairs}_ddqs_{qubits}_"
        job_tag += f"ids_{num_ids}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_phi+"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_bell_fid_decay_free_not_default(self):
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        qubit_pairs = [(0, 1), (3, 4)]
        num_ids = 5
        dd_qubits = [1, 2, 4]
        exps = edde.circ.bell_fid_decay_free(qubit_pairs, num_ids, london,
                                             dd_qubits)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        time = num_ids * id_time
        job_tag = f"bell_fid_decay_free_pairs_{qubit_pairs}_ddqs_{dd_qubits}_"
        job_tag += f"ids_{num_ids}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_phi+"
        self.assertEquals(exps[1][0].name, exp_tag1)

    def test_bell_fid_decay_dd_default(self):
        '''tests bell_fid_decay_dd with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        qubit_pairs = [(0, 1), (3, 4)]
        num_reps = 5
        id_pad = 1
        exps = edde.circ.bell_fid_decay_dd(qubit_pairs, 'xy4', num_reps,
                                           id_pad, london)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        u3_time = london.get_gate_times()['u3']
        time = (4 * num_reps) * (u3_time + id_time)
        qubits = [0, 1, 3, 4]
        job_tag = f"bell_fid_decay_xy4_pairs_{qubit_pairs}_ddqs_{qubits}_"
        job_tag += f"reps_{num_reps}_idpad_{id_pad}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_phi+"
        self.assertEquals(exps[1][0].name, exp_tag1)

    def test_bell_fid_decay_dd_not_default(self):
        '''tests bell_fid_decay_dd with default args'''
        # load backend and create experiments
        london = IBMQBackend('ibmq_london')
        qubit_pairs = [(0, 1), (3, 4)]
        num_reps = 5
        id_pad = 1
        dd_qubits = [1, 2, 4]
        exps = edde.circ.bell_fid_decay_dd(qubit_pairs, 'xy4', num_reps,
                                           id_pad, london, dd_qubits)

        # first check that job title is as expected
        id_time = london.get_gate_times()['id']
        u3_time = london.get_gate_times()['u3']
        time = (4 * num_reps) * (u3_time + id_time)
        job_tag = f"bell_fid_decay_xy4_pairs_{qubit_pairs}_ddqs_{dd_qubits}_"
        job_tag += f"reps_{num_reps}_idpad_{id_pad}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdCircuit)
        exp_tag1 = job_tag + "_phi+"
        self.assertEquals(exps[1][0].name, exp_tag1)




if __name__=="__main__":
    unittest.main()

