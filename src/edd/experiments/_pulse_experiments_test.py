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
from edd.pulse import IBMQDdSchedule

import unittest

def dt_time_to_ns(dt_time, dt):
    '''converts time in normalzied dt units to ns'''
    return (dt_time * dt * 1e9)

class PulseExperimentsTest(unittest.TestCase):
    """ Tests that experiments run smoothly. """

    def test_theta_sweep_free_default(self):
        '''test theta_sweep_free with default args'''
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        num_ids = 5
        experiments = edde.pulse.theta_sweep_free(num_ids, armonk)

        # first check that job title is as expected
        sched = IBMQDdSchedule(armonk)
        sched.add_id(0)
        id_time = sched.get_phys_time()
        time = num_ids * id_time
        qubits = [n for n in range(armonk.get_number_qubits())]
        job_tag = f"theta_sweep_free_encodeqs_{qubits}_ddqs_{qubits}_"
        job_tag += f"ids_{num_ids}_T_{time}ns"
        self.assertEquals(experiments[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(experiments[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(experiments[1][0].name, exp_tag1)

    def test_theta_sweep_free_not_default(self):
        '''test theta_sweep_free with default args'''
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        # this is the most complicated case with non-contiguous
        # encoding_qubits and dd_qubits
        e_qubits = [0]
        dd_qubits = [0]
        num_ids = 5
        exps = edde.pulse.theta_sweep_free(num_ids, armonk, e_qubits, dd_qubits)

        # first check that job title is as expected
        sched = IBMQDdSchedule(armonk)
        sched.add_id(0)
        id_time = sched.get_phys_time()
        time = num_ids * id_time
        job_tag = f"theta_sweep_free_encodeqs_{e_qubits}_ddqs_{dd_qubits}_"
        job_tag += f"ids_{num_ids}_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_theta_sweep_dd_default(self):
        '''test theta_sweep_dd with default args'''
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk)
        dt = sched.dt
        num_reps = 5
        tau = 100
        exps = edde.pulse.theta_sweep_dd('xy4', num_reps, tau, armonk)

        # first check that job title is as expected
        sched.add_x(0)
        x_time = sched.get_phys_time()
        # there are 4 u3 gates for each rep of xy4 and 5 reps
        # then there is 1 identity gate in between each pulse
        time = (4 * num_reps) * (x_time + dt_time_to_ns(tau, dt))
        qubits = [n for n in range(armonk.get_number_qubits())]
        job_tag = f"theta_sweep_xy4_encodeqs_{qubits}_ddqs_{qubits}_"
        job_tag += f"reps_{num_reps}_tau_{tau}dt_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_theta_sweep_dd_not_default(self):
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk)
        dt = sched.dt
        # this is the most complicated case with non-contiguous
        # encoding_qubits and dd_qubits
        e_qubits = [0]
        dd_qubits = [0]
        num_reps = 5
        tau = 100
        exps = edde.pulse.theta_sweep_dd('xy4', num_reps, tau, armonk,
                                        e_qubits, dd_qubits)

        # first check that job title is as expected
        sched.add_x(0)
        x_time = sched.get_phys_time()
        # there are 4 u3 gates for each rep of xy4
        # then there is 1 identity gate in between each pulse
        time = (4 * num_reps) * (x_time + dt_time_to_ns(tau, dt))
        job_tag = f"theta_sweep_xy4_encodeqs_{e_qubits}_ddqs_{dd_qubits}_"
        job_tag += f"reps_{num_reps}_tau_{tau}dt_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_0.0"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_static_haar_fid_decay_free_default(self):
        '''tests static_haar_fid_decay_free with default args'''
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        free_time = 100
        exps = edde.pulse.static_haar_fid_decay_free(haar_params_list,
                                                    free_time, armonk)

        # first check that job title is as expected
        sched = IBMQDdSchedule(armonk)
        T = dt_time_to_ns(free_time, sched.dt)
        qubits = [n for n in range(armonk.get_number_qubits())]
        job_tag = f"static_haar_fid_decay_free_encodeqs_{qubits}_"
        job_tag += f"ddqs_{qubits}_time_{free_time}dt_T_{T}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_static_haar_fid_decay_free_not_default(self):
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        free_time = 100
        e_qubits = [0]
        dd_qubits = [0]
        exps = edde.pulse.static_haar_fid_decay_free(haar_params_list,
                                                    free_time, armonk,
                                                    e_qubits, dd_qubits)

        # first check that job title is as expected
        sched = IBMQDdSchedule(armonk)
        T = dt_time_to_ns(free_time, sched.dt)
        job_tag = f"static_haar_fid_decay_free_encodeqs_{e_qubits}_"
        job_tag += f"ddqs_{dd_qubits}_time_{free_time}dt_T_{T}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)

    def test_static_haar_fid_decay_dd_default(self):
        '''tests static_haar_fid_decay_dd with default args'''
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk)
        dt = sched.dt
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        num_reps = 5
        tau = 100
        exps = edde.pulse.static_haar_fid_decay_dd(haar_params_list, 'xy4',
                                                  num_reps, tau, armonk)

        # first check that job title is as expected
        sched.add_x(0)
        x_time = sched.get_phys_time()
        time = (4 * num_reps) * (x_time + dt_time_to_ns(tau, dt))
        qubits = [n for n in range(armonk.get_number_qubits())]
        job_tag = f"static_haar_fid_decay_xy4_encodeqs_{qubits}_"
        job_tag += f"ddqs_{qubits}_reps_{num_reps}_tau_{tau}dt_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)


    def test_static_haar_fid_decay_dd_not_default(self):
        # load backend and create experiments
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk)
        dt = sched.dt
        haar_params_list = [(1, 1, 1), (0.26, .6, .178)]
        num_reps = 5
        tau = 100
        e_qubits = [0]
        dd_qubits = [0]
        exps = edde.pulse.static_haar_fid_decay_dd(haar_params_list, 'xy4',
                                                    num_reps, tau, armonk,
                                                    e_qubits, dd_qubits)

        # first check that job title is as expected
        sched.add_x(0)
        x_time = sched.get_phys_time()
        time = (4 * num_reps) * (x_time + dt_time_to_ns(tau, dt))
        job_tag = f"static_haar_fid_decay_xy4_encodeqs_{e_qubits}_"
        job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_tau_{tau}dt_T_{time}ns"
        self.assertEquals(exps[0], job_tag)

        # next, check that experiment list is as expected
        self.assertIsInstance(exps[1][0], IBMQDdSchedule)
        exp_tag1 = job_tag + "_theta_1_phi_1_lambda_1"
        self.assertEquals(exps[1][0].name, exp_tag1)


if __name__=="__main__":
    unittest.main()
