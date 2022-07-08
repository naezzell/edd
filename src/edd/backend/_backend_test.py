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

import numpy as np
from edd.backend import IBMQBackend

import unittest

# there may be need for unit testing in future, so this is here for that
class BackendTest(unittest.TestCase):
    """ Units tests for IBMQBackend. """

    ##################################################
    # Armonk Tests
    ##################################################
    def test_init_armonk(self):
        armonk = IBMQBackend("ibmq_armonk")
        self.assertIsInstance(armonk, IBMQBackend)

    def test_get_number_qubits_armonk(self):
        armonk = IBMQBackend("ibmq_armonk")
        n_qubits = armonk.get_number_qubits()
        self.assertEqual(n_qubits, 1)

    def test_get_native_gates_armonk(self):
        armonk = IBMQBackend("ibmq_armonk")
        native_gates = armonk.get_native_gates()
        manual_natives = ['id', 'u1', 'u2', 'u3']
        self.assertCountEqual(native_gates, manual_natives)

    def test_get_gate_times_armonk(self):
        armonk = IBMQBackend("ibmq_armonk")
        native_gates = armonk.get_native_gates()
        # get the first gate since all single qubit anyway
        f_g = native_gates[0]
        # get time it takes to run gate on average over the qubits
        avg_times = armonk.get_gate_times()
        f_g_time = avg_times[f_g]
        self.assertIsInstance(f_g_time, float)

        # get times for gate on each qubit
        n_qubits = armonk.get_number_qubits()
        all_times = armonk.get_gate_times(avg=False)
        f_g_times = all_times[f_g]
        self.assertEqual(len(f_g_times), n_qubits)

    def test_get_readable_props_str_armonk(self):
        armonk = IBMQBackend("ibmq_armonk")
        prop_str = armonk.get_readable_props_str()
        print(prop_str)
        self.assertIsInstance(prop_str, str)

    def test_get_max_runs_armonk(self):
        armonk = IBMQBackend("ibmq_armonk")
        max_runs = armonk.get_max_runs()
        max_experiments = max_runs['max_experiments']
        max_shots = max_runs['max_shots']
        self.assertIsInstance(max_experiments, int)
        self.assertIsInstance(max_shots, int)

    def test_get_noisemodel(self):
        armonk = IBMQBackend("ibmq_armonk")
        noise_info = armonk.get_noisemodel()
        self.assertIsInstance(noise_info, dict)

    ##################################################
    # TODO: Write Tests of Basic Circuits and Actually Run them on hardware
    ##################################################


    ##################################################
    # Ourense Tests
    ##################################################
    def test_init_ourense(self):
        ourense = IBMQBackend("ibmq_ourense")
        self.assertIsInstance(ourense, IBMQBackend)

    def test_get_number_qubits_ourense(self):
        ourense = IBMQBackend("ibmq_ourense")
        n_qubits = ourense.get_number_qubits()
        self.assertEqual(n_qubits, 5)

    def test_get_native_gates_ourense(self):
        ourense = IBMQBackend("ibmq_ourense")
        native_gates = ourense.get_native_gates()
        manual_natives = ['id', 'u1', 'u2', 'u3', 'cx']
        self.assertCountEqual(native_gates, manual_natives)

    def test_get_gate_times_ourense(self):
        ourense = IBMQBackend("ibmq_ourense")
        native_gates = ourense.get_native_gates()
        # get the first gate since all single qubit anyway
        f_g = native_gates[0]
        # get time it takes to run gate on average over the qubits
        avg_times = ourense.get_gate_times()
        f_g_time = avg_times[f_g]
        self.assertIsInstance(f_g_time, float)

        # get times for gate on each qubit
        n_qubits = ourense.get_number_qubits()
        all_times = ourense.get_gate_times(avg=False)
        f_g_times = all_times[f_g]
        self.assertEqual(len(f_g_times), n_qubits)

    def test_get_readable_props_str_ourense(self):
        armonk = IBMQBackend("ibmq_ourense")
        prop_str = armonk.get_readable_props_str()
        print(prop_str)
        self.assertIsInstance(prop_str, str)

    def test_get_max_runs_ourense(self):
        armonk = IBMQBackend("ibmq_ourense")
        max_runs = armonk.get_max_runs()
        max_experiments = max_runs['max_experiments']
        max_shots = max_runs['max_shots']
        self.assertIsInstance(max_experiments, int)
        self.assertIsInstance(max_shots, int)

    def test_get_noisemodel(self):
        ourense = IBMQBackend("ibmq_ourense")
        noise_info = ourense.get_noisemodel()
        self.assertIsInstance(noise_info, dict)

    ##################################################
    # TODO: Write Tests of Basic Circuits and Actually Run them on hardware
    ##################################################


if __name__ == "__main__":
    unittest.main()
    
