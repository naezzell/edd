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

import unittest
import numpy as np

from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit

from edd.backend import IBMQBackend
from edd.circuit import IBMQDdCircuit

# there may be need for unit testing in future, so this is here for that
class DdCircuitTest(unittest.TestCase):
    """ Units tests for IBMQBackend. """

    ##################################################
    # General IBMQDdCircuit method tests
    ##################################################
    def test_get_gate_count_no_trans_armonk(self):
        armonk = IBMQBackend('ibmq_armonk')
        circ = IBMQDdCircuit(1)
        circ.x(0)
        circ.z(0)
        circ.h(0)
        answer = {'x': 1, 'z': 1, 'h': 1}
        gate_count = circ.get_gate_count(False, armonk)
        self.assertDictEqual(answer, gate_count)

    def test_get_gate_count_with_trans_armonk(self):
        armonk = IBMQBackend('ibmq_armonk')
        circ = IBMQDdCircuit(1)
        circ.x(0)
        circ.z(0)
        circ.h(0)
        answer = {'u2': 1}
        gate_count = circ.get_gate_count(True, armonk)
        self.assertDictEqual(answer, gate_count)

    def test_get_gate_count_with_trans_and_barriers_armonk(self):
        armonk = IBMQBackend('ibmq_armonk')
        circ = IBMQDdCircuit(1)
        circ.x(0)
        circ.barrier(0)
        circ.z(0)
        circ.barrier(0)
        circ.h(0)
        circ.barrier(0)
        answer = {'u1': 1, 'u2': 1, 'u3': 1, 'barrier': 3}
        gate_count = circ.get_gate_count(True, armonk)
        self.assertDictEqual(answer, gate_count)

    def test_get_transpiled_circ_armonk(self):
        armonk = IBMQBackend('ibmq_armonk')
        circ = IBMQDdCircuit(1)
        circ.x(0)
        circ.barrier(0)
        circ.z(0)
        circ.barrier(0)
        circ.h(0)
        t_circ = circ.get_transpiled_circ(armonk)
        circ_dag = circuit_to_dag(t_circ)
        gate_count = circ_dag.count_ops()
        answer = {'u1': 1, 'u2': 1, 'u3': 1, 'barrier': 2}
        self.assertDictEqual(answer, gate_count)

    def test_get_phys_time_armonk(self):
        armonk = IBMQBackend('ibmq_armonk')
        gate_times = armonk.get_gate_times()
        circ = IBMQDdCircuit(1)
        circ.z(0)
        curr_time = circ.get_phys_time(armonk)
        # at this point, only added virtual z
        self.assertEquals(0, curr_time)

        # now add identity
        circ.barrier(0)
        circ.id(0)
        answer = gate_times['id']
        curr_time = circ.get_phys_time(armonk)
        self.assertEquals(answer, curr_time)

        # now ensure that adding gates increases the time
        circ.barrier(0)
        # this is x in terms of u3
        circ.add_x(0)
        answer += gate_times['u3']
        curr_time = circ.get_phys_time(armonk)
        self.assertEquals(answer, curr_time)

    def test_get_statevector(self):
        circ1 = IBMQDdCircuit(1)
        answer1 = np.array([1.+0.j, 0.+0.j])
        s = circ1.get_statevector()
        self.assertEquals(np.allclose(s, answer1), True)

        circ2 = IBMQDdCircuit(5)
        answer2 = []
        for j in range(2**5):
            if j == 0:
                answer2.append(1.+0.j)
            else:
                answer2.append(0.+0.j)
        answer2 = np.array(answer2)
        s = circ2.get_statevector()
        self.assertEquals(np.allclose(s, answer2), True)


    def test_get_unitary(self):
        circ1 = IBMQDdCircuit(1)
        id2 = np.identity(2)
        u = circ1.get_unitary()
        self.assertEquals(np.allclose(u, id2), True)

        circ2 = IBMQDdCircuit(5)
        id5 = np.identity(2**5)
        u = circ2.get_unitary()
        self.assertEquals(np.allclose(u, id5), True)


    ##################################################
    # Basic Gate Tests
    ##################################################
    def test_add_x(self):
        circ = IBMQDdCircuit(1)
        circ.add_x(0)
        x = np.array([[0, 1], [1, 0]])
        u = circ.get_unitary()
        self.assertEquals(np.allclose(u, x), True)

    def test_add_y(self):
        circ = IBMQDdCircuit(1)
        circ.add_y(0)
        y = np.array([[0, -1j], [1j, 0]])
        u = circ.get_unitary()
        self.assertEquals(np.allclose(u, y), True)

    def test_add_z(self):
        circ = IBMQDdCircuit(1)
        circ.add_z(0)
        z = np.array([[1, 0], [0, -1]])
        u = circ.get_unitary()
        self.assertEquals(np.allclose(u, z), True)

    def test_add_zii(self):
        circ = IBMQDdCircuit(1)
        circ.add_zii(0)
        z = np.array([[1, 0], [0, -1]])
        u = circ.get_unitary()
        # check that you get correct unitary
        self.assertEquals(np.allclose(u, z), True)
        # now check that it really has added z_i_i
        gate_counts = {"u1": 1, "barrier": 2, "id": 2}
        self.assertDictEqual(circ.get_gate_count(), gate_counts)

    def test_add_xb(self):
        circ = IBMQDdCircuit(1)
        circ.add_xb(0)
        xb = np.array([[1, 0], [0, -1]])
        u = circ.get_unitary()
        # check that you get correct unitary
        self.assertEquals(np.allclose(u, xb), True)
        # now check that it really has added z_i_i
        gate_counts = {"u1": 1, "barrier": 2, "id": 2}
        self.assertDictEqual(circ.get_gate_count(), gate_counts)

    def test_add_yb(self):
        circ = IBMQDdCircuit(1)
        circ.add_yb(0)
        yb = np.array([[1, 0], [0, -1]])
        u = circ.get_unitary()
        # check that you get correct unitary
        self.assertEquals(np.allclose(u, yb), True)
        # now check that it really has added z_i_i
        gate_counts = {"u1": 1, "barrier": 2, "id": 2}
        self.assertDictEqual(circ.get_gate_count(), gate_counts)

    def test_add_zb(self):
        circ = IBMQDdCircuit(1)
        circ.add_zb(0)
        zb = np.array([[0, 1], [1, 0]])
        u = circ.get_unitary()
        self.assertEquals(np.allclose(u, zb), True)

    def test_add_h(self):
        circ = IBMQDdCircuit(1)
        circ.add_h(0)
        norm = 1 / np.sqrt(2)
        h = norm * np.array([[1, 1], [1, -1]])
        u = circ.get_unitary()
        self.assertEquals(np.allclose(u, h), True)


    ##################################################
    # Encoding/ Decoding Tests
    ##################################################
    def test_encode_theta_state(self):
        circ = IBMQDdCircuit(1)
        circ.encode_theta_state(0, np.pi/2)
        circ_state = circ.get_statevector()
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_decode_theta_state(self):
        circ = IBMQDdCircuit(1)
        circ.encode_theta_state(0, np.pi/2)
        circ.decode_theta_state(0, np.pi/2)
        circ_state = circ.get_statevector()
        state = np.array([1, 0])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_encode_bell_phi_plus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_phi_plus(qubit_pairs)
        circ_state = circ.get_statevector()
        norm = 1 / np.sqrt(2)
        state = norm * np.array([1, 0, 0, 1])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_decode_bell_phi_plus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_phi_plus(qubit_pairs)
        circ.decode_bell_phi_plus(qubit_pairs, False)
        circ_state = circ.get_statevector()
        state = np.array([1, 0, 0, 0])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_encode_bell_phi_minus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_phi_minus(qubit_pairs)
        circ_state = circ.get_statevector()
        norm = 1 / np.sqrt(2)
        state = norm * np.array([1, 0, 0, -1])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_decode_bell_phi_minus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_phi_minus(qubit_pairs)
        circ.decode_bell_phi_minus(qubit_pairs, False)
        circ_state = circ.get_statevector()
        state = np.array([1, 0, 0, 0])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_encode_bell_psi_plus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_psi_plus(qubit_pairs)
        circ_state = circ.get_statevector()
        norm = 1 / np.sqrt(2)
        state = norm * np.array([0, 1, 1, 0])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_decode_bell_psi_plus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_psi_plus(qubit_pairs)
        circ.decode_bell_psi_plus(qubit_pairs, False)
        circ_state = circ.get_statevector()
        state = np.array([1, 0, 0, 0])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_encode_bell_psi_minus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_psi_minus(qubit_pairs)
        circ_state = circ.get_statevector()
        norm = 1 / np.sqrt(2)
        #state = norm * np.array([0, 1, -1, 0])
        #NOTE: I got tricked by qiskit qubit indexing (last qubit first)
        state = norm * np.array([0, -1, 1, 0])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_decode_bell_psi_minus(self):
        circ = IBMQDdCircuit(2)
        qubit_pairs = [(0, 1)]
        circ.encode_bell_psi_minus(qubit_pairs)
        circ.decode_bell_psi_minus(qubit_pairs, False)
        circ_state = circ.get_statevector()
        state = np.array([1, 0, 0, 0])
        self.assertEquals(np.allclose(circ_state, state), True)


    def test_encode_ghz_n(self):
        circ = IBMQDdCircuit(3)
        qubit_tuples = [(0, 1, 2)]
        circ.encode_ghz_n(qubit_tuples)
        circ_state = circ.get_statevector()
        norm = 1 / np.sqrt(2)
        state = norm * np.array([1, 0, 0, 0, 0, 0, 0, 1])
        self.assertEquals(np.allclose(circ_state, state), True)

    def test_decode_ghz_n(self):
        circ = IBMQDdCircuit(3)
        qubit_tuples = [(0, 1, 2)]
        circ.encode_ghz_n(qubit_tuples)
        circ.decode_ghz_n(qubit_tuples, False)
        circ_state = circ.get_statevector()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEquals(np.allclose(circ_state, state), True)


    ##################################################
    # Test DD Circuits
    ##################################################

    def test_add_free(self):
        qubits = [0, 2]
        num_ids = 7
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(3)
        for _ in range(num_ids):
            man_circ.id(qubits)
            man_circ.barrier(qubits)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        edd_circ = IBMQDdCircuit(3)
        edd_circ.add_free(qubits, num_ids)
        edd_qasm = edd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(edd_qasm, man_qasm)

    def test_add_hahn(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_hahn(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_super_hahn(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # free
            man_circ.add_free(qubits, id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_super_hahn(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_cpmg(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_cpmg(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_super_cpmg(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_super_cpmg(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_super_euler(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_super_euler(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_xy4(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # free
            man_circ.add_free(qubits, id_pad)
            # y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, 2*id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, 2*id_pad)
            # y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, 2*id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_xy4(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_cdd_n(self):
        qubits = [0, 2, 3]
        n = 1
        id_pad = 1
        # check that n = 1 works
        xy4_circ = IBMQDdCircuit(4)
        xy4_circ.add_xy4(qubits, 1, id_pad)
        xy4_qasm = xy4_circ.qasm()
        cdd_1_circ = IBMQDdCircuit(4)
        cdd_1_circ.add_cdd_n(qubits, n, id_pad)
        cdd_1_qasm = cdd_1_circ.qasm()
        self.assertEquals(xy4_qasm, cdd_1_qasm)

        # check that n = 2 works
        man_circ = IBMQDdCircuit(4)
        man_circ.add_free(qubits, id_pad)
        man_circ.add_y(qubits)
        man_circ.barrier(qubits)
        man_circ.add_free(qubits, id_pad)

        man_circ.add_xy4(qubits, 1, id_pad)

        man_circ.add_free(qubits, id_pad)
        man_circ.add_x(qubits)
        man_circ.barrier(qubits)
        man_circ.add_free(qubits, id_pad)

        man_circ.add_xy4(qubits, 1, id_pad)

        man_circ.add_free(qubits, id_pad)
        man_circ.add_y(qubits)
        man_circ.barrier(qubits)
        man_circ.add_free(qubits, id_pad)

        man_circ.add_xy4(qubits, 1, id_pad)

        man_circ.add_free(qubits, id_pad)
        man_circ.add_x(qubits)
        man_circ.barrier(qubits)
        man_circ.add_free(qubits, id_pad)

        man_circ.add_xy4(qubits, 1, id_pad)
        man_circ_qasm = man_circ.qasm()

        cdd_2_circ = IBMQDdCircuit(4)
        n = 2
        cdd_2_circ.add_cdd_n(qubits, n, id_pad)
        cdd_2_qasm = cdd_2_circ.qasm()

        self.assertEquals(cdd_2_qasm, man_circ_qasm)


    def test_add_rga2x(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # free
            man_circ.add_free(qubits, id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga2x(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga2y(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # free
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # free
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga2y(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga2z(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Zb
            man_circ.add_zb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # z_i_i
            man_circ.add_zii(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga2z(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga4(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga4(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga4p(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga4p(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga8a(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)

        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga8a(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga8c(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, 2*id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga8c(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga16a(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Zb
            man_circ.add_zb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RGA8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # z
            man_circ.add_zii(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RGA8a
            man_circ.add_rga8a(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga16a(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga16b(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RGA4'
            man_circ.add_rga4p(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # RGA4'
            man_circ.add_rga4p(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # rga4'
            man_circ.add_rga4p(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # rga4'
            man_circ.add_rga4p(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga16b(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)


    def test_add_rga32a(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RGA8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RG8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga32a(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)


    def test_add_rga32c(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RGA4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # RGA4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga4
            man_circ.add_rga4(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga4
            man_circ.add_rga4(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga32c(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga64a(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Xb
            man_circ.add_xb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8a
            man_circ.add_rga8a(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga64a(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_rga64c(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # x
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Y
            man_circ.add_y(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.id(qubits)
            man_circ.barrier(qubits)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga8c
            man_circ.add_rga8c(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga64c(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)


    def test_add_rga256a(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga64a
            man_circ.add_rga64a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga64a
            man_circ.add_rga64a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # Yb
            man_circ.add_yb(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga64a
            man_circ.add_rga64a(qubits, 1, id_pad)
            # id
            man_circ.add_free(qubits, id_pad)
            # X
            man_circ.add_x(qubits)
            man_circ.barrier(qubits)
            # id
            man_circ.add_free(qubits, id_pad)
            # rga64a
            man_circ.add_rga64a(qubits, 1, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_rga256a(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_comp_kdd(self):
        qubits = [0, 2, 3]
        id_pad = 1
        phi = 1.2
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        x_params = np.array([np.pi, 0, np.pi])
            # id
        man_circ.add_free(qubits, id_pad)
        # p1
        p1_params = x_params + np.pi/6 + phi
        man_circ.u3(*p1_params, qubits)
        man_circ.barrier(qubits)
        # id
        man_circ.add_free(qubits, 2*id_pad)
        # p2
        p2_params = x_params + phi
        man_circ.u3(*p2_params, qubits)
        man_circ.barrier(qubits)
        # id
        man_circ.add_free(qubits, 2*id_pad)
        # p3
        p3_params = x_params + np.pi/2 + phi
        man_circ.u3(*p3_params, qubits)
        man_circ.barrier(qubits)
        # id
        man_circ.add_free(qubits, 2*id_pad)
        # p2
        man_circ.u3(*p2_params, qubits)
        man_circ.barrier(qubits)
        # id
        man_circ.add_free(qubits, 2*id_pad)
        # p1
        man_circ.u3(*p1_params, qubits)
        man_circ.barrier(qubits)
        # id
        man_circ.add_free(qubits, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_comp_kdd(qubits, phi, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_kdd(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # create a manual circuit with intended sequence
        man_circ = IBMQDdCircuit(4)
        for _ in range(num_reps):
            man_circ.add_comp_kdd(qubits, np.pi/2, id_pad)
            man_circ.add_comp_kdd(qubits, 0, id_pad)
            man_circ.add_comp_kdd(qubits, np.pi/2, id_pad)
            man_circ.add_comp_kdd(qubits, 0, id_pad)
        # get qasm from manual circuit
        man_qasm = man_circ.qasm()

        # create our IBMQDdCircuit version
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_kdd(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()

        # assert that qasm strings are the same
        self.assertEquals(dd_qasm, man_qasm)

    def test_add_ur(self):
        qubits = [0, 2, 3]
        num_reps = 4
        id_pad = 1
        # just test that it runs properly
        dd_circ = IBMQDdCircuit(4)
        dd_circ.add_ur50(qubits, num_reps, id_pad)
        dd_qasm = dd_circ.qasm()
        boolval = isinstance(dd_qasm, str)
        self.assertEquals(boolval, True)

if __name__ == "__main__":
    unittest.main()
