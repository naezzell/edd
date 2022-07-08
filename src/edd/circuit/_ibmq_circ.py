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
from qiskit import (QuantumCircuit, IBMQ, execute, transpile,
                    schedule as build_schedule, Aer)
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer


class IBMQDdCircuit(QuantumCircuit):
    """
    DdCircuit is class for Dynamical Decoupling Circuits.
    Inherits all mehods/ valid data from qiskit's QuantumCircuit
    """

    def __init__(self, *args, name=None, ibmq_backend=None, mode='circ'):
        super(IBMQDdCircuit, self).__init__(*args, name=name)
        self.ibmq_backend = ibmq_backend
        self.gate_dict = None

    ##################################################
    # General Useful Utility Methods
    ##################################################
    def link_with_backend(self, ibmq_backend):
        """
        Links self with [ibmq_backend] which changes the default
        behavior of adding gates by converting all gates to
        native gates for [backend] before adding from then on.
        Also allows for get_transpiled_circ method to work.
        """
        self.ibmq_backend = ibmq_backend
        return

    def predefine_common_gates(self, mode='circ'):
        """
        Given that we're linked to ibmq_backend, it's useful
        to pre-define the common gates (i.e. X, Y, etc...)
        a single time and not have to transpile each time.
        This predefines them.
        """
        gates = ['X', 'Y', 'Z', 'Xb', 'Yb']
        gate_dict = {}
        for g in gates:
            # init circ
            if self.num_clbits == 0:
                circ = IBMQDdCircuit(self.num_qubits)
            else:
                circ = IBMQDdCircuit(self.num_qubits, self.num_clbits)
            # link with backend
            circ.link_with_backend(self.ibmq_backend)
            # make gate g for each qubit q
            for q in range(self.num_qubits):
                if g == 'X':
                    circ.add_x(q)
                    t_circ = circ.get_transpiled_circ()
                    re_circ = replace_sx_sx_with_x(t_circ)
                    key = f"{g}{q}"
                    if mode == 'circ':
                        gate_dict[key] = re_circ
                    else:
                        gate_dict[key] = re_circ.qasm()
                if g == 'Y':
                    circ.add_y(q)
                    t_circ = circ.get_transpiled_circ()
                    re_circ = replace_sx_sx_with_x(t_circ)
                    key = f"{g}{q}"
                    if mode == 'circ':
                        gate_dict[key] = re_circ
                    else:
                        gate_key[key] = re_circ.qasm()
                if g == 'Z':
                    circ.add_z(q)
                    t_circ = circ.get_transpiled_circ()
                    re_circ = replace_sx_sx_with_x(t_circ)
                    key = f"{g}{q}"
                    if mode == 'circ':
                        gate_dict[key] = re_circ
                    else:
                        gate_key[key] = re_circ.qasm()
                if g == 'Xb':
                    circ.add_xb(q)
                    t_circ = circ.get_transpiled_circ()
                    re_circ = replace_sx_sx_with_x(t_circ)
                    key = f"{g}{q}"
                    if mode == 'circ':
                        gate_dict[key] = re_circ
                    else:
                        gate_key[key] = re_circ.qasm()
                if g == 'Yb':
                    circ.add_yb(q)
                    t_circ = circ.get_transpiled_circ()
                    re_circ = replace_sx_sx_with_x(t_circ)
                    key = f"{g}{q}"
                    if mode == 'circ':
                        gate_dict[key] = re_circ
                    else:
                        gate_key[key] = re_circ.qasm()

        self.gate_dict = gate_dict
        return



    def get_transpiled_circ(self, scheduling_method=None):
        """ Returns transpiled circ provided linked with backend """
        try:
            return transpile(self, self.ibmq_backend.backend, scheduling_method=scheduling_method)
        except AttributeError:
            e = "ibmq_backend not defined. Try self.link_with_backend(backend)."
            raise AttributeError(e)

    def get_gate_count(self, trans=False, ibmq_backend=None):
        """Returns dictionary with which gates are in circuit and
        how many of them there are."""
        circ = self.copy()
        if trans:
            circ = transpile(circ, ibmq_backend.backend)
        circ_dag = circuit_to_dag(circ)
        gate_count = circ_dag.count_ops()

        return gate_count


    def get_phys_time(self, ibmq_backend='linked'):
        """Given an IBMQBackend called [backend], returns the total
        run_time of this circuit in ns. Substracts any gates in [sub]
        along with their count, i.e. if state prep is 2 u3s, then
        sub = {'u3': 2} and this would reduce phys_time by the time of
        2 u3 gates.

        NOTE: This assumes circuit is written in terms of native gates, but
        the methods we've developed do this automatically.
        """
        if ibmq_backend == 'linked':
            ibmq_backend = self.ibmq_backend

        # cast as schedule
        dt = ibmq_backend.backend.configuration().dt
        sched = build_schedule(self, ibmq_backend.backend)
        dt_duration = sched.duration
        # convert time to nano-seconds
        phys_time = dt_duration * dt * 10**9

        return phys_time


    def get_statevector(self):
        """
        Returns statevector obtained by running the circuit assuming no
        noise.
        """
        job = execute(self, Aer.get_backend('statevector_simulator'))
        state_results = job.result()
        return state_results.results[0].data.statevector

    def get_unitary(self):
        """
        Gets unitary equivalent of running the circuit with no noise.
        """
        job = execute(self, Aer.get_backend('unitary_simulator'))
        unitary_results = job.result()
        return unitary_results.results[0].data.unitary


    ##################################################
    # Methods that add gates of interest
    ##################################################
    def add_rn(self, theta, phi, alpha, qubits):
        """
        Appends R_n(\alpha) gate to [qubits].
        n = (theta, phi) is unit vector and \alpha is
        rotation angle about n-axis.
        """
        if self.ibmq_backend is None:
            self.rz(-phi, qubits)
            self.ry(-theta, qubits)
            self.rz(alpha, qubits)
            self.ry(theta, qubits)
            self.rz(phi, qubits)
        else:
            if self.num_clbits == 0:
                circ = IBMQDdCircuit(self.num_qubits)
                circ.add_rn(theta, phi, alpha, qubits)
                circ.link_with_backend(self.ibmq_backend)
                t_circ = circ.get_transpiled_circ()
                #replace_circ = replace_sx_sx_with_x(t_circ)
                #self.extend(replace_circ)
                self.extend(t_circ)
            else:
                circ = IBMQDdCircuit(self.num_qubits, self.num_clbits)
                circ.add_rn(theta, phi, alpha, qubits)
                circ.link_with_backend(self.ibmq_backend)
                t_circ = circ.get_transpiled_circ()
                replace_circ = replace_sx_sx_with_x(t_circ)
                #self.extend(replace_circ)
                self.extend(t_circ)
        return

    def add_rx(self, alpha, qubits):
        """
        Appends R_X(\alpha) gate to [qubits].
        """
        self.add_rn(np.pi/2, 0, alpha, qubits)

    def add_ry(self, alpha, qubits):
        """
        Appends R_Y(\alpha) gate to [qubits].
        """
        self.add_rn(np.pi/2, np.pi/2, alpha, qubits)

    def add_rz(self, alpha, qubits):
        """
        Appends R_Z(\alpha) gate to [qubits].
        """
        self.add_rn(0, 0, alpha, qubits)

    def add_pi_eta(self, eta, qubits):
        """
        Appends (\pi)_{\eta} gate to [qubits].
        \eta is angle from pos x axis, so pi rotation about \eta.
        """
        self.add_rn(np.pi/2, eta, np.pi, qubits)
        return

    def add_x(self, qubits):
        """
        Appends X gate to circuit on [qubits].
        """
        if self.gate_dict is not None:
            key = f"X{qubits}"
            self.extend(self.gate_dict[key])
        else:
            self.add_pi_eta(0, qubits)
        return

    def add_y(self, qubits):
        """
        Appends Y gate to circuit on [qubits].
        """
        if self.gate_dict is not None:
            key = f"Y{qubits}"
            self.extend(self.gate_dict[key])
        else:
            self.add_pi_eta(np.pi / 2, qubits)
        return

    def add_xb(self, qubits):
        """
        Appends Xb gate to circuit on [qubits].
        """
        if self.gate_dict is not None:
            key = f"Xb{qubits}"
            self.extend(self.gate_dict[key])
        else:
            self.add_pi_eta(np.pi, qubits)
        return

    def add_yb(self, qubits):
        """
        Appends Yb gate to circuit on [qubits].
        """
        if self.gate_dict is not None:
            key = f"Yb{qubits}"
            self.extend(self.gate_dict[key])
        else:
            self.add_pi_eta(3*np.pi / 2, qubits)
        return

    def add_z(self, qubits):
        """
        Appends (virutal) Z gate to circuit on [qubits].
        """
        if self.gate_dict is not None:
            key = f"Z{qubits}"
            self.extend(self.gate_dict[key])
        else:
            self.add_rn(0, 0, np.pi, qubits)
        return

    def add_zb(self, qubits):
        """
        Appends (virutal) Zb gate to circuit on [qubits].
        """
        if self.gate_dict is not None:
            key = f"Z{qubits}"
            self.extend(self.gate_dict[key])
        else:
            self.add_rn(0, 0, -np.pi, qubits)
        return

    def add_zii(self, qubits):
        """
        Appends (virtual Z gate to ciruit on [qubits] followed by
        two identities.
        """
        self.add_z(qubits)
        self.barrier(qubits)
        self.id(qubits)
        self.barrier(qubits)
        self.id(qubits)

    def add_error_mitigation_0(self, qubits):
        """
        Adds measurement error mitigation schedule for |0>,
        Id-Measure.
        """
        self.id(qubits)
        self.barrier(qubits)
        self.id(qubits)
        self.barrier(qubits)
        self.measure(qubits, qubits)
        return

    def add_error_mitigation_1(self, qubits):
        """
        Adds measurement error mitigation schedule for |1>,
        X-Measure.
        """
        self.add_x(qubits)
        self.barrier(qubits)
        self.measure(qubits, qubits)
        return

    def add_measurement(self, qubits):
        """
        Adds measurement from [qubits] to [qubits].
        """
        if mode == 'circ':
            self.measure(qubits, qubits)
        else:
            if self.num_clbits == 0:
                circ = IBMQDdCircuit(self.num_qubits)
                circ.measure(qubits, qubits)
                circ.link_with_backend(self.ibmq_backend)
                t_circ = circ.get_transpiled_circ()
                replace_circ = replace_sx_sx_with_x(t_circ)
                self.extend(replace_circ)
                #self.extend(t_circ)
            else:
                circ = IBMQDdCircuit(self.num_qubits, self.num_clbits)
                circ.add_rn(theta, phi, alpha, qubits)
                circ.link_with_backend(self.ibmq_backend)
                t_circ = circ.get_transpiled_circ()
                replace_circ = replace_sx_sx_with_x(t_circ)
                self.extend(replace_circ)
                #self.extend(t_circ)

            gate_key[key] = re_circ.qasm()

    ##################################################
    # State Prep and Decoding Methods
    ##################################################
    def encode_podal_state(self, qubits, pole=2, offset=0):
        """
        Assuming you are start in the |0> state, prepares a podal
        state--essentially a variation of Pauli-states. If offset=0,
        this method prepares the following states based on [pole]
        pole = 0 --> |0>
        pole = 1 --> |1>
        pole = 2 --> |+>
        pole = 3 --> |->
        pole = 4 --> |+i>
        pole = 5 --> |-i>
        [offset] sets a systematic tilt to all states in same direction.
        In particular, when [offset] = pi / 2, we get
        |0> --> |+>
        |1> --> |->
        |+> --> |+i>
        |-> --> |-i>
        |+i> --> |0>
        |-i> --> |1>
        """
        if pole == 0:
            self.add_ry(0 + offset, qubits)
        elif pole == 1:
            self.add_ry(np.pi + offset, qubits)
        elif pole == 2:
            self.add_ry(np.pi/2, qubits)
            self.add_rz(offset, qubits)
        elif pole == 3:
            self.add_ry(-np.pi/2 , qubits)
            self.add_rz(offset, qubits)
        elif pole == 4:
            self.add_rx(-np.pi/2 + offset, qubits)
        elif pole == 5:
            self.add_rx(np.pi/2 + offset, qubits)
        self.barrier(qubits)
        return

    def decode_podal_state(self, qubits, pole=2, offset=0):
        """
        Decodes podal state of same parameters to return back to |0>.
        """
        if pole == 0:
            self.add_ry(0 - offset, qubits)
        elif pole == 1:
            self.add_ry(-np.pi - offset, qubits)
        elif pole == 2:
            self.add_rz(-offset, qubits)
            self.add_ry(-np.pi/2, qubits)
        elif pole == 3:
            self.add_rz(-offset, qubits)
            self.add_ry(np.pi/2 , qubits)
        elif pole == 4:
            self.add_rx(np.pi/2 - offset, qubits)
        elif pole == 5:
            self.add_rx(-np.pi/2 - offset, qubits)
        self.barrier(qubits)
        return

    def encode_theta_state(self, qubits, theta):
        """
        Appends u3(theta, 0, 0) gate to [qubits] to prep each state
        into the 1/sqrt(2) * (cos(theta/2)|0> + sin(theta/2)|1>) state.
        """
        self.u3(theta, 0, 0, qubits)
        self.barrier(qubits)
        return

    def decode_theta_state(self, qubits, theta):
        """
        Appends u3(-theta, 0, 0) gate to [qubits] in order to decode the
        1/sqrt(2)*(cos(theta/2)|0> + sin(theta/2)|1>) back to |0> state.
        """
        self.barrier(qubits)
        self.u3(-1*theta, 0, 0, qubits)
        self.barrier(qubits)
        return

    def encode_bell_phi_plus(self, qubit_pairs):
        """
        Encodes the |phi^+> = |00> + |11> bell state on all (q1, q2)
        pairs in qubit_pairs starting with |00> state.

        Input: qubit_pairs -- list of tuples: [(q1, q2), (q3, q4), ...
        """
        for q_pair in qubit_pairs:
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])

        return

    def decode_bell_phi_plus(self, qubit_pairs, measure=True):
        """
        Decodes the |phi^+> = |00> + |11> bell state on all (q1, q2)
        pairs in [qubit_pairs] back into |00> state.
        If measure is True, also adds measurements on relevant qubits.
        """
        for q_pair in qubit_pairs:
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])

        if measure is True:
            # qubit to register  may not be 1-1 since not all qubits
            # can be used (i.e. odd qubit number or whatever)
            count = 0
            for q_pair in qubit_pairs:
                self.measure(q_pair[0], count)
                count += 1
                self.measure(q_pair[1], count)
                count += 1

        return

    def encode_bell_phi_minus(self, qubit_pairs):
        """
        Encodes the |phi^-> = |00> - |11> bell state on all (q1, q2)
        pairs in qubit_pairs starting with the |00> state.

        Input: qubit_pairs -- list of tuples: [(q1, q2), (q3, q4)...]
        """
        for q_pair in qubit_pairs:
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_z(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])

        return

    def decode_bell_phi_minus(self, qubit_pairs, measure=True):
        """
        Decodes the |phi^-> = |00> - |11> bell state on all (q1, q2)
        pairs in [qubit_pairs] back into |00> state.
        If measure is True, also adds measurements on relevant qubits.
        """
        for q_pair in qubit_pairs:
            self.add_z(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])

        if measure is True:
            # qubit to register may not be 1-1 since not all qubits
            # can be used (i.e. odd qubit number or whatever)
            count = 0
            for q_pair in qubit_pairs:
                self.measure(q_pair[0], count)
                count += 1
                self.measure(q_pair[1], count)
                count += 1

        return

    def encode_bell_psi_plus(self, qubit_pairs):
        """
        Encodes the |psi^+> = |01> + |10> bell state on all (q1, q2)
        pairs in qubit_pairs starting with |00> state.

        Input: qubit_pairs -- list of tuples: [(q1, q2), (q3, q4), ...
        """
        for q_pair in qubit_pairs:
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_x(q_pair[1])
            self.barrier(q_pair[0], q_pair[1])

        return

    def decode_bell_psi_plus(self, qubit_pairs, measure=True):
        """
        Decodes the |psi^+> = |01> + |10> bell state on all (q1, q2)
        pairs in [qubit_pairs] back into |00> state.
        If measure is True, also adds measurements on relevant qubits.
        """
        for q_pair in qubit_pairs:
            self.add_x(q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])

        if measure is True:
            # qubit to register may not be 1-1 since not all qubits
            # can be used (i.e. odd qubit number or whatever)
            count = 0
            for q_pair in qubit_pairs:
                self.measure(q_pair[0], count)
                count += 1
                self.measure(q_pair[1], count)
                count += 1
        return

    def encode_bell_psi_minus(self, qubit_pairs):
        """
        Encodes the |psi^-> = |01> - |10> bell state on all (q1, q2)
        pairs in qubit_pairs starting with |00> state.

        Input: qubit_pairs -- list of tuples: [(q1, q2), (q3, q4), ...
        """
        for q_pair in qubit_pairs:
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_x(q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_z(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])

        return

    def decode_bell_psi_minus(self, qubit_pairs, measure=True):
        """
        Decodes the |psi^-> = |01> - |10> bell state on all (q1, q2)
        pairs in [qubit_pairs] back into |00> state.
        If measure is True, also adds measurements on relevant qubits.
        """
        for q_pair in qubit_pairs:
            self.add_z(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])
            self.add_x(q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.cx(q_pair[0], q_pair[1])
            self.barrier(q_pair[0], q_pair[1])
            self.add_h(q_pair[0])
            self.barrier(q_pair[0], q_pair[1])

        if measure is True:
            # qubit to register may not be 1-1 since not all qubits
            # can be used (i.e. odd qubit number or whatever)
            count = 0
            for q_pair in qubit_pairs:
                self.measure(q_pair[0], count)
                count += 1
                self.measure(q_pair[1], count)
                count += 1

        return

    def encode_ghz_n(self, qubit_tuples):
        """
        Encodes the |0...0> + |1...1> GHZ state on all (q1, ..., qn)
        n-tuples in qubit_tuples starting with |0...0> state.
        NOTE: The assumption is that CX should be applied on pairs
        such as (q1, q2), (q2, q3), etc..., so obviously this doesn't
        apply to all hardware natively. In general, must be more manual.

        Input: qubit_tuples -- list of tuples of the form
        [(q1, ..., qn), (qn+1, ..., qn+n), ...]
        """
        for q_tup in qubit_tuples:
            self.add_h(q_tup[0])
            n = len(q_tup)
            for j in range(n-1):
                self.barrier([q for q in q_tup])
                self.cx(q_tup[j], q_tup[j+1])
            self.barrier([q for q in q_tup])

        return

    def decode_ghz_n(self, qubit_tuples, measure=True):
        """
        Decodes the |0...0> + |1...1> GHZ state on all (q1, ..., qn)
        n-tuples in qubit_tuples starting with |0...0> state.
        NOTE: The assumption is that CX should be applied on pairs
        such as (q1, q2), (q2, q3), etc..., so obviously this doesn't
        apply to all hardware natively. In general, must be more manual.

        Input: qubit_tuples -- list of tuples of the form
        [(q1, ..., qn), (qn+1, ..., qn+n), ...]
        """
        for q_tup in qubit_tuples:
            n = len(q_tup)
            for j in range(n-1):
                self.barrier([q for q in q_tup])
                # add cx in opposite order of encoding
                # which means running list backwards
                self.cx(q_tup[n-j-2], q_tup[n-j-1])
            self.barrier([q for q in q_tup])
            # add the hadarmard which used to be at the beginning
            self.add_h(q_tup[0])
            self.barrier([q for q in q_tup])

        if measure is True:
            # qubit to register may not be 1-1 since not all qubits
            # can be used (i.e. odd qubit number or whatever)
            count = 0
            for q_tup in qubit_tuples:
                for q in q_tup:
                    self.measure(q, count)
                    count += 1

        return

    ##################################################
    # DD Circuit Methods
    ##################################################
    def add_free(self, qubits, num_ids):
        """
        Appends free evolution to [qubits] over [n] identity gates.
        """
        for _ in range(num_ids):
            self.id(qubits)
            self.barrier(qubits)
        return


    def add_pause(self, qubits, time, unit='dt'):
        '''adds pause (id gate) for [time]dt on [qubits]'''
        if time != 0:
            self.delay(time, qubits, unit=unit)
            self.barrier(qubits)

        return

    ######################################################################
    # Basic DD sequences (XY4) and slight augmentations
    ######################################################################

    def add_hahn(self, qubits, num_reps=1, tau=0):
        """
        Appends Hahn echo sequence to [qubits] with [idpad] ids between pulses.
        -X- where - is [idpad] free evo.

        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_super_hahn(self, qubits, num_reps=1, tau=0):
        """
        Appends symmetric eulerian super-cycle Hahn sequence to
        [qubits] with [idpad] ids between pulses, i.e.
        -X--Xb- where - is [idpad] freee evo.
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # Xb (aka -X)
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_cpmg(self, qubits, num_reps=1, tau=0):
        """
        Appends cpmg sequence to [qubits] with [idpad] ids between pulses
        -X--X- where - is [idpad] free evo.
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_super_cpmg(self, qubits, num_reps=1, tau=0):
        """
        Appends symmetric eulerian super-cycle cpmg sequence to [qubits]
        with [idpad] ids between pulses, i.e.
        -X--X--Xb--Xb-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_super_euler(self, qubits, num_reps=1, tau=0):
        """
        Appends symmetric eulerian super-cycle sequence to [qubits]
        with [idpad] ids between pulses, i.e.
        -X--Y--X--Y--Y--X--Y--X--Xb--Yb--Xb--Yb--Yb--Xb--Yb--Xb-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_xy4(self, qubits, num_reps=1, tau=0):
        """
        Appends xy4 sequence to [qubits] for [num_reps] times with [tau]
        id gates between each DD pulse gate.
        -Y--X--Y--X- where - is [idpad] free evo.
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2 * idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)

        return

    def add_cdd_n(self, qubits, n, tau=0):
        if n == 1:
            self.add_xy4(qubits, 1, tau)
        else:
            # add free.Y.free motif
            self.add_pause(qubits, tau)
            self.add_y(qubits)
            self.barrier(qubits)
            self.add_pause(qubits, tau)
            # add U^(n-1)
            self.add_cdd_n(qubits, n-1, tau)
            # add free.X.free motif
            self.add_pause(qubits, tau)
            self.add_x(qubits)
            self.barrier(qubits)
            self.add_pause(qubits, tau)
            # add U^(n-1)
            self.add_cdd_n(qubits, n-1, tau)
            # add free.Y.free motif
            self.add_pause(qubits, tau)
            self.add_y(qubits)
            self.barrier(qubits)
            self.add_pause(qubits, tau)
            # add U^(n-1)
            self.add_cdd_n(qubits, n-1, tau)
            # add free.X.free motif
            self.add_pause(qubits, tau)
            self.add_x(qubits)
            self.barrier(qubits)
            self.add_pause(qubits, tau)
            # add U^(n-1)
            self.add_cdd_n(qubits, n-1, tau)
        return

    def add_cdd2(self, qubits, num_reps=1, tau=0):
        """
        Appends CDD_2 sequence to [qubits] for [num_reps] times with [tau]
        in between each DD pulse gate.
        """
        for _ in range(num_reps):
            self.add_cdd_n(qubits, 2, tau)

        return

    def add_cdd3(self, qubits, num_reps=1, tau=0):
        """
        Appends CDD_3 sequence to [qubits] for [num_reps] times with [tau]
        in between each DD pulse sequence.
        """
        for _ in range(num_reps):
            self.add_cdd_n(qubits, 3, tau)

        return

    def add_cdd4(self, qubits, num_reps=1, tau=0):
        """
        Appends CDD_4 sequence to [qubits] for [num_reps] times with [tau]
        in between each DD pulse sequence.
        """
        for _ in range(num_reps):
            self.add_cdd_n(qubits, 4, tau)

        return

    def add_cdd5(self, qubits, num_reps=1, tau=0):
        """
        Appends CDD_5 sequence to [qubits] for [num_reps] times with [tau]
        in between each DD pulse sequence.
        """
        for _ in range(num_reps):
            self.add_cdd_n(qubits, 5, tau)

        return

    ######################################################################
    # RGA DD Sequences
    ######################################################################

    def add_rga2x(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA2 sequence to [qubits] for [num_reps] times with [tau]
        in between each DD pulse gate.
        Sequence: Xb X
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        We implement Xb = Z as vZ.I.I.
        -Xb--X-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)

        return

    def add_rga2y(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA2 sequence to [qubits] for [num_reps] times with [tau]
        id_gates beteen each DD pulse gate.
        Sequence: Yb Y
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        -Yb--Y-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)

        return

    def add_rga2z(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA2z sequence to [qubits] for [num_reps] times with [tau]
        id gates between each DD pulse gate.
        Sequence: Zb Z
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        -Zb--Z-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Zb
            self.add_zb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Z
            self.add_zii(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_rga4(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA4 sequence to [qubits] for [num_reps] times with [tau]
        identity gates between each DD pulse gate.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        -Yb--X--Yb--X-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_rga4p(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA4' to [qubits] for [num_reps] times with [tau] id
        gates each DD pulse gate.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        -Yb--Xb--Yb--X-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)

        return

    def add_rga8a(self, qubits, num_reps=1, tau=0):
        """
        appends RGA8a to [qubits] with [tau] pause between pulses
        -X--Yb--X--Yb--Y--Xb--Y--Xb-
        Note: This is a more symmetrized version suggested by Greg that
        is not in the original paper.
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_rga8c(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA8c to [qubits] for [num_reps] times with [tau] id_gates
        between each DD pulse gate.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        -X--Y--X--Y--Y--X--Y--X-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (2*idpad)
            self.add_pause(qubits, 2*tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
        return

    def add_rga16a(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA16a to [qubits] for [num_reps] times with [tau] id
        gates each DD pulse gate.
        NOTE: We choose P2 = X and P1 = Y for no particular reason.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        Sequence: -Zb-(RGA8a)-Z-(RGA8a)
        """
        for _ in range(num_reps):
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Zb
            self.add_zb(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # RGA8a
            self.add_rga8a(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Z
            self.add_zii(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # RGA8a
            self.add_rga8a(qubits, 1, tau)
        return

    def add_rga16b(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA16b'' to [qubits] for [num_reps] times with [tau] id
        gates each DD pulse gate.
        NOTE: We choose P2 = X and P1 = Y for no particular reason.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        RGA4'[RGA4'], where RGA4' is
        -Yb--Xb--Yb--X-
        """
        for _ in range(num_reps):
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4'
            self.add_rga4p(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4'
            self.add_rga4p(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4'
            self.add_rga4p(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4'
            self.add_rga4p(qubits, 1, tau)

        return


    def add_rga32a(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA32a to [qubits] for [num_reps] times with [tau] id
        gates each DD pulse gate.
        NOTE: We choose P2 = X and P1 = Y for no particular reason.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        RGA4[RGA8a] where RGA4 is given by
        -Yb--X--Yb--X-
        """
        for _ in range(num_reps):
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)

        return

    def add_rga32c(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA32c to [qubits] for [num_reps] times with [tau] id
        gates each DD pulse gate.
        NOTE: We choose P2 = X and P1 = Y for no particular reason.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        RGA8c[RGA4], where RGA8c sequence is given by
        -X--Y--X--Y--Y--X--Y--X-
        """
        for _ in range(num_reps):
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (tau)
            self.add_pause(qubits, tau)
            # rga4
            self.add_rga4(qubits, 1, tau)
        return

    def add_rga64a(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA64a to [qubits] for [num_reps] times with [tau] id_gates
        between each DD pulse gate.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        RGA8a[RGA8a] where the RGA8a sequence is
        -X--Yb--X--Yb--Y--Xb--Y--Xb-
        """
        for _ in range(num_reps):
            # add free (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # Yb 
            self.add_yb(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rg8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # Xb
            self.add_xb(qubits)
            self.barrier(qubits)
            # add free (idpad)
            self.add_pause(qubits, tau)
            # rga8a
            self.add_rga8a(qubits, 1, tau)
        return


    def add_rga64c(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA64c to [qubits] for [num_reps] times with [tau] id_gates
        between each DD pulse gate.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        Sequence: RGA8c[RGA8c] where the RGA8c sequence is
        -X--Y--X--Y--Y--X--Y--X-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Y
            self.add_y(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga8c
            self.add_rga8c(qubits, 1, tau)
        return

    def add_rga256a(self, qubits, num_reps=1, tau=0):
        """
        Appends RGA256a sequence to [qubits] for [num_reps] times with [tau]
        identity gates between each DD pulse gate.
        NOTE: Xb means X-bar which is pi phase flip on all axes of Y gate.
        RGA4[RGA64a] where the RGA4 sequence is given by
        -Yb--X--Yb--X-
        """
        for _ in range(num_reps):
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga64a
            self.add_rga64a(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga64a
            self.add_rga64a(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # Yb
            self.add_yb(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga64a
            self.add_rga64a(qubits, 1, tau)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # X
            self.add_x(qubits)
            self.barrier(qubits)
            # free evo (idpad)
            self.add_pause(qubits, tau)
            # rga64a
            self.add_rga64a(qubits, 1, tau)
        return

    ##################################################
    # KDD Sequence
    ##################################################
    # define the KDD_\phi composite pulse
    def add_comp_kdd(self, qubits, phi, tau=0):
        """
        Appends the KDD_\phi composite pulse which consists of the following:
        - (\pi)_{\pi/6 + \phi} -- (\pi)_{\phi} -- (\pi)_{\pi/2 + \phi}
        -- (\pi)_{\phi} -- (\pi)_{\pi/6 + \phi} -
        where - is free evo for [tau] id gates and (\pi)_{\phi} is
        a Pi rotation pulse about the \phi axis where \phi is angle between
        positive x-axis and point in x-y axis, counter-clockwise.
        """
        # free evo (idpad)
        self.add_pause(qubits, tau)
        # p1 pulse
        p1 = np.pi/6 + phi
        self.add_pi_eta(p1, qubits)
        self.barrier(qubits)
        # free evo (2*idpad)
        self.add_pause(qubits, 2*tau)
        # p2 pulse
        p2 = phi
        self.add_pi_eta(p2, qubits)
        self.barrier(qubits)
        # free evo (2*idpad)
        self.add_pause(qubits, 2*tau)
        # p3 + \phi pulse
        p3 = np.pi/2 + phi
        self.add_pi_eta(p3, qubits)
        self.barrier(qubits)
        # free evo (2*idpad)
        self.add_pause(qubits, 2*tau)
        # p2 pulse
        self.add_pi_eta(p2, qubits)
        self.barrier(qubits)
        # free evo (2*idpad)
        self.add_pause(qubits, 2*tau)
        # p1 pulse
        self.add_pi_eta(p1, qubits)
        self.barrier(qubits)
        # free evo (idpad)
        self.add_pause(qubits, tau)

        return

    def add_kdd(self, qubits, num_reps=1, tau=0):
        """
        Appends KDD sequence to [qubits] for [num_reps] times with [tau]
        pause between each DD pulse.

        (KDD_pi/2)(KDD_0)(KDD_pi/2)(KDD_0) = "Y.X.Y.X" with composite pulses
        """
        for _ in range(num_reps):
            self.add_comp_kdd(qubits, np.pi/2, tau)
            self.add_comp_kdd(qubits, 0, tau)
            self.add_comp_kdd(qubits, np.pi/2 , tau)
            self.add_comp_kdd(qubits, 0, tau)

    ##################################################
    # Universal Robust (UR) DD Sequence
    ##################################################
    def add_ur(self, qubits, n, num_reps=1, tau=0):
        """
        Appends the UR_n sequence to [qubits] for [num_reps] times with
        [idpad] pause between each DD pulse.
        The sequence consists of [pi]_\phi_k rotations where phi_k is
        rotation axis (standard phi in x-y plane polar coords) in:
        "Arbitrarily Accurate Pulse Sequences for Robust" DD by
        Genov, Schraft, Vitanov, and Halfmann in PRL. The
        get_urdd_phis() function below should also make it clear.
        """
        # get list of pulses from unique phi information
        phi_list, _, _ = get_urdd_phis(n)
        for _ in range(num_reps):
            for phi in phi_list:
                # free evo
                self.add_pause(qubits, tau)
                # pulse
                self.add_pi_eta(phi, qubits)
                self.barrier(qubits)
                # free evo
                self.add_pause(qubits, tau)
        return

    def add_ur6(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 6, num_reps, tau)
        return

    def add_ur10(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 10, num_reps, tau)
        return

    def add_ur20(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 20, num_reps, tau)
        return

    def add_ur50(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 50, num_reps, tau)
        return

    def add_ur100(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 100, num_reps, tau)
        return

    def add_ur200(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 200, num_reps, tau)
        return

    def add_ur300(self, qubits, num_reps=1, tau=0):
        """
        Just a wrapper to run experiments more easily.
        """
        self.add_ur(qubits, 300, num_reps, tau)
        return

##################################################
# Utilities for this class (put in this file for now because there are
# so few. If I start collecting more--I'll add seperate file.
##################################################
def replace_sx_sx_with_x(circ):
    """
    Given a [circ], replace sx-sx motif with
    a single x gate.
    """
    qasm_string = circ.qasm()
    new_qasm_string = ""
    prev_was_sx = False
    replace_sx_idxs = []

    # find and keep track of sx-sx motifs
    for idx, inst in enumerate(qasm_string.split('\n')):
        if 'sx' in inst:
            if prev_was_sx == True:
                replace_sx_idxs.append(idx - 1)
            else:
                prev_was_sx = True
                new_qasm_string += inst + "\n"
        else:
            new_qasm_string += inst + "\n"
            prev_was_sx == False

    # replace sx values with x if applicable
    if replace_sx_idxs != []:
        qasm_split = new_qasm_string.split('\n')
        for idx in replace_sx_idxs:
            qasm_split[idx] = "x" + qasm_split[idx][2:]
        new_qasm_string = "\n".join(qasm_split)

    # get new circ with old global phase
    new_circ = QuantumCircuit.from_qasm_str(new_qasm_string)
    new_circ.global_phase = circ.global_phase

    return new_circ


def get_harr_random_u3_params():
    """
    Generates the theta, phi, and lambda parameters for a harr-random
    unitary in u3(theta, phi, lambda) form
    """
    decomp = OneQubitEulerDecomposer(basis='U3')
    harr_random = random_unitary(2).data

    return decomp.angles(harr_random)

#########################
# URDD Functions
#########################
def get_urdd_phis(n):
    ''' Gets \phi_k values for n pulse UR sequence'''
    if n % 2 == 1:
        raise ValueError("n must be even")
    elif n < 4:
        raise ValueError("n must be >= 4")
    # phi1 = 0 by convention
    phis = [0]

    # get capital Phi value
    if n % 4 == 0:
        m = int(n / 4)
        big_phi = np.pi / m
    else:
        m = int((n - 2) / 4)
        big_phi = (2 * m * np.pi) / (2 * m + 1)

    # keep track of unique phi added; we choose phi2 = big_phi by convention--
    # only real requirement is (n * big_phi = 2pi * j for j int)
    unique_phi = [0, big_phi]
    # map each phi in [phis] to location (by index) of corresponding [unique_phi]
    phi_indices = [0, 1]
    # populate remaining phi values
    for k in range(3, n+1):
        phi_k = (k * (k - 1) * big_phi) / 2
        # values only matter modulo 2 pi
        phi_k = (phi_k) % (2 * np.pi)
        if np.isclose(phi_k, 0):
            phi_k = 0
        elif np.isclose(phi_k, 2 * np.pi):
            phi_k = 0

        added_new = False
        for idx, u_phi in enumerate(unique_phi):
            if np.isclose(u_phi, phi_k, atol=0.001):
                added_new = True
                phi_indices.append(idx)

        if added_new == False:
            unique_phi.append(phi_k)
            phi_indices.append(len(unique_phi)-1)

    # construct phi list
    phis = []
    for idx in phi_indices:
        phis.append(unique_phi[idx])

    return (phis, unique_phi, phi_indices)
