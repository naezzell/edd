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
from edd.circuit import IBMQDdCircuit
from qiskit.pulse import Waveform, Schedule, Delay, Play, DriveChannel, Drag
from qiskit import transpile, QuantumCircuit, schedule as build_schedule

class IBMQDdSchedule(Schedule):
    """
    Our in-house Schedule class inhereting from
    qiskit Pulse
    """

    def __init__(self, ibmq_backend, basis_version, name=''):
        super(IBMQDdSchedule, self).__init__(name=name)

        self.ibmq_backend = ibmq_backend
        # for now, we just create basis based on 1st qubit
        #TODO: possibly change this behavior when access to multi-qubit pulse
        self.basis_version = basis_version
        n = ibmq_backend.get_number_qubits()
        if basis_version == 'x_basis':
            qubit_bases = []
            for q in range(n):
                basis = create_from_x_basis(ibmq_backend, q)
                qubit_bases.append(basis)
            self.basis = {}
            for basis in qubit_bases:
                for key, value in basis.items():
                    self.basis[key] = value
        elif basis_version == 'g_basis':
            qubit_bases = []
            for q in range(n):
                basis = create_from_greg_basis(ibmq_backend, q)
                qubit_bases.append(basis)
            self.basis = {}
            for basis in qubit_bases:
                for key, value in basis.items():
                    self.basis[key] = value
        elif basis_version == 'c_basis':
            qubit_bases = []
            for q in range(n):
                basis = create_from_circ_basis(ibmq_backend, n, q)
                qubit_bases.append(basis)
            self.basis = {}
            for basis in qubit_bases:
                for key, value in basis.items():
                    self.basis[key] = value
        # get normalized time unit of backend object (not our personal one)
        self.dt = ibmq_backend.backend.configuration().dt
        self.sched = Schedule(name=name)
        self.tot_delay = 0
        return

    def get_schedule(self):
        return self.sched

    def get_pulse_list(self):
        return list(self.sched._instructions())

    def get_tot_delay(self):
        return self.tot_delay

    def get_pulse_names(self):
        pulse_list = self.get_pulse_list()
        pulse_names = []
        for pulse in pulse_list:
            pulse_names.append(pulse[1].name)
        return pulse_names

    def get_duration(self):
        return self.sched.duration

    def get_phys_time(self):
        '''returns physical time of schedule in ns'''
        return dt_time_to_ns(self.get_duration(), self.dt)

    def draw(self):
        return self.sched.draw()

    def reset(self):
        '''removes all pulses added so far'''
        self.sched = Schedule(name=self.sched.name)
        return self.sched

    ##################################################
    # Add simple pulses (X, Y, measurement, ...)
    ##################################################

    # identity pulse of same length as X/Y
    def add_id(self, qubits):
        '''adds id pulse to [qubits] of same length as X/Y pulses'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            if self.basis_version == 'c_basis':
                self.sched += self.basis[f'I_{qubit}']
            else:
                self.sched += Play(self.basis[f'I_{qubit}'], DriveChannel(qubit))
        return

    # X pulses
    def add_x(self, qubits):
        '''adds X pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            if self.basis_version == 'c_basis':
                self.sched += self.basis[f'X_{qubit}']
            else:
                self.sched += Play(self.basis[f'X_{qubit}'], DriveChannel(qubit))
        return

    def add_xb(self, qubits):
        '''adds Xb pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            if self.basis_version == 'c_basis':
                self.sched += self.basis[f'Xb_{qubit}']
            else:
                self.sched += Play(self.basis[f'Xb_{qubit}'], DriveChannel(qubit))
        return

    def add_x90(self, qubits):
        '''adds X90 pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            self.sched += Play(self.basis[f'X2_{qubit}'], DriveChannel(qubit))
        return

    def add_x90b(self, qubits):
        '''adds X90b pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            self.sched += Play(self.basis[f'X2b_{qubit}'], DriveChannel(qubit))
        return

    def add_y(self, qubits):
        '''adds Y pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            if self.basis_version == 'c_basis':
                self.sched += self.basis[f'Y_{qubit}']
            else:
                self.sched += Play(self.basis[f'Y_{qubit}'], DriveChannel(qubit))
        return

    def add_symy(self, qubits):
        '''adds symmetric Y (circuit) pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            if self.basis_version == 'c_basis':
                self.sched += self.basis[f'symY_{qubit}']
            else:
                raise ValueError("Only works for circuit basis.")
        return

    def add_yb(self, qubits):
        '''adds Yb pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            if self.basis_version == 'c_basis':
                self.sched += self.basis[f'Yb_{qubit}']
            else:
                self.sched += Play(self.basis[f'Yb_{qubit}'], DriveChannel(qubit))
        return

    def add_y90(self, qubits):
        '''adds Y90 pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            self.sched += Play(self.basis[f'Y2_{qubit}'], DriveChannel(qubit))
        return

    def add_y90b(self, qubits):
        '''adds Y pulse to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            self.sched += Play(self.basis[f'Y2b_{qubit}'], DriveChannel(qubit))
        return

    def add_z(self, qubits):
        '''adds Z pulse to [qubits].'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            self.sched += self.basis[f'Z_{qubit}']
        return

    def add_zb(self, qubits):
        '''adds Zb pulse to [qubits].'''
        if type(qubits) is int:
            qubits = [qubits]
        for qubit in qubits:
            self.sched += self.basis[f'Zb_{qubit}']
        return

    def add_zi(self, qubits):
        '''adds Z then I to [qubits]'''
        self.add_z(qubits)
        self.add_id(qubits)
        return

    def add_old_measurement(self, qubit):
        """
        Adds measurement to [qubit] along with any ancillary qubits (that is,
        not all qubits can be measured solo due to hardware constraints.)
        #TODO: allow adding measurements to more than one qubit at once
        """
        backend_defaults = self.ibmq_backend.backend.defaults()
        backend_config = self.ibmq_backend.backend.configuration()
        meas_map_idx = None
        for i, measure_group in enumerate(backend_config.meas_map):
            if qubit in measure_group:
                meas_map_idx = i
                break
            assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
        # now create measurement with measurement indices
        inst_sched_map = backend_defaults.instruction_schedule_map
        measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

        # add measurement to schedule (shift to end with <<)
        self.sched += measure << self.get_duration()
        return

    def add_measurement(self, qubits, cbits):
        """
        Adds measurement to [qubit].
        """
        if type(qubits) == int:
            qubits = [qubits]
        if type(cbits) == int:
            cbits = [cbits]
        # first build the measurement circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n, len(cbits))
        for i in range(len(qubits)):
            circ.measure(qubits[i], cbits[i])
        t_circ = transpile(circ, self.ibmq_backend.backend)
        # make Pulse schedule of measurement
        measure = build_schedule(t_circ, self.ibmq_backend.backend)
        # append measurement to schedule at the end
        self.sched += measure << self.get_duration()

        return

    def add_measurement(self, qubits, cbits):
        """
        Adds measurement to [qubit].
        """
        if type(qubits) == int:
            qubits = [qubits]
        if type(cbits) == int:
            cbits = [cbits]
        # first build the measurement circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n, len(cbits))
        for i in range(len(qubits)):
            circ.measure(qubits[i], cbits[i])
        t_circ = transpile(circ, self.ibmq_backend.backend)
        # make Pulse schedule of measurement
        measure = build_schedule(t_circ, self.ibmq_backend.backend)
        # append measurement to schedule at the end
        self.sched += measure << self.get_duration()

        return

    def add_error_mitigation_0(self, qubit):
        """
        Adds measurement error mitigation schedule for |0>,
        Id-Measure.
        """
        self.add_id(qubit)
        self.add_measurement(qubit, 0)
        return

    def add_error_mitigation_1(self, qubit):
        """
        Adds measurement error mitigation schedule for |1>,
        X-Measure.
        """
        self.add_x(qubit)
        self.add_measurement(qubit, 0)
        return

    ##################################################
    # Native Gate Pulses
    ##################################################
    def add_u3(self, qubits, t, p, l):
        '''adds u3(t, p, l) to [qubits]'''
        if type(qubits) is int:
            qubits = [qubits]
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        circ.u(t, p, l, qubits)
        t_circ = transpile(circ, self.ibmq_backend.backend)
        u3sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += u3sched

        return

    def add_circDD_part1(self, gidx, qubits):
        """
        Adds state prep and Q1/Q2 swap.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # prepare state
        circ.unitary(g1, [q0, q1], label="U_1")
        circ.unitary(g2, [q1, q2], label="U_2")
        circ.barrier([q0, q1, q2])
        # swap operation between Q1 and Q2
        circ.cnot(q2, q1)
        circ.cnot(q1, q2)
        circ.cnot(q2, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return


    def add_circDD_part2(self, qubits):
        """
        Adds entangling operation on Q0 and Q1
        """
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        circ.h(q0)
        circ.cnot(q0, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return


    def add_circDD_part3(self, qubits):
        """
        Adds dientangling operation on Q0 and Q1
        """
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        circ.cnot(q0, q1)
        circ.h(q0)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return


    def add_circDD_part4(self, qubits):
        """
        Swaps Q1 and Q2 back to original position.
        """
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        circ.cnot(q2, q1)
        circ.cnot(q1, q2)
        circ.cnot(q2, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return


    def add_circDD_part5(self, gidx, qubits):
        """
        Decodes input state.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # prepare state
        circ.unitary(g2dg, [q1, q2], label="U_2^{dg}")
        circ.unitary(g1dg, [q0, q1], label="U_1^{dg}")
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched
        cbits = list(range(len(qubits)))
        self.add_measurement(qubits, cbits)

        return

    def add_circDDv2_part1(self, gidx, qubits):
        """
        Adds first state prep on Q0/Q1.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # prepare state on Q0/Q1
        circ.unitary(g1, [q0, q1], label="U_1")
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return

    def add_circDDv2_part2(self, gidx, qubits):
        """
        Adds seconds state prep on Q1/Q2.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # prepare state on Q1/Q2
        circ.unitary(g2, [q1, q2], label="U_2")
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return

    def add_circDDv2_part3(self, qubits):
        """
        Adds SWAP between Q1 and Q2.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # swap operation between Q1 and Q2
        circ.cnot(q2, q1)
        circ.cnot(q1, q2)
        circ.cnot(q2, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return

    def add_circDDv2_part4(self, qubits):
        """
        Adds entangling operation on Q0 and Q1
        """
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        circ.h(q0)
        circ.cnot(q0, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return

    def add_circDDv2_part5(self, qubits):
        """
        Adds disentangling operation on Q0/Q1.
        """
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        circ.h(q0)
        circ.cnot(q0, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return

    def add_circDDv2_part6(self, qubits):
        """
        Swaps Q1 and Q2 back to original position.
        """
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        circ.cnot(q2, q1)
        circ.cnot(q1, q2)
        circ.cnot(q2, q1)
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched

        return

    def add_circDDv2_part7(self, gidx, qubits):
        """
        Decodes input state on Q1/Q2.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # prepare state
        circ.unitary(g2dg, [q1, q2], label="U_2^{dg}")
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched
        cbits = list(range(len(qubits)))
        self.add_measurement(qubits, cbits)

        return

    def add_circDDv2_part8(self, gidx, qubits):
        """
        Decodes input state on Q0/Q1.
        """
        # extract state prep gates
        big_gate_list = np.load("2q_gate_list.npy", allow_pickle=True)
        gates = big_gate_list[gidx]
        g1, g1dg, g2, g2dg, _, _ = gates
        # form circuit
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        q0, q1, q2 = qubits
        # prepare state
        circ.unitary(g1dg, [q0, q1], label="U_1^{dg}")
        circ.barrier([q0, q1, q2])
        # transpile the circuit, convert to sched, and append
        t_circ = transpile(circ, self.ibmq_backend.backend)
        circ_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += circ_sched
        cbits = list(range(len(qubits)))
        self.add_measurement(qubits, cbits)

        return



    ##################################################
    # Encoding and Decoding of States
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
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        if pole == 0:
            circ.ry(0 + offset, qubits)
        elif pole == 1:
            circ.ry(np.pi + offset, qubits)
        elif pole == 2:
            circ.ry(np.pi/2, qubits)
            circ.rz(offset, qubits)
        elif pole == 3:
            circ.ry(-np.pi/2 , qubits)
            circ.rz(offset, qubits)
        elif pole == 4:
            circ.rx(-np.pi/2 + offset, qubits)
        elif pole == 5:
            circ.rx(np.pi/2 + offset, qubits)

        t_circ = transpile(circ, self.ibmq_backend.backend)
        pode_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += pode_sched
        return

    def decode_podal_state(self, qubits, pole=2, offset=0):
        """
        Decodes podal state of same parameters to return back to |0>.
        """
        n = self.ibmq_backend.get_number_qubits()
        circ = QuantumCircuit(n)
        if pole == 0:
            circ.ry(0 - offset, qubits)
        elif pole == 1:
            circ.ry(-np.pi - offset, qubits)
        elif pole == 2:
            circ.rz(-offset, qubits)
            circ.ry(-np.pi/2, qubits)
        elif pole == 3:
            circ.rz(-offset, qubits)
            circ.ry(np.pi/2 , qubits)
        elif pole == 4:
            circ.rx(np.pi/2 - offset, qubits)
        elif pole == 5:
            circ.rx(-np.pi/2 - offset, qubits)

        t_circ = transpile(circ, self.ibmq_backend.backend)
        pode_sched = build_schedule(t_circ, self.ibmq_backend.backend)
        self.sched += pode_sched
        return

    def encode_theta_state(self, qubits, t):
        """
        Adds U3(t, 0, 0) to [qubits] which takes |0> state to
        cos(t/2)|0> + sin(t/2)|1> state.
        """
        self.add_u3(qubits, t, 0, 0)

    def decode_theta_state(self, qubits, t):
        """
        Adds U3(-t, 0, 0) to [qubits] which takes the
        cos(t/2)|0> + sin(t/2)|1> back to |0> state.
        """
        self.add_u3(qubits, -t, 0, 0)
    ##################################################
    # DD Sequences
    ##################################################
    def add_pause(self, qubit, time):
        '''adds pause (id gate) for [time]dt on [qubits]'''
        if time != 0:
            self.sched += Delay(time, DriveChannel(qubit))
            self.tot_delay += time
        return

    # diagnose sum of delays not respecting acquire constraint
    def diagnose_acquire_constraint(self, qubit):
        """
        Adds additional pause at the end (before measurement)
        if necessary to ensure acquire constraint is satisfied.
        """
        aa = self.ibmq_backend.get_acquire_alignment()
        r = self.tot_delay % aa
        if r != 0:
            #print(f"Diagnosed with {aa - r}dt delay.")
            self.add_pause(qubit, aa - r)

        return r




    def add_free(self, qubit, num_reps=1, du=0, sym=False):
        """
        Adds [num_reps] identity gates with [du] pause
        between them (redundanet but for consistency)
        onto [qubits]. [sym] is also not used but there
        for consistency.
        """
        for _ in range(num_reps):
            # add I
            self.add_id(qubit)
            # add delay
            self.add_pause(qubit, du)


    def add_hahn(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of Hahn sequence to [qubits] back-to-back.
        * sym=False: X-
        * sym=True: -X-
        where - is delay of duration [d]
        """
        for _ in range(num_reps):
            if sym is False:
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    def add_purex(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CPMG sequence to [qubit].
        * sym=False: X-X-
        * sym=True: -X=X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return

    def add_purey(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CPMG sequence to [qubit].
        * sym=False: Y-Y-
        * sym=True: -Y=Y-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    def add_sympurey(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CPMG sequence to [qubit].
        * sym=False: Y-Y-
        * sym=True: -Y=Y-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # Y
                self.add_symy(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_symy(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_symy(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_symy(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    def add_xy4(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of XY4 sequence to [qubit].
        * sym=False: Y-X-
        * sym=True: -X=X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    def add_cdd_n(self, n, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CDD_n sequence to [qubit].
        * sym=False
        Y(CDD_{n-1})X(CDD_{n-1})Y(CDD_{n-1})X(CDD_{n-1})
        * sym=True
        -Y(CDD_{n-1})X(CDD_{n-1})Y(CDD_{n-1})X(CDD_{n-1})-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d

        # step 1: create cdd schedule as a str
        cdd_str = simplify_cdd_str(make_cdd_n_str(n), sym)

        # step 2: add it num_reps times
        for _ in range(num_reps):
            if sym is True:
                # free evo (d)
                self.add_pause(qubit, d)
            # add relevant instructions from cdd_str
            for loc, inst in enumerate(cdd_str):
                if inst == 'X':
                    self.add_x(qubit)
                elif inst == 'Y':
                    self.add_y(qubit)
                elif inst == 'Z':
                    self.add_z(qubit)
                elif inst == 'f':
                    # second condition ensures we add (d) pause at end always
                    if sym is True and (loc < len(cdd_str) - 2):
                        self.add_pause(qubit, ds)
                    else:
                        self.add_pause(qubit, d)

        return


    def add_cdd2(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CDD_2 sequence to [qubit].
        * sym=False
        Y(CDD_{1})X(CDD_{1})Y(CDD_{1})X(CDD_{1})
        * sym=True
        -Y(CDD_{1})X(CDD_{1})Y(CDD_{1})X(CDD_{1})-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        self.add_cdd_n(2, qubit, num_reps, d, sym)
        return


    def add_cdd3(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CDD_3 sequence to [qubit].
        * sym=False
        Y(CDD_{2})X(CDD_{2})Y(CDD_{2})X(CDD_{2})
        * sym=True
        -Y(CDD_{2})X(CDD_{2})Y(CDD_{2})X(CDD_{2})-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        self.add_cdd_n(3, qubit, num_reps, d, sym)
        return


    def add_cdd4(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CDD_4 sequence to [qubit].
        * sym=False
        Y(CDD_{3})X(CDD_{3})Y(CDD_{3})X(CDD_{3})
        * sym=True
        -Y(CDD_{3})X(CDD_{3})Y(CDD_{3})X(CDD_{3})-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        self.add_cdd_n(4, qubit, num_reps, d, sym)
        return


    def add_cdd5(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of CDD_5 sequence to [qubit].
        * sym=False
        Y(CDD_{4})X(CDD_{4})Y(CDD_{4})X(CDD_{4})
        * sym=True
        -Y(CDD_{4})X(CDD_{4})Y(CDD_{4})X(CDD_{4})-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        self.add_cdd_n(5, qubit, num_reps, d, sym)
        return


    #################################################
    # Begin ROBUST Sequences
    #################################################
    def add_super_hahn(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of super Hahn sequence to [qubit].
        * sym=False: X-Xb-
        * sym=True: -X=Xb-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Xb
                self.add_xb(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Xb
                self.add_xb(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    def add_super_cpmg(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of super CPMG sequence to [qubit].
        * sym=False: X-X-Xb-Xb-
        * sym=True: -X=X=Xb=Xb-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Xb
                self.add_xb(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Xb
                self.add_xb(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Xb
                self.add_xb(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Xb
                self.add_xb(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    def add_super_euler(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of super euler sequence to [qubit].
        * sym=False
        X-Y-X-Y-Y-X-Y-X-Xb-Yb-Xb-Yb-Yb-Xb-Yb-Xb-
        * sym=True
        -X=Y=X=Y=Y=X=Y=X=Xb=Yb=Xb=Yb=Yb=Xb=Yb=Xb-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            if sym is False:
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # Y
                self.add_y(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)
            else:
                # free evo (d)
                self.add_pause(qubit, d)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # Y
                self.add_y(qubit)
                # free evo (ds)
                self.add_pause(qubit, ds)
                # X
                self.add_x(qubit)
                # free evo (d)
                self.add_pause(qubit, d)

        return


    ##################################################
    # KDD Sequence
    ##################################################
    # define the KDD_\phi composite pulse
    def add_comp_kdd(self, phi, qubit, num_reps=1, d=0, sym=False):
        """
        Appends the KDD_\phi composite pulse which consists of the following:
        f_{tau/2} - (\pi)_{\pi/6 + \phi} - f_{\tau} - (\pi)_{\phi} -
        f_{\tau} - (\pi)_{\pi/2 + \phi} - f_{\tau} - (\pi)_{\phi} -
        f_{\tau} - (\pi)_{\pi/6 + \phi} - f_{\tau/2},
        where f_{\tau} is free evo for time \tau and (\pi)_{\phi} is
        a Pi rotation pulse about the \phi axis where \phi is angle between
        positive x-axis and point in x-y axis, counter-clockwise.
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        # create pulses from defaults
        if phi == 0:
            p1 = self.basis[f'X30_{qubit}']
            p2 = self.basis[f'X_{qubit}']
            p3 = self.basis[f'Y_{qubit}']
        elif np.isclose(phi, np.pi / 2):
            p1 = self.basis[f'X120_{qubit}']
            p2 = self.basis[f'Y_{qubit}']
            p3 = self.basis[f'Xb_{qubit}']
        else:
        # obtain X\pi pulse, i.e. [180]_(0) pulse from defaults lib
            # p1
            p1_ang = np.pi/6 + phi
            p1_name = f'[pi]_({p1_ang:.6f})'
            if self.basis_version == 'x_basis':
                p1 = Waveform(samples = rotate(self.basis[f'X_{qubit}'].samples,
                                                     p1_ang), name = p1_name)
            elif self.basis_version == 'g_basis':
                dur = g_basis[f'X_{qubit}'].duration
                amp = g_basis[f'X_{qubit}'].amp
                sigma = g_basis[f'X_{qubit}'].sigma
                beta = g_basis[f'X_{qubit}'].beta
                p1 = Drag(dur, rotate(amp, p1_ang), sigma, beta, name=p1_name)
            elif self.basis_version == 'c_basis':
                num_regs = self.ibmq_backend.get_number_qubit()
                circ = IBMQDdCircuit(num_regs, name=p1_name, ibmq_backend=self.ibmq_backend)
                circ.add_pi_eta(p1_ang, qubit)
                p1 = build_schedule(circ, self.ibmq_backend.backend)
            # p2
            p2_ang = phi
            p2_name = f'[pi]_({p2_ang:.6f})'
            if self.basis_version == 'x_basis':
                p2 = Waveform(samples = rotate(self.basis[f'X_{qubit}'].samples,
                                                     p2_ang), name = p2_name)
            elif self.basis_version == 'g_basis':
                dur = g_basis[f'X_{qubit}'].duration
                amp = g_basis[f'X_{qubit}'].amp
                sigma = g_basis[f'X_{qubit}'].sigma
                beta = g_basis[f'X_{qubit}'].beta
                p2 = Drag(dur, rotate(amp, p2_ang), sigma, beta, name=p2_name)
            elif self.basis_version == 'c_basis':
                num_regs = self.ibmq_backend.get_number_qubit()
                circ = IBMQDdCircuit(num_regs, name=p2_name, ibmq_backend=self.ibmq_backend)
                circ.add_pi_eta(p2_ang, qubit)
                p2 = build_schedule(circ, self.ibmq_backend.backend)
            # p3
            p3_ang = np.pi/2 + phi
            p3_name = f'[pi]_({p3_ang:.6f})'
            if self.basis_version == 'x_basis':
                p3 = Waveform(samples = rotate(self.basis[f'X_{qubit}'].samples,
                                                     p3_ang), name = p3_name)
            elif self.basis_version == 'g_basis':
                dur = g_basis[f'X_{qubit}'].duration
                amp = g_basis[f'X_{qubit}'].amp
                sigma = g_basis[f'X_{qubit}'].sigma
                beta = g_basis[f'X_{qubit}'].beta
                p3 = Drag(dur, rotate(amp, p3_ang), sigma, beta, name=p3_name)
            elif self.basis_version == 'c_basis':
                num_regs = self.ibmq_backend.get_number_qubit()
                circ = IBMQDdCircuit(num_regs, name=p3_name, ibmq_backend=self.ibmq_backend)
                circ.add_pi_eta(p3_ang, qubit)
                p3 = build_schedule(circ, self.ibmq_backend.backend)

        # add the composite pulse building block
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # p1
            if self.basis_version == 'c_basis':
                self.sched += p1
            else:
                self.sched += Play(p1, DriveChannel(qubit))
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # add p2
            if self.basis_version == 'c_basis':
                self.sched += p2
            else:
                self.sched += Play(p2, DriveChannel(qubit))
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # add p3
            if self.basis_version == 'c_basis':
                self.sched += p3
            else:
                self.sched += Play(p3, DriveChannel(qubit))
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # add p2
            if self.basis_version == 'c_basis':
                self.sched += p2
            else:
                self.sched += Play(p2, DriveChannel(qubit))
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # add p1
            if self.basis_version == 'c_basis':
                self.sched += p1
            else:
                self.sched += Play(p1, DriveChannel(qubit))
            # free evo (d)
            self.add_pause(qubit, d)

        return


    def add_kdd(self, qubit, num_reps=1, d=0, sym=False):
        """
        Appends KDD sequence to [qubit] for [num_reps] times with [tau]
        pause between each DD pulse.

        (KDD_pi/2)(KDD_0)(KDD_pi/2)(KDD_0) = "Y.X.Y.X" with composite pulses
        """
        for _ in range(num_reps):
            self.add_comp_kdd(np.pi / 2, qubit, 1, d, sym)
            self.add_comp_kdd(0, qubit, 1, d, sym)
            self.add_comp_kdd(np.pi / 2, qubit, 1, d, sym)
            self.add_comp_kdd(0, qubit, 1, d, sym)

        return


    ######################################################################
    # RGA DD Sequences
    ######################################################################
    def add_rga2x(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of rga2x sequence to [qubit].
        * sym=False
        Xb-X-
        * sym=True
        -Xb=X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Xb
            self.add_xb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d)
            self.add_pause(qubit, d)

        return


    def add_rga4(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of rga4 sequence to [qubit].
        * sym=False
        Yb-X-Yb-X-
        * sym=True
        -Yb=X=Yb=X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d)
            self.add_pause(qubit, d)

        return


    def add_rga4p(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of rga4p sequence to [qubit].
        * sym=False
        Yb-Xb-Yb-X-
        * sym=True
        -Yb=Xb=Yb=X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Xb
            self.add_xb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d)
            self.add_pause(qubit, d)

        return


    def add_rga8a(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of rga8a sequence to [qubit].
        * sym=False
        X-Yb-X-Yb-Y-Xb-Y-Xb-
        * sym=True
        -X=Yb=X=Yb=Y=Xb=Y=Xb-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Y
            self.add_y(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Xb
            self.add_xb(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Y
            self.add_y(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Xb
            self.add_xb(qubit)
            # free evo (d)
            self.add_pause(qubit, d)

        return


    def add_rga8c(self, qubit, num_reps=1, d=0, sym=False):
        """
        Adds [num_reps] of rga8c sequence to [qubit].
        * sym=False
        X-Y-X-Y-Y-X-Y-X-
        * sym=True
        -X=Y=X=Y=Y=X=Y=X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Y
            self.add_y(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Y
            self.add_y(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Y
            self.add_y(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # Y
            self.add_y(qubit)
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # free evo (d)
            self.add_pause(qubit, d)

        return


    def add_rga16a(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga16a sequence to [qubit].
        * sym=False
        Zb(RGA8a)Z(RGA8a)
        * sym=True
        Zb(RGA8a)Z(RGA8a)
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Zb
            self.add_zb(qubit)
            # RGA8a
            self.add_rga8a(qubit, 1, d, sym)
            # Z
            self.add_z(qubit)
            # RGA8a
            self.add_rga8a(qubit, 1, d, sym)
            if sym is True:
                self.add_pause(qubit, d)

        return


    def add_rga16b(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga16b'' sequence to [qubit].
        * sym=False
        RGA4'[RGA4']
        * sym=True
        RGA4'[RGA4']
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width and
        RGA4 is Yb-Xb-Yb-X-
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # RGA4p
            self.add_rga4p(qubit, 1, d, sym)
            # Xb
            self.add_xb(qubit)
            # RGA4p
            self.add_rga4p(qubit, 1, d, sym)
            # Yb
            self.add_yb(qubit)
            # RGA4p
            self.add_rga4p(qubit, 1, d, sym)
            # X
            self.add_x(qubit)
            # RGA4p
            self.add_rga4p(qubit, 1, d, sym)
            if sym is True:
                self.add_pause(qubit, d)

        return


    def add_rga32a(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga32a sequence to [qubit].
        * sym=False
        RGA4[RGA8a] for RGA4 is -Yb--X--Yb--X-
        * sym=True
        RGA4[RGA8a] for RGA4 is -Yb--X--Yb--X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Yb
            self.add_yb(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)

        return


    def add_rga32c(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga32c sequence to [qubit].
        * sym=False
        RGA8c[RGA4] for RGA8c is -X--Y--X--Y--Y--X--Y--X-
        * sym=True
        RGA8c[RGA4] for RGA8c is -X--Y--X--Y--Y--X--Y--X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA4
            self.add_rga4(qubit, num_reps, d, sym)
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)

        return


    def add_rga64a(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga64a sequence to [qubit].
        * sym=False
        RGA8a[RGA8a] for RGA8a is -X--Yb--X--Yb--Y--Xb--Y--Xb-
        * sym=True
        RGA8a[RGA8a] for RGA8a is -X--Yb--X--Yb--Y--Xb--Y--Xb-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Yb
            self.add_yb(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Yb
            self.add_yb(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Xb
            self.add_xb(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # Xb
            self.add_xb(qubit)
            # RGA8a
            self.add_rga8a(qubit, num_reps, d, sym)
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)

        return


    def add_rga64c(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga sequence to [qubit].
        * sym=False
        RGA8c[RGA8c] for RGA8c is -X--Y--X--Y--Y--X--Y--X-
        * sym=True
        RGA8c[RGA8c] for RGA8c is -X--Y--X--Y--Y--X--Y--X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # X
            self.add_x(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # Y
            self.add_y(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA8c
            self.add_rga8c(qubit, num_reps, d, sym)
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)

        return


    def add_rga256a(self, qubit, num_reps=1, d=1, sym=False):
        """
        Adds [num_reps] of rga256a sequence to [qubit].
        * sym=False
        RGA4[RGA64a] where RGA4 is -Yb--X--Yb--X-
        * sym=True
        RGA4[RGA64a] where RGA4 is -Yb--X--Yb--X-
        where - is delay of duration [d] and = is delay
        of duration [2ds + delta] for delta pulse width
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # Yb
            self.add_yb(qubit)
            # RGA64a
            self.add_rga64a(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA64a
            self.add_rga64a(qubit, num_reps, d, sym)
            # Yb
            self.add_yb(qubit)
            # RGA64a
            self.add_rga64a(qubit, num_reps, d, sym)
            # X
            self.add_x(qubit)
            # RGA64a
            self.add_rga64a(qubit, num_reps, d, sym)
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)

        return


    ##################################################
    # Universal Robust (UR) DD Sequence
    ##################################################
    def add_ur(self, n, qubit, num_reps=1, d=0, sym=False):
        """
        Appends the UR_n sequence to [qubit] for [num_reps] times with
        [tau] pause between each DD pulse.
        The sequence consists of [pi]_\phi_k rotations where phi_k is
        rotation axis (standard phi in x-y plane polar coords) in:
        "Arbitrarily Accurate Pulse Sequences for Robust" DD by
        Genov, Schraft, Vitanov, and Halfmann in PRL. The
        get_urdd_phis() function below should also make it clear.
        """
        # assumes all pulses have same delay as X which is typical
        delta = self.basis[f'X_{qubit}'].duration
        ds = 2 * d
        # get list of pulses from unique phi information
        _, unique_phi, indices = get_urdd_phis(n)
        pulse_list = []
        for phi in unique_phi:
            # check if any standard pulses
            if np.isclose(phi, 0):
                pulse_list.append(self.basis[f'X_{qubit}'])
            elif np.isclose(phi, np.pi):
                pulse_list.append(self.basis[f'Xb_{qubit}'])
            elif np.isclose(phi, np.pi / 2):
                pulse_list.append(self.basis[f'Y_{qubit}'])
            elif np.isclose(phi, ((np.pi / 2) + np.pi)):
                pulse_list.append(self.basis[f'Yb_{qubit}'])
            elif np.isclose(phi, (np.pi / 6)):
                pulse_list.append(self.basis[f'X30_{qubit}'])
            elif np.isclose(phi, ((2 * np.pi) / 3)):
                pulse_list.append(self.basis[f'X120_{qubit}'])
            else:
                # make pulse manually
                name = fr'[$\pi$]_({phi:.2f})'
                if self.basis_version == 'x_basis':
                    pulse = Waveform(samples = rotate(self.basis[f'X_{qubit}'].samples,
                                                     phi), name = name)
                elif self.basis_version == 'g_basis':
                    dur = self.basis[f'X_{qubit}'].duration
                    amp = self.basis[f'X_{qubit}'].amp
                    sigma = self.basis[f'X_{qubit}'].sigma
                    beta = self.basis[f'X_{qubit}'].beta
                    pulse = Drag(dur, rotate(amp, phi), sigma, beta, name)
                elif self.basis_version == 'c_basis':
                    num_regs = self.ibmq_backend.get_number_qubits()
                    circ = IBMQDdCircuit(num_regs, name=name, ibmq_backend=self.ibmq_backend)
                    circ.add_pi_eta(phi, qubit)
                    pulse = build_schedule(circ, self.ibmq_backend.backend)

                pulse_list.append(pulse)

        # create pulse schedule from indices and pulses
        pulse_sch = []
        for idx in indices:
            pulse_sch.append(pulse_list[idx])

        for _ in range(num_reps):
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)
            # first pulse
            if self.basis_version == 'c_basis':
                self.sched += pulse_sch[0]
            else:
                self.sched += Play(pulse_sch[0], DriveChannel(qubit))
            # add 2nd to n-1st pulse
            for pulse in pulse_sch[1:len(pulse_sch)-1]:
                # free evo (d or ds)
                if sym is True:
                    self.add_pause(qubit, ds)
                else:
                    self.add_pause(qubit, d)
                # add pulse
                if self.basis_version == 'c_basis':
                    self.sched += pulse
                else:
                    self.sched += Play(pulse, DriveChannel(qubit))
            # free evo (d or ds)
            if sym is True:
                self.add_pause(qubit, ds)
            else:
                self.add_pause(qubit, d)
            # add final pulse
            if self.basis_version == 'c_basis':
                self.sched += pulse_sch[-1]
            else:
                self.sched += Play(pulse_sch[-1], DriveChannel(qubit))
            # free evo (d)
            if sym is True:
                self.add_pause(qubit, d)

        return pulse_sch


    ##################################################
    # UDD and QDD Sequences
    ##################################################
    def add_uddx(self, n, qubit, num_reps=1, d=0, sym=False):
        """
        Appends the UDD_x sequence of order [n] to [qubits] where
        total sequence lasts time [T]. Appends sequence [num_reps] times in
        succession.
        If [T]='min', finds the minimum time that
        the UDD_x sequence of order [n] can run and uses this T.
        """
        # default to smallest pulse interval--add controlled delay later
        x_width = self.basis[f'X_{qubit}'].duration
        T = find_min_udd_time(n, x_width)
        udd_times = make_udd_sched(n, T, x_width)
        # adjust times with possible delay [d] and symmetry [sym]
        ds = 2 * d
        add_delay_T = 0
        for i in range(len(udd_times)):
            if sym is False:
                if i > 0:
                    udd_times[i] += (i * d)
                    add_delay_T += d
            else:
                if i == 0:
                    udd_times[i] += d
                    add_delay_T += d
                else:
                    udd_times[i] += (i * ds) + d
                    add_delay_T += ds

        # offset times by current running time
        for _ in range(num_reps):
            app_times = udd_times[::]
            curr_dur = self.get_duration()
            for i in range(len(app_times)):
                app_times[i] += curr_dur
            added_time = 0
            for t in app_times:
                curr_dur = self.get_duration()
                self.add_pause(qubit, t - curr_dur)
                added_time += (t - curr_dur)
                self.add_x(qubit)
                added_time += x_width
            # if even n, doesn't end on pulse, so need to add last free evo
            if n % 2 == 0:
                pause_dur = T - (added_time - add_delay_T) + d
                self.add_pause(qubit, pause_dur)
            else:
                if d > 0:
                    self.add_pause(qubit, d)

        return udd_times, T


    def add_uddy(self, n, qubit, num_reps=1, d=0, sym=False):
        """
        Appends the UDD_y sequence of order [n] to [qubits] where
        total sequence lasts time [T]. Appends sequence [num_reps] times in
        succession.
        If [T]='min', finds the minimum time that
        the UDD_y sequence of order [n] can run and uses this T.
        """
        # default to smallest pulse interval--add controlled delay later
        y_width = self.basis[f'Y_{qubit}'].duration
        T = find_min_udd_time(n, y_width)
        udd_times = make_udd_sched(n, T, y_width)
        # adjust times with possible delay [d] and symmetry [sym]
        ds = 2 * d
        add_delay_T = 0
        for i in range(len(udd_times)):
            if sym is False:
                if i > 0:
                    udd_times[i] += (i * d)
                    add_delay_T += d
            else:
                if i == 0:
                    udd_times[i] += d
                    add_delay_T += d
                else:
                    udd_times[i] += (i * ds) + d
                    add_delay_T += ds

        # offset times by current running time
        for _ in range(num_reps):
            app_times = udd_times[::]
            curr_dur = self.get_duration()
            for i in range(len(app_times)):
                app_times[i] += curr_dur
            added_time = 0
            for t in app_times:
                curr_dur = self.get_duration()
                self.add_pause(qubit, t - curr_dur)
                added_time += (t - curr_dur)
                self.add_y(qubit)
                added_time += y_width
            # if even n, doesn't end on pulse, so need to add last free evo
            if n % 2 == 0:
                pause_dur = T - (added_time - add_delay_T) + d
                self.add_pause(qubit, pause_dur)
            else:
                if d > 0:
                    self.add_pause(qubit, d)

        return udd_times, T


    def add_qdd(self, n, m, qubit, num_reps=1, d=0, sym=False):
        """
        Appends the QDD sequence of outer order [n] and inner order [m]
        to [qubits] where total sequence lasts time [T].
        Here, we choose Y in outer sequence and X in inner, i.e.
        Y X(s_(n+1)\tau) Y X(s_n\tau) ... Y X(s_2\tau) Y X(s_1\tau).
        """
        # default to smallest pulse interval--add controlled delay later
        x_width = self.basis[f'X_{qubit}'].duration
        y_width = self.basis[f'Y_{qubit}'].duration
        z_width = self.basis[f'Z_{qubit}'].duration
        T = find_min_qdd_time(n, m, x_width, y_width, z_width)
        qdd_times, pulses = make_qdd_sched(n, m, T, x_width, y_width, z_width)
        # adjust times with possible delay [d] and symmetry [sym]
        ds = 2 * d
        add_delay_T = 0
        for i in range(len(qdd_times)):
            if sym is False:
                if i > 0:
                    qdd_times[i] += (i * d)
                    add_delay_T += d
            else:
                if i == 0:
                    qdd_times[i] += d
                    add_delay_T += d
                else:
                    qdd_times[i] += (i * ds) + d
                    add_delay_T += ds

        for _ in range(num_reps):
            # offset times by current time
            app_times = qdd_times[::]
            curr_dur = self.get_duration()
            for i in range(len(qdd_times)):
                app_times[i] += curr_dur
            added_time = 0
            for idx, t in enumerate(app_times):
                curr_dur = self.get_duration()
                self.add_pause(qubit, t - curr_dur)
                added_time += (t - curr_dur)
                if pulses[idx] == f'X':
                    self.add_x(qubit)
                    added_time += x_width
                elif pulses[idx] == 'Y':
                    self.add_y(qubit)
                    added_time += y_width
                elif pulses[idx] == 'Z':
                    self.add_z(qubit)
            # if even n, doesn't end on pulse, so need to add last free evo
            if n % 2 == 0:
                pause_dur = T - (added_time - add_delay_T) + d
                self.add_pause(qubit, pause_dur)
            else:
                if d > 0:
                    self.add_pause(qubit, d)

        return qdd_times, pulses


    def add_qddzi(self, n, m, qubits, num_reps=1, d=0, sym=False):
        """
        Appends the QDD sequence of outer order [n] and inner order [m]
        to [qubits] where total sequence lasts time [T].
        Here, we choose Y in outer sequence and X in inner, i.e.
        Y X(s_(n+1)\tau) Y X(s_n\tau) ... Y X(s_2\tau) Y X(s_1\tau).
        """
        # default to smallest pulse interval--add controlled delay later
        x_width = self.basis[f'X_{qubit}'].duration
        y_width = self.basis['Y'].duration
        z_width = self.basis['Z'].duration + self.basis['I'].duration
        T = find_min_qdd_time(n, m, x_width, y_width, z_width)
        qdd_times, pulses = make_qdd_sched(n, m, T, x_width, y_width, z_width)
        # adjust times with possible delay [d] and symmetry [sym]
        ds = 2 * d
        add_delay_T = 0
        for i in range(len(qdd_times)):
            if sym is False:
                if i > 0:
                    qdd_times[i] += (i * d)
                    add_delay_T += d
            else:
                if i == 0:
                    qdd_times[i] += d
                    add_delay_T += d
                else:
                    qdd_times[i] += (i * ds) + d
                    add_delay_T += ds

        for _ in range(num_reps):
            # offset times by current time
            app_times = qdd_times[::]
            curr_dur = self.get_duration()
            for i in range(len(qdd_times)):
                app_times[i] += curr_dur
            added_time = 0
            for idx, t in enumerate(app_times):
                curr_dur = self.get_duration()
                self.add_pause(qubits, t - curr_dur)
                added_time += (t - curr_dur)
                if pulses[idx] == f'X_{qubit}':
                    self.add_x(qubits)
                    added_time += x_width
                elif pulses[idx] == 'Y':
                    self.add_y(qubits)
                    added_time += y_width
                elif pulses[idx] == 'Z':
                    self.add_z(qubits)
                    self.add_id(qubits)
            # if even n, doesn't end on pulse, so need to add last free evo
            if n % 2 == 0:
                pause_dur = T - (added_time - add_delay_T) + d
                self.add_pause(qubits, pause_dur)
            else:
                if d > 0:
                    self.add_pause(qubits, d)

        return qdd_times, pulses

######################################################################
#                          Helper Functions
######################################################################
# Rotate and create_basis obtained from Greg Quiroz and Lina Tewala
# but written by Jacob Epstein

########################################
# General Signal Utils
########################################
def rotate(complex_signal: np.ndarray, delta_phi: float) -> np.ndarray:
    '''Adds a phase to a complex signal.'''
    phi = np.angle(complex_signal) + delta_phi
    return np.abs(complex_signal) * np.exp(1j*phi)

def dt_time_to_ns(dt_time, dt):
    '''converts time in normalzied dt units to ns'''
    return (dt_time * dt * 10**9)


def ns_time_to_dt(phys_time, dt):
    '''converts time from physical (ns) to dt normalized time'''
    return (phys_time * 10**(-9)) / dt

# CDD_n helper functions
def make_cdd_n_str(n, cdd_str=""):
    """
    Creates string representation of CDD_n
    using recursion.
    """
    if n == 1:
        cdd_str += "Y-f-X-f-Y-f-X-f-"
    else:
        cdd_str += "Y"
        cdd_str = make_cdd_n_str(n - 1, cdd_str)
        cdd_str += "X"
        cdd_str = make_cdd_n_str(n - 1, cdd_str)
        cdd_str += "Y"
        cdd_str = make_cdd_n_str(n - 1, cdd_str)
        cdd_str += "X"
        cdd_str = make_cdd_n_str(n - 1, cdd_str)

    return cdd_str

def mult_paulies(p1, p2):
    """
    Multiplies two string representations
    of Pauli operators [p1] and [p2].
    """
    # handle trivial cases first
    if p1 == "I":
        return p2
    elif p2 == "I":
        return p1
    elif p1 == p2:
        return "I"
    # now handle non-trivial case
    else:
        p_list = ["X", "Y", "Z"]
        p_other = [p for p in p_list if p != "X" and p != "Y"]
        return p_other[0]

def simplify_cdd_str(cdd_str, sym=False):
    """
    Simplifies CDD_n str by using Pauli rules,
    i.e. YYY --> Y and whatnot.
    """
    simp_str = ""
    for p_combo in cdd_str.split("-f-"):
        # add simplfieid Pauli
        if len(p_combo) == 0:
            continue
        elif len(p_combo) == 1:
            simp_str += p_combo
        else:
            mult = mult_paulies(p_combo[0], p_combo[1])
            for j in range(len(p_combo) - 2):
                mult = mult_paulies(mult, p_combo[j])
            simp_str += mult
        # add delay after
        simp_str += "-f-"

    if sym is True:
        simp_str = "-f-" + simp_str + "-f-"

    return simp_str

########################################
# Init Helper Function
########################################
def create_basis(backend, qubit):
    '''given a backend, return APL style basis set of single qubit gates in the IBMQ pulse format'''
    # extract backend optimized pulses (no virtual Z gates)
    defaults = backend.backend.defaults(refresh=True)
    x90_index = [pulse.name for pulse in defaults.pulse_library].index(f'X90p_d{qubit}')
    x180_index = [pulse.name for pulse in defaults.pulse_library].index(f'Xp_d{qubit}')
    x90_samples = defaults.pulse_library[x90_index].samples
    x180_samples = defaults.pulse_library[x180_index].samples
    id_samples = np.zeros(len(x180_samples))

    # construct a 'basis' of pulses
    basis = {}
    # construct the standard X, Y pulses with 'half rotations'
    basis['I'] = SamplePulse(samples = id_samples, name = 'I')
    basis['X'] = SamplePulse(samples = x180_samples, name = 'X')
    basis['X90'] = SamplePulse(samples = x90_samples, name = 'X90')
    basis['Y'] = SamplePulse(samples = rotate(x180_samples, np.pi/2), name = 'Y')
    basis['Y90'] = SamplePulse(samples = rotate(x90_samples, np.pi/2), name = 'Y90')
    # constuct the robust pulses, i.e. X-bar, Y-bar, etc...
    basis['Xb'] = SamplePulse(samples = rotate(x180_samples, np.pi), name = 'Xb')
    basis['X90b'] = SamplePulse(samples = rotate(x90_samples, np.pi), name = 'X90b')
    basis['Yb'] = SamplePulse(samples = rotate(x180_samples, -np.pi/2), name = 'Yb')
    basis['Y90b'] = SamplePulse(samples = rotate(x90_samples, -np.pi/2), name = 'Y90b')
    # construct additional pulses needed for KDD
    basis['X30'] = SamplePulse(samples = rotate(x180_samples, np.pi/6), name = 'X30')
    basis['X120'] = SamplePulse(samples = rotate(x180_samples, (2*np.pi)/3), name = 'X120')
    return basis

def create_from_x_basis(backend, qubit):
    '''given a backend, return APL style basis set of single qubit gates in the IBMQ pulse format'''
    pulse_basis = {}
    defaults = backend.backend.defaults(refresh=True).instruction_schedule_map
    x180 = defaults.get('x',qubit).instructions[0][1].pulse
    x180_samples = x180.get_waveform().samples
    id_samples = np.zeros(len(x180_samples))

    # construct the standard X, Y pulses with 'half rotations'
    pulse_basis[f'I_{qubit}'] = Waveform(samples = id_samples, name = 'I')
    pulse_basis[f'X_{qubit}'] = Waveform(samples = x180_samples, name = 'X')
    pulse_basis[f'Y_{qubit}'] = Waveform(samples = rotate(x180_samples, np.pi/2), name = 'Y')
    # constuct the robust pulses, i.e. X-bar, Y-bar, etc...
    pulse_basis[f'Xb_{qubit}'] = Waveform(samples = rotate(x180_samples, np.pi), name = 'Xb')
    pulse_basis[f'Yb_{qubit}'] = Waveform(samples = rotate(x180_samples, -np.pi/2), name = 'Yb')
    # construct additional pulses needed for KDD
    pulse_basis[f'X30_{qubit}'] = Waveform(samples = rotate(x180_samples, np.pi/6), name = 'X30')
    pulse_basis[f'X120_{qubit}'] = Waveform(samples = rotate(x180_samples, (2*np.pi)/3), name = 'X120')
    return pulse_basis

def create_from_greg_basis(backend, qubit):
    '''given a backend, return APL style basis set of single qubit gates in the IBMQ pulse format'''
    defaults = backend.backend.defaults(refresh=True).instruction_schedule_map
    greg_basis = {}
    x180 = defaults.get('x', qubit).instructions[0][1].pulse
    x180_samples = x180.get_waveform().samples
    dur = x180.parameters['duration']
    amp = x180.parameters['amp']
    sigma = x180.parameters['sigma']
    beta = x180.parameters['beta']
    x180 = Drag(dur, amp, sigma, beta, name="X")
    x90 = Drag(dur, amp/2, sigma, beta, name='X2')
    x90_samples = x90.get_waveform().samples
    x90m = Drag(dur, -amp/2, sigma, beta, name='X2m')
    x180m = Drag(dur, -amp, sigma, beta, name='Xb')

    # get the X30 and X120 pulses for KDD
    x30 = Drag(dur, rotate(amp, np.pi/6), sigma, beta, name='X30')
    x120 = Drag(dur, rotate(amp, (2*np.pi)/3), sigma, beta, name='X120')
    success = False
    index = 1
    while not success:
        try:
            y90 = defaults.get('u2', qubit, P0=0, P1=0).instructions[index][1].pulse
            success = True
        except:
            index += 1

    dur = y90.parameters['duration']
    amp = y90.parameters['amp']
    sigma = y90.parameters['sigma']
    beta = y90.parameters['beta']
    y90m = Drag(dur, -amp, sigma, beta, name='Y2m')
    y180 = Drag(dur, 2*amp, sigma, beta, name='Y')
    y180m = Drag(dur, -2*amp, sigma, beta, name='Yb')

    empty_samples = np.zeros(len(x180_samples))

    greg_basis[f'X_{qubit}'] = x180 #Waveform(samples = x180_samples, name = 'X')
    greg_basis[f'X2_{qubit}'] = x90 #Waveform(samples = x90_samples, name = 'X2')
    greg_basis[f'Y_{qubit}'] = y180 #Waveform(samples = rotate(x180_samples,np.pi/2), name = 'Y')
    greg_basis[f'Y2_{qubit}'] = y90 #Waveform(samples = rotate(x90_samples,np.pi/2), name = 'Y2')
    greg_basis[f'Xb_{qubit}'] = x180m #Waveform(samples = rotate(x180_samples,np.pi), name = 'Xm')
    greg_basis[f'X2m_{qubit}'] = x90m #Waveform(samples = rotate(x90_samples,np.pi), name = 'X2m')
    greg_basis[f'Yb_{qubit}'] = y180m #Waveform(samples = rotate(x180_samples,-np.pi/2), name = 'Ym')
    greg_basis[f'Y2m_{qubit}'] = y90m #Waveform(samples = rotate(x90_samples,-np.pi/2), name = 'Y2m')
    greg_basis[f'X30_{qubit}'] = x30
    greg_basis[f'X120_{qubit}'] = x120
    greg_basis[f"I_{qubit}"] = defaults.get('id', qubit).instructions[0][1].pulse
    #greg_basis[f'I_{qubit}'] = Waveform(samples = empty_samples, name = 'I')
    # create Z pulse
    num_regs = backend.get_number_qubits()
    circ = IBMQDdCircuit(num_regs, name='Z', ibmq_backend=backend)
    circ.add_z(qubit)
    greg_basis[f'Z_{qubit}'] = build_schedule(circ, backend.backend)
    circ = IBMQDdCircuit(num_regs, name='Zb', ibmq_backend=backend)
    circ.add_zb(qubit)
    greg_basis[f'Zb_{qubit}'] = build_schedule(circ, backend.backend)
    return greg_basis

def create_from_circ_basis(backend, num_regs, qubit):
    '''given a backend, return APL style basis set of single qubit gates in the IBMQ pulse format derived from Circuit API choices'''
    circ_basis = {}
    # create X pulse
    circ = IBMQDdCircuit(num_regs, name='X', ibmq_backend=backend)
    circ.add_x(qubit)
    circ_basis[f'X_{qubit}'] = build_schedule(circ, backend.backend)
    # create Xb pulse
    circ = IBMQDdCircuit(num_regs, name='Xb', ibmq_backend=backend)
    circ.add_xb(qubit)
    circ_basis[f'Xb_{qubit}'] = build_schedule(circ, backend.backend)
    # create Y pulse
    circ = IBMQDdCircuit(num_regs, name='Y', ibmq_backend=backend)
    circ.add_y(qubit)
    circ_basis[f'Y_{qubit}'] = build_schedule(circ, backend.backend)
    # add symmetric Y
    circ = IBMQDdCircuit(num_regs, name="sym-Y", ibmq_backend=backend)
    circ.rz(np.pi/2, qubit)
    circ.barrier(qubit)
    circ.x(qubit)
    circ.barrier(qubit)
    circ.rz(-np.pi/2, qubit)
    circ.barrier(qubit)
    circ_basis[f'symY_{qubit}'] = build_schedule(circ, backend.backend)
    # create Yb pulse
    circ = IBMQDdCircuit(num_regs, name='Yb', ibmq_backend=backend)
    circ.add_yb(qubit)
    circ_basis[f'Yb_{qubit}'] = build_schedule(circ, backend.backend)
    # create X30 pulse
    circ = IBMQDdCircuit(num_regs, name='X30', ibmq_backend=backend)
    circ.add_pi_eta(30, qubit)
    circ_basis[f'X30_{qubit}'] = build_schedule(circ, backend.backend)
    # create X120 pulse
    circ = IBMQDdCircuit(num_regs, name='X120', ibmq_backend=backend)
    circ.add_pi_eta(120, qubit)
    circ_basis[f'X120_{qubit}'] = build_schedule(circ, backend.backend)
    # create I pulse
    circ = IBMQDdCircuit(num_regs, name='I', ibmq_backend=backend)
    circ.id(qubit)
    circ.barrier(qubit)
    circ.id(qubit)
    circ_basis['I'] = build_schedule(circ, backend.backend)
    # create Z pulse
    circ = IBMQDdCircuit(num_regs, name='Z', ibmq_backend=backend)
    circ.add_z(qubit)
    circ_basis[f'Z_{qubit}'] = build_schedule(circ, backend.backend)

    return circ_basis


########################################
# DD Sequence Helper Functions
########################################

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

#########################
# UDD/QDD Functions
#########################
def tj(j, n, T):
    """ Returns [j]th UDD_[n] time for total DD time [T]."""
    frac = (j * np.pi) / (2 * n + 2)
    return T * (np.sin(frac))**2

def make_tj_list(n, T):
    """ Returns list of tj times for UDD_[n] for DD time [T]."""
    if n % 2 == 0:
        j_list = [j for j in range(1, n+1)]
    else:
        j_list = [j for j in range(1, n+2)]
    return np.array([tj(j, n, T) for j in j_list])

def tj_k(k, m, tau_j, tj_m_1):
    """
    Returns tj_[k] for QDD_n_[m] where inner pulses occur over
    time [tau_j] and last outer pulse is at [tj_m_1].
    """
    frac = (k * np.pi) / (2 * m + 2)
    return tau_j * (np.sin(frac))**2 + tj_m_1

def make_tj_k_list(m, tau_j, tj_m_1):
    """
    Returns list of tj_k times for QDD_n_[m] inner pulses
    over time [tau_j] where last outer pulse is at [tj_m_1].
    """
    if m % 2 == 0:
        k_list = [k for k in range(1, m+1)]
    else:
        k_list = [k for k in range(1, m+2)]
    return np.array([tj_k(k, m, tau_j, tj_m_1) for k in k_list])

def make_udd_sched(n, T, pulse_width=0):
    """
    Make a UDD_[n] time schedule over total DD time [T] where non-ideal
    pulses have finite [pulse_width]. Makes idealized tj times and then
    substracts off [pulse_width] then checks no pulse overlaps.

    Returns physical_times when pulses should START.
    """
    # get idealized tj times
    tj_list = make_tj_list(n, T)
    # turn physical times into integers
    tj_list = np.array(list(map(int, np.floor(tj_list))))
    # subtract off pulse_width to get correct time to begin gates
    phys_tj_list = list(map(int, tj_list - pulse_width))

    # ensure that no gates must be applied at "negative" times
    if phys_tj_list[0] < 0:
        e = (f"Either n too large or T too small to accomodate pulses with\
        width {pulse_width} since first gate must be applied at t1\
        = {phys_tj_list[0]}.\n")
        raise ValueError(e)
    # ensure no diff in t between gates is smaller than pulse_width
    diffs = []
    for idx in range(1, len(phys_tj_list)):
        diffs.append(phys_tj_list[idx] - phys_tj_list[idx - 1])
    min_diff = min(diffs)
    if min_diff < pulse_width:
        e = (f"Minimum pulse spacing required for n={n} and T={T} is\
            {min_diff}, but pulse_width is {pulse_width}.")

    return phys_tj_list

def find_min_udd_time(n, x_width):
    """
    Finds smallest acceptable UDD time which
    corresponds to smallest pulse delay.
    """
    # first, get order of magnitude guess for T
    T = x_width
    success = False
    while success is False:
        try:
            make_udd_sched(n, T, x_width)
            success = True
        except:
            prev_T = T
            T *= 10

    # now try guesses incrementally until minimum T found
    for Tg in range(prev_T, T):
        try:
            make_udd_sched(n, Tg, x_width)
            break
        except:
            continue

    return Tg

def make_mid_udd_sched(n, T, pulse_width=0):
    """
    Make a UDD_[n] time schedule over total DD time [T] where non-ideal
    pulses have finite [pulse_width]. Makes idealized tj times and then
    substracts off [pulse_width]/2 then checks no pulse overlaps.

    Returns physical_times when pulses should START.
    """
    # get idealized tj times
    tj_list = make_tj_list(n, T)
    # subtract off pulse_width to get correct time to begin gates
    phys_tj_list = tj_list - (pulse_width / 2)
    # schedule can only be specified to nearest integer
    phys_tj_list = list(map(int, np.ceil(phys_tj_list)))

    # ensure that no gates must be applied at "negative" times
    if phys_tj_list[0] < 0:
        e = (f"Either n too large or T too small to accomodate pulses with\
        width {pulse_width} since first gate must be applied at t1\
        = {phys_tj_list[0]}.\n")
        raise ValueError(e)
    # ensure no diff in t between gates is smaller than pulse_width
    diffs = []
    for idx in range(1, len(phys_tj_list)):
        diffs.append(phys_tj_list[idx] - phys_tj_list[idx - 1])
    min_diff = min(diffs)
    if min_diff < pulse_width:
        e = (f"Minimum pulse spacing required for n={n} and T={T} is\
            {min_diff}, but pulse_width is {pulse_width}.")

    return phys_tj_list

def make_end_udd_sched(n, T, pulse_width=0):
    """
    Make a UDD_[n] time schedule over total DD time [T] where non-ideal
    pulses have finite [pulse_width]. Makes idealized tj times and uses
    these as start times of pulses.

    Returns physical_times when pulses should START.
    """
    # get idealized tj times
    tj_list = make_tj_list(n, T)
    # subtract off pulse_width to get correct time to begin gates
    phys_tj_list = tj_list - 0
    # schedule can only be specified to nearest integer
    phys_tj_list = list(map(int, np.ceil(phys_tj_list)))

    # ensure that no gates must be applied at "negative" times
    if phys_tj_list[0] < 0:
        e = (f"Either n too large or T too small to accomodate pulses with\
        width {pulse_width} since first gate must be applied at t1\
        = {phys_tj_list[0]}.\n")
        raise ValueError(e)
    # ensure no diff in t between gates is smaller than pulse_width
    diffs = []
    for idx in range(1, len(phys_tj_list)):
        diffs.append(phys_tj_list[idx] - phys_tj_list[idx - 1])
    min_diff = min(diffs)
    if min_diff < pulse_width:
        e = (f"Minimum pulse spacing required for n={n} and T={T} is\
            {min_diff}, but pulse_width is {pulse_width}.")

    return phys_tj_list

def make_qdd_sched(n, m, T, xpw=0, ypw=0, zpw=0):
    """
    Make a QDD_[n]_[m] schedule over total DD time [T] where non-ideal
    X pulse has width [xpw] and non-ideal Y pulse has width [Y]. Makes
    idealized times and then substracts of [xpw] or [ypw] as appropriate.
    Then checks no pulses overlap.

    Returns physical_times when pulses should START as well as the pulse
    order, i.e. [1, 5, 7] and ['X', 'Y', 'X'].
    """
    # first, we will make an idealized qdd schedule
    # first, get ideal outer pulse times
    outer_times = make_tj_list(n, T)
    # cast as integers
    outer_times = list(map(int, np.floor(outer_times)))
    # keep track of all ideal times (inner and outer) along
    # with what pulse is applied (order associated to ideal_times)
    ideal_times = []
    pulses = []
    # add inner pulses to ideal_times/ pulses
    for j in range(len(outer_times)):
        tj = outer_times[j]
        # if first element, then previous time is implicitly 0
        if j == 0:
            tj_m_1 = 0
        else:
            tj_m_1 = outer_times[j-1]
        # tau_j is the pulse interval (or diff, t_j - t_(j-1))
        tau_j = tj - tj_m_1
        inner_times = make_tj_k_list(m, tau_j, tj_m_1)
        inner_times = list(map(int, np.floor(inner_times)))

        # add inner times/ X pulses to combined lists
        ideal_times.extend(inner_times)
        pulses.extend(['X' for _ in range(len(inner_times))])
        # add outer times/ Y pulses to combined lists
        ideal_times.append(outer_times[j])
        pulses.append('Y')

        # when m is odd, last inner pulse is simultaneous to outer
        if m % 2 == 1:
            pulses = pulses[:-2 or None]
            pulses.append('Z')
            ideal_times = ideal_times[:-1 or None]

    # now, make these ideal times "physical" by subtracting pulse widths
    # off of times when ideal gate should be applied (so that pulse
    # end when it should ideally be applied)
    ideal_times = list(map(int, np.ceil(ideal_times)))
    physical_times = []
    for (idx, p) in enumerate(pulses):
        if p == 'X':
            tj = ideal_times[idx] - xpw
            physical_times.append(tj)

            # make finite pulse width checks
            if idx == 0:
                tj_m_1 = 0
            else:
                tj_m_1 = physical_times[idx - 1]
            tau_j = tj - tj_m_1
            if xpw > 0 and tau_j < xpw:
                e = (f"Either n, m too large or T too small to accomodate\
                     X pulse of width {xpw} since two pulses must be applied\
                     with tau_{idx} = {tau_j} < {xpw}.")
                raise ValueError(e)
        elif p == 'Y':
            tj = ideal_times[idx] - ypw
            physical_times.append(tj)

            # make finite pulse width checks
            if idx == 0:
                tj_m_1 = 0
            else:
                tj_m_1 = physical_times[idx - 1]
            tau_j = tj - tj_m_1
            if ypw > 0 and tau_j < ypw:
                e = (f"Either n, m too large or T too small to accomodate\
                     Y pulse of width {ypw} since two pulses must be applied\
                     with tau_{idx} = {tau_j} < {ypw}.")
                raise ValueError(e)
        elif p == 'Z':
            tj = ideal_times[idx] - zpw
            physical_times.append(tj)

            # make finite pulse width checks
            if idx == 0:
                tj_m_1 = 0
            else:
                tj_m_1 = physical_times[idx - 1]
            tau_j = tj - tj_m_1
            if zpw > 0 and tau_j < zpw:
                e = (f"Either n, m too large or T too small to accomodate\
                     Z pulse of width {zpw} since two pulses must be applied\
                     with tau_{idx} = {tau_j} < {zpw}.")
                raise ValueError(e)

    return physical_times, pulses

def find_min_qdd_time(n, m, x_width, y_width, z_width):
    """
    Finds smallest acceptable UDD time which
    corresponds to smallest pulse delay.
    """
    # first, get order of magnitude guess for T
    T = max([x_width, y_width, z_width])
    success = False
    while success is False:
        try:
            #print(f"trying T: {T}")
            make_qdd_sched(n, m, T, x_width, y_width)
            success = True
        except:
            #print(f"failue with T: {T}")
            prev_T = T
            T *= 10

    # now try guesses incrementally until minimum T found
    for Tg in range(prev_T, T):
        try:
            #print(f"trying T: {T} in second loop")
            make_qdd_sched(n, m, Tg, x_width, y_width)
            break
        except:
            #print(f"failue with T: {T}")
            continue

    return Tg
