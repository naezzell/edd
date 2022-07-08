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

import datetime
import random
import numpy as np
from qiskit import IBMQ, execute, Aer, assemble
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.aer.noise import NoiseModel

# type check from IBMQDdSchedule
from edd.pulse import IBMQDdSchedule

class IBMQBackend():
    """
    This is backend.
    """
    def __init__(self, strname, hub, group, project, token):

        """
        Loads backend with [strname].
        """
        IBMQ.save_account(hub=hub, group=group, project=project, token=token, overwrite=True)
        self.provider = IBMQ.load_account()
        self.backend = self.provider.get_backend(strname)
        now = datetime.datetime.now()
        print("IBM Backend {} loaded at date/time: {}".format(strname, now.strftime("%Y-%m-%d %H:%M:%S")))
        # now get properties of device and save as member data
        self.config = self.backend.configuration().to_dict()
        if strname != 'ibmq_qasm_simulator':
            self.props = self.backend.properties().to_dict()
            self.gate_props = self.backend.properties()._gates
        else:
            self.props = "N/A"
            self.gate_props = "N/A"

        return

    def change_backend(self, strname):
        """
        Loads [strname] backend. Replaces current backend in doing so.
        """
        self.provider = IBMQ.load_account()
        oldname = self.config['backend_name']
        self.backend = self.provider.get_backend(strname)
        now = datetime.datetime.now()
        print("Old backend {} switched to {} at date/time {}".format(
            oldname, strname, now.strftime("%Y-%m-%d %H:%M:%S")))

        # now get properties of device and save as member data
        self.config = self.backend.configuration().to_dict()
        self.props = self.backend.properties().to_dict()
        self.gate_props = self.backend.properties()._gates

        return

    def get_remaining_jobs_count(self):
        """
        Get number of remaining jobs to run.
        """
        return self.backend.remaining_jobs_count()

    def get_backend_config_dict(self):
        return self.config

    def get_backend_props_dict(self):
        return self.props

    def get_dt(self, unit):
        """
        Get inverse sampling rate of backend in [unit].
        """
        dt = self.backend.configuration().dt

        if unit == 's':
            dt = dt
        elif unit == 'ns':
            dt = dt * 1e9
        else:
            e = f"Unit {unit} not supported."
            raise ValueError(e)

        return dt

    def get_acquire_alignment(self):
        """
        Instructions must be a multiple of acquire_alignment
        for a measurement to make sense.
        """
        return self.config['timing_constraints']['acquire_alignment']

    def get_readable_props_str(self):
        p = self.get_backend_props_dict()
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        """Returns backend props info in human-readable string."""
        prop_str = "Backend Properties\n"
        prop_str += "---\n"
        prop_str += f"Experiment Date: {now}\n"
        prop_str += f"Backend Name: {p['backend_name']}\n"
        prop_str += f"Version: {p['backend_version']}\n"
        prop_str += f"Last Update Date: {p['last_update_date']}\n"

        prop_str += "\nGate Info\n"
        prop_str += "---\n"
        # add gate header information
        gate_header = "name"
        for gate_params in p['gates'][0]['parameters']:
            gate_header += f", {gate_params['name']}({gate_params['unit']})"
        gate_header += "\n"
        prop_str += gate_header
        # add info corresponding to header info for each gate
        for gate in p['gates']:
            gate_info = f"{gate['gate']}"
            for param in gate['parameters']:
                gate_info += ", "
                gate_info += f"{param['value']}"
            prop_str += (gate_info + "\n")

        prop_str += "\nQubit Info\n"
        prop_str += "---\n"
        # add qubit header information
        qubit_header = "qubit"
        for qubit_params in p['qubits'][0]:
            qubit_header += f", {qubit_params['name']}({qubit_params['unit']})"
        qubit_header += "\n"
        prop_str += qubit_header
        # add info corresponding to header info for each gate
        for qubit, qubit_params in enumerate(p['qubits']):
            qubit_info = str(qubit)
            for param in qubit_params:
                qubit_info += ", "
                qubit_info += f"{param['value']}"
            prop_str += (qubit_info + "\n")

        return prop_str

    def save_backend_props(self, fname):
        prop_str = self.get_readable_props_str()
        with open(fname, 'w+') as f:
            f.write(prop_str)
        return

    def get_number_qubits(self):
        """Outputs the number of qubits on this backend."""
        n_qubits = self.config['n_qubits']
        return n_qubits

    def get_native_gates(self):
        """Outputs gates native to this backend."""
        nat_gates = self.config['basis_gates']
        return nat_gates

    def get_gate_times(self, avg=True):
        """ Acquires gate application times in ns. If avg is True, returns
        the average time it takes gate to apply across all qubits. If False,
        returns times of gate for each qubit. """

        # first get the native gate set
        nat_gates = self.config['basis_gates']
        gate_to_time = {}
        for gate in nat_gates:
            gate_info = self.gate_props[gate]
            # iterate over qubits the gate is defined over and check if
            # gate times are all the same
            times = []
            for qubit in gate_info:
                times.append(gate_info[qubit]['gate_length'][0] * 1e9)
            if avg is True:
                gate_to_time[gate] = np.mean(times)
            else:
                gate_to_time[gate] = times

        return gate_to_time


    def get_max_runs(self):
        """
        Given the backend, returns the max # experiments and max # shots
        that can be queued in a single job.
        """

        max_experiments = self.config['max_experiments']
        max_shots = self.config['max_shots']
        max_runs = {'max_experiments': max_experiments, 'max_shots': max_shots}
        return max_runs


    def submit_job(self, experiments, qobj_id, num_shots='max',
                   shuffle=False):
        """
        Submit [experiments] to [backend] and run [num_shots] for each of
        the [experiments]. Results get a qobj_id tag labelled with [qobj_id].
        If [shuffle] is True, randomly shuffles input data before sending to
        prevent biasing data with time.
        """
        # set runs to max runs allowed by hardware if set to do so
        max_runs = self.get_max_runs()
        if str(num_shots).lower() == 'max':
            num_shots = max_runs['max_shots']

        if not isinstance(experiments, list):
            experiments = [experiments]

        # parse experiments a bit before passing to assemble
        parsed_experiments = []
        for exp in experiments:
            if isinstance(exp, IBMQDdSchedule):
                parsed_experiments.append(exp.sched)
            else:
                parsed_experiments.append(exp)

        # shuffle data if desired
        if shuffle is True:
            random.shuffle(parsed_experiments)

        # submit the job in the background and output status information
        program = assemble(parsed_experiments, backend=self.backend,
                           shots=num_shots, qobj_id=qobj_id)
        job = self.backend.run(program)
        return job

    def get_noisemodel(self):
        """
        Given a backend, loads in the necessary information to run a noisy
        simulation emulating noise on actual device.
        """
        # set-up noise model for simulator
        noise_model = NoiseModel.from_backend(self.backend)
        # Get coupling map from backend
        coupling_map = self.config['coupling_map']
        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        noise_info = {'noise_model': noise_model, 'coupling_map': coupling_map, 'basis_gates': basis_gates}

        return noise_info

    def submit_test(self, experiments, qobj_id, num_shots = 1000):
        """
        Submits [experiments] to simulator with noisemodel of [backend] and
        runs [num_shots] for each
        """
        # set up noise model of backend
        noise_info = self.get_noisemodel()
        coupling_map = noise_info['coupling_map']
        basis_gates = noise_info['basis_gates']
        noise_model = noise_info['noise_model']

        job = execute(experiments, Aer.get_backend('qasm_simulator'),
                      coupling_map=coupling_map,
                      basis_gates=basis_gates,
                      noise_model=noise_model,
                      shots=num_shots, qobj_id=qobj_id)

        return job.result()
