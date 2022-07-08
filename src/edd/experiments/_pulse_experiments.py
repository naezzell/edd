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
#   limitations under the Licens

import datetime
import numpy as np
import os
from edd.pulse import IBMQDdSchedule

######################################################################
# Hosts functions for experiments using the OpenPulse API
######################################################################

# Load in relevant state input data
import pathlib
path = pathlib.Path(__file__).parent.resolve()
u3_params = np.load(os.path.join(path, "../states/u3_list.npy"))

######################################################################
# THETA SWEEP FUNCTIONS
# Description: Generically, these experiemnts test the efficacy of
# different DD sequences over a range of easy to prepare superposition
# states of the form cos(t/2)|0> + sin(t/2)|1>. This prevents undue
# biasing due to robustness of the |0> state.
######################################################################
def theta_sweep_free(num_ids, backend, basis, encoding_qubits='all',
                     dd_qubits='all', theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- free -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [num_ids] identity gates is applied on [dd_qubits].
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_free(0, num_ids)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"theta_sweep_free_encodeqs_{encoding_qubits}_ddqs_{dd_qubits}_"
    job_tag += f"ids_{num_ids}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a schedule with correct number of qubits
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t))
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        sched.encode_theta_state(encoding_qubits, t)
        # apply identity gatesb
        sched.add_free(dd_qubits, num_ids)
        # decode superposition state back into |0> state
        sched.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        #TODO: add support for choosing classical register
        for idx, q in enumerate(encoding_qubits):
                sched.add_measurement(q)
        experiments.append(sched)

    return (job_tag, experiments)

def theta_sweep_dd(seq_name, num_reps, tau, backend, basis,
                   encoding_qubits='all', dd_qubits='all',
                   theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- DD seq -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] pause between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # prepend the 'add' to seq name string
    seq_method = 'add_' + seq_name
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this 'DD' sequence
    time_sched = IBMQDdSchedule(backend, basis)
    getattr(time_sched, seq_method)(0, num_reps, tau)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    phys_tau = ns_time_to_dt(tau, time_sched.dt)
    job_tag = f"pulse_{basis}_"
    job_tag += f"theta_sweep_{seq_name}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_tau_{phys_tau}ns_T_{T}ns"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a schedule with correct number of qubits
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t))
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        sched.encode_theta_state(encoding_qubits, t)
        # apply DD sequence with correct parameters
        getattr(sched, seq_method)(0, num_reps, tau)
        # decode superposition state back into |0> state
        sched.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        #TODO: add support for choosing classical regsiter
        for idx, q in enumerate(encoding_qubits):
                sched.add_measurement(q)
        experiments.append(sched)

    return (job_tag, experiments)

def theta_sweep_uddx(n, time, backend, basis, encoding_qubits='all',
                     dd_qubits='all', theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- udd_x -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    UDDx_n is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] pause between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_uddx(0, n, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"theta_sweep_uddx_{n}_T_{time}dt_{T}ns_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a schedule with correct number of qubits
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t))
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        sched.encode_theta_state(encoding_qubits, t)
        # apply uddx_[n] sequence
        sched.add_uddx(dd_qubits, n, time)
        # decode superposition state back into |0> state
        sched.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        #TODO: add support for choosing classical regsiter
        for idx, q in enumerate(encoding_qubits):
                sched.add_measurement(q)
        experiments.append(sched)

    return (job_tag, experiments)

def theta_sweep_uddy(n, time, backend, basis, encoding_qubits='all',
                     dd_qubits='all', theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- udd_y -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    UDDx_n is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] pause between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_uddy(0, n, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"theta_sweep_uddy_{n}_T_{time}dt_{T}ns_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a schedule with correct number of qubits
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t))
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        sched.encode_theta_state(encoding_qubits, t)
        # apply uddx_[n] sequence
        sched.add_uddy(dd_qubits, n, time)
        # decode superposition state back into |0> state
        sched.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        #TODO: add support for choosing classical regsiter
        for idx, q in enumerate(encoding_qubits):
                sched.add_measurement(q)
        experiments.append(sched)

    return (job_tag, experiments)

def theta_sweep_qdd(n, m, time, backend, basis, encoding_qubits='all',
                    dd_qubits='all', theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- qdd -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    QDD_{n}_{m} is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] pause between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_qdd(0, n, m, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"theta_sweep_qdd_{n}_{m}_T_{time}dt_{T}ns_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a schedule with correct number of qubits
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t))
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        sched.encode_theta_state(encoding_qubits, t)
        # apply uddx_[n] sequence
        sched.add_qdd(dd_qubits, n, m, time)
        # decode superposition state back into |0> state
        sched.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        #TODO: add support for choosing classical regsiter
        for idx, q in enumerate(encoding_qubits):
                sched.add_measurement(q)
        experiments.append(sched)

    return (job_tag, experiments)

######################################################################
# Fideltiy Decay Functions
# Description: Experimentally measures the fidelity of different DD
# sequenences as a function of total sequence time. To be specific,
# suppose XY4 sequence takes 100ns to run. Then We'd measure
# the fidelity for XY4 1 rep (100ns), 2 reps (200ns), etc...
# Ideally, we'd like to know the fidelity "on average" for a "typical
# state", so perform the DD sequence on a list of haar random states.
######################################################################

############################################################
# Type 1 Fidelity Decay
# Here, we allow the function to take as input which haar
# random states to try and protect for all times tried.
# That is, we try same inputted haar random states for 1 rep
# of XY4 as we do for 2 reps of XY4.
############################################################
def static_haar_fid_decay_free(haar_params_list, time, backend, basis,
                               encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- free -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    idenity (i.e. pause) is applied for [time]dt units of time.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_pause(0, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_free_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        sched.add_pause(dd_qubits, time)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def static_haar_fid_decay_dd(haar_params_list, seq_name, num_reps, tau, backend,
                             basis, encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector) and n \in num_id_list of the form:
    |0> -- U3(t,0,0) -- DD seq -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] identities between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # prepend the 'add' to seq name string
    seq_method = 'add_' + seq_name
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many reps of dd seq
    time_sched = IBMQDdSchedule(backend, basis)
    getattr(time_sched, seq_method)(0, num_reps, tau)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_{seq_name}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_tau_{tau}dt_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        getattr(sched, seq_method)(dd_qubits, num_reps, tau)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def pauli_pode_fid_decay_free(offset, time, backend, basis,
                               encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments which probe fidelity of Pauli podes.
    |0> -- encode_pode_state(offset) -- free -- decode_pode_state(offset),
    where encoding is prepared on [encoding_qubits] and
    idenity (i.e. pause) is applied for [time]dt units of time.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_pause(0, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_pauli_pode_offset_{offset}_"
    job_tag += f"fid_decay_free_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_T_{T}ns"
    exp_tag = job_tag + "_pode_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for pode in range(6):
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(pode))
        sched.encode_podal_state(encoding_qubits, pode, offset)
        sched.add_pause(dd_qubits, time)
        sched.decode_podal_state(encoding_qubits, pode, offset)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def pauli_pode_fid_decay_dd(offset, seq_name, sym, num_reps, d,
                            d_label, backend, basis,
                            encoding_qubit, dd_qubit):
    """
    -->Generates list of experiments which probe fidelity of Pauli podes.
    |0> -- encode_pode_state(offset) -- DD -- decode_pode_state(offset),
    where encoding done on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] identities between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    if "uddx" in seq_name:
        seq_method = "add_uddx"
        ord_n = int(seq_name[4:])
    elif "uddy" in seq_name:
        seq_method = "add_uddy"
        ord_n = int(seq_name[4:])
    elif "qdd" in seq_name:
        seq_method = "add_qdd"
        _, ord_n, ord_m = seq_name.split("_")
        ord_n = int(ord_n)
        ord_m = int(ord_m)
    elif "ur" in seq_name and "x" not in seq_name and "y" not in seq_name:
        seq_method = "add_ur"
        ord_n = int(seq_name[2:])
    else:
        seq_method = "add_" + seq_name
    # set up which qubits to run DD sequence on (not same as those
    # get amount of time it takes to run this many reps of dd seq
    qubit = dd_qubit
    time_sched = IBMQDdSchedule(backend, basis)
    if "uddx" in seq_name or "uddy" in seq_name or ("ur" in seq_name and "x" not in seq_name and "y" not in seq_name):
        getattr(time_sched, seq_method)(ord_n, qubit, 1, d, sym)
    elif "qdd" in seq_name:
        getattr(time_sched, seq_method)(ord_n, ord_m, qubit, 1, d, sym)
    else:
        getattr(time_sched, seq_method)(qubit, 1, d, sym)
    T = (time_sched.get_phys_time() * num_reps)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_pauli_pode_offset_{offset}_dtype_{d_label}_"
    job_tag += f"fid_decay_{seq_name}_sym_{sym}_encodeq_{encoding_qubit}_"
    job_tag += f"ddq_{dd_qubit}_reps_{num_reps}_delay_{d}dt_T_{T}ns"
    exp_tag = job_tag + "_pode_{}"

    # create single rep of DD seq
    sing_dd_sched = IBMQDdSchedule(backend, basis)
    if "uddx" in seq_name or "uddy" in seq_name or ("ur" in seq_name and "x" not in seq_name and "y" not in seq_name):
        getattr(sing_dd_sched, seq_method)(ord_n, dd_qubit, 1, d, sym)
    elif "qdd" in seq_name:
        getattr(sing_dd_sched, seq_method)(ord_n, ord_m, dd_qubit, 1, d, sym)
    else:
        getattr(sing_dd_sched, seq_method)(dd_qubit, 1, d, sym)

    # create dd sched with number of reps desired
    dd_sched = IBMQDdSchedule(backend, basis)
    for j in range(num_reps):
        dd_sched.sched += sing_dd_sched.sched

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for pode in range(6):
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(pode))
        sched.encode_podal_state(encoding_qubit, pode, offset)
        sched.sched += dd_sched.sched
        sched.decode_podal_state(encoding_qubit, pode, offset)
        # diagnose acquire delay constraint issues
        aa = backend.get_acquire_alignment()
        r = (num_reps * sing_dd_sched.tot_delay) % aa
        if r != 0:
            #print(f"Diagnosed with {aa - r}dt delay.")
            sched.add_pause(encoding_qubit, aa - r)
        # measure encoded qubits
        sched.add_measurement(encoding_qubit, 0)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)


def haar_fid_decay_dd(N, seq_name, sym, num_reps,
                      pause_padding, d, d_label,
                      backend, basis,
                      encoding_qubit, dd_qubit):
    """
    -->Generates list of experiments which probe fidelity of Pauli podes.
    |0> -- encode_pode_state(offset) -- DD -- decode_pode_state(offset),
    where encoding done on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] identities between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    if "uddx" in seq_name:
        seq_method = "add_uddx"
        ord_n = int(seq_name[4:])
    elif "uddy" in seq_name:
        seq_method = "add_uddy"
        ord_n = int(seq_name[4:])
    elif "qdd" in seq_name:
        seq_method = "add_qdd"
        _, ord_n, ord_m = seq_name.split("_")
        ord_n = int(ord_n)
        ord_m = int(ord_m)
    elif "ur" in seq_name and "x" not in seq_name and "y" not in seq_name:
        seq_method = "add_ur"
        ord_n = int(seq_name[2:])
    else:
        seq_method = "add_" + seq_name
    # get amount of time it takes to run this many reps of dd seq
    qubit = dd_qubit
    time_sched = IBMQDdSchedule(backend, basis)
    if "uddx" in seq_name or "uddy" in seq_name or ("ur" in seq_name and "x" not in seq_name and "y" not in seq_name):
        getattr(time_sched, seq_method)(ord_n, qubit, 1, d, sym)
    elif "qdd" in seq_name:
        getattr(time_sched, seq_method)(ord_n, ord_m, qubit, 1, d, sym)
    else:
        getattr(time_sched, seq_method)(qubit, 1, d, sym)
    T = (time_sched.get_phys_time() * num_reps)
    delay_sched = IBMQDdSchedule(backend, basis)
    delay_sched.add_pause(qubit, pause_padding)
    T += delay_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_haar_{N}_dtype_{d_label}_"
    job_tag += f"fid_decay_{seq_name}_sym_{sym}_encodeqs_{encoding_qubit}_"
    job_tag += f"ddqs_{dd_qubit}_reps_{num_reps}_delay_{d}dt_T_{T}ns"
    exp_tag = job_tag + "_pode_{}"

    # create single rep of DD seq
    sing_dd_sched = IBMQDdSchedule(backend, basis)
    if "uddx" in seq_name or "uddy" in seq_name or ("ur" in seq_name and "x" not in seq_name and "y" not in seq_name):
        getattr(sing_dd_sched, seq_method)(ord_n, dd_qubit, 1, d, sym)
    elif "qdd" in seq_name:
        getattr(sing_dd_sched, seq_method)(ord_n, ord_m, dd_qubit, 1, d, sym)
    else:
        getattr(sing_dd_sched, seq_method)(dd_qubit, 1, d, sym)

    # create dd sched with number of reps desired
    dd_sched = IBMQDdSchedule(backend, basis)
    for j in range(num_reps):
        dd_sched.sched += sing_dd_sched.sched
    # create pause that completes the schedule time
    dd_sched.add_pause(dd_qubit, pause_padding)

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for pode in range(N):
        p, inv_p = u3_params[pode]
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(pode))
        sched.add_u3(encoding_qubit, *p)
        sched.sched += dd_sched.sched
        sched.add_u3(encoding_qubit, *inv_p)
        # diagnose acquire delay constraint issues
        aa = backend.get_acquire_alignment()
        r = (num_reps * sing_dd_sched.tot_delay) % aa
        if r != 0:
            #print(f"Diagnosed with {aa - r}dt delay.")
            sched.add_pause(encoding_qubit, aa - r)
        # measure encoded qubits
        sched.add_measurement(encoding_qubit, 0)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)


def static_haar_fid_decay_uddx(haar_params_list, n, backend, basis,
                               time='min', num_reps=1, encoding_qubits='all',
                               dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- uddx([n], [time]) -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    uddx of order [n] is applied over time [time].
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_uddx(0, n, time, num_reps)
    T = time_sched.get_phys_time()
    if time == 'min':
        time = int(time_sched.get_duration() / num_reps)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_uddx_{n}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_reps_{num_reps}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        sched.add_uddx(dd_qubits, n, time, num_reps)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def static_haar_fid_decay_uddxm(haar_params_list, n, time, backend, basis,
                               encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- uddx([n], [time]) -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    uddx of order [n] is applied over time [time].
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_uddxm(0, n, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_uddxm_{n}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        sched.add_uddxm(dd_qubits, n, time)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def static_haar_fid_decay_uddxe(haar_params_list, n, time, backend, basis,
                               encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- uddx([n], [time]) -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    uddx of order [n] is applied over time [time].
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_uddxe(0, n, time)
    T = time_sched.get_phys_time()
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_uddxe_{n}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        sched.add_uddxe(dd_qubits, n, time)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def static_haar_fid_decay_uddy(haar_params_list, n, backend, basis,
                               time='min', num_reps=1, encoding_qubits='all',
                               dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- uddx([n], [time]) -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    uddx of order [n] is applied over time [time].
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_uddy(0, n, time, num_reps)
    T = time_sched.get_phys_time()
    if time == 'min':
        time = int(time_sched.get_duration() / num_reps)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_uddy_{n}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_reps_{num_reps}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        sched.add_uddy(dd_qubits, n, time, num_reps)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)


def static_haar_fid_decay_qdd(haar_params_list, n, m, backend, basis,
                              time='min', num_reps=1, encoding_qubits='all',
                              dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- qdd([n], [m], [time]) -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    qdd of order [n, m] is applied over time [time].
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_sched = IBMQDdSchedule(backend, basis)
    time_sched.add_qdd(0, n, m, time, num_reps)
    T = time_sched.get_phys_time()
    if time == 'min':
        time = int(time_sched.get_duration() / num_reps)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to schedule
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_fid_decay_qdd_{n}_{m}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_time_{time}dt_reps_{num_reps}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        sched.add_qdd(dd_qubits, n, m, time, num_reps)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

######################################################################
# Tau Test Functions
# Description: Experimentally measures the fidelity of different DD
# sequences as a function of pulse spacing \tau. To be specific,
# suppose we run 25 repetitions of XY4 with tau = 0, tau = 10, tau = 100
# in schedule normalized units (dt).
# Ideally, we'd like to know the fidelity "on average" for a "typical
# state", so perform the DD sequence on a list of haar random states.
######################################################################
def static_haar_tau_test_dd(haar_params_list, seq_name, num_reps, tau,
                            backend, basis, encoding_qubits='all',
                            dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector) and n \in num_id_list of the form:
    |0> -- U3(t,0,0) -- DD seq -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [tau] identities between pulses.
    -->[backend] is needed to specify DD schedule time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # prepend the 'add' to seq name string
    seq_method = 'add_' + seq_name
    # if encoding_qubits set to 'all,' get all backend qubits
    if encoding_qubits == 'all':
        encoding_qubits = list(range(backend.get_number_qubits()))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many reps of dd seq
    time_sched = IBMQDdSchedule(backend, basis)
    getattr(time_sched, seq_method)(0, num_reps, tau)
    T = time_sched.get_phys_time()
    # having parsed input args, set job and exp tags which associate
    # input params to schedule
    phys_tau = ns_time_to_dt(tau, time_sched.dt)
    job_tag = f"pulse_{basis}_"
    job_tag += f"static_haar_tau_test_{seq_name}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_tau_{phys_tau}ns_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        sched = IBMQDdSchedule(backend, basis, name=exp_tag.format(t, p, l))
        sched.add_u3(encoding_qubits, t, p, l)
        getattr(sched, seq_method)(dd_qubits, num_reps, tau)
        sched.add_u3(encoding_qubits, -t, -l, -p)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            sched.add_measurement(q)
        # append this schedule to experiments
        experiments.append(sched)

    return (job_tag, experiments)

def dt_time_to_ns(dt_time, dt):
    '''converts time in normalzied dt units to ns'''
    return (dt_time * dt * 1e9)

def ns_time_to_dt(phys_time, dt):
    '''converts time from physical (ns) to dt normalized time'''
    return (phys_time * 1e-9) / dt
