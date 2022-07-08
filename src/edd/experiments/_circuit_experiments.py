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
from edd.circuit import IBMQDdCircuit
from edd.data import IBMQData

######################################################################
# This file hosts functions which carry out experiments on IBMQ
######################################################################

######################################################################
# THETA SWEEP FUNCTIONS
# Description: Generically, these experiemnts test the efficacy of
# different DD sequences over a range of easy to prepare superposition
# states of the form cos(t/2)|0> + sin(t/2)|1>. This prevents undue
# biasing due to robustness of the |0> state.
######################################################################
def theta_sweep_free(num_ids, backend, encoding_qubits='all',
                     dd_qubits='all', theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- free -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [num_ids] identity gates is applied on [dd_qubits].
    -->[backend] is needed to specify DD circuit time.
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
    time_circ = IBMQDdCircuit(1, ibmq_backend=backend)
    time_circ.add_free(0, num_ids)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"theta_sweep_free_encodeqs_{encoding_qubits}_ddqs_{dd_qubits}_"
    job_tag += f"ids_{num_ids}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a circuit with correct number of qubits
        n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
        n_bits = len(encoding_qubits)
        circ = IBMQDdCircuit(n_qubits, n_bits,
                             name=exp_tag.format(t), ibmq_backend=backend)
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        circ.encode_theta_state(encoding_qubits, t)
        # apply identity gates
        circ.add_free(dd_qubits, num_ids)
        # decode superposition state back into |0> state
        circ.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
                circ.measure(q, idx)
        experiments.append(circ)

    return (job_tag, experiments)

def theta_sweep_dd(seq_name, num_reps, id_pad, backend, encoding_qubits='all',
                   dd_qubits='all', theta_list=np.linspace(0, np.pi, 16)):
    """
    -->Generates list of experiments indexed by t \in theta_list of form:
    |0> -- U3(t,0,0) -- DD seq -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [id_pad] identities between pulses.
    -->[backend] is needed to specify DD circuit time.
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
    time_circ = IBMQDdCircuit(1, ibmq_backend=backend)
    getattr(time_circ, seq_method)(0, num_reps, id_pad)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"theta_sweep_{seq_name}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_idpad_{id_pad}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}"

    # create the list of experiments to run as batch job on QC
    experiments = []
    for t in theta_list:
        # create a circuit with correct number of qubits
        n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
        n_bits = len(encoding_qubits)
        circ = IBMQDdCircuit(n_qubits, n_bits,
                             name=exp_tag.format(t), ibmq_backend=backend)
        # prepare cos(theta / 2) |0> + cos(theta / 2) |1> state
        circ.encode_theta_state(encoding_qubits, t)
        # apply DD sequence with correct parameters
        getattr(circ, seq_method)(0, num_reps, id_pad)
        # decode superposition state back into |0> state
        circ.decode_theta_state(encoding_qubits, t)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
                circ.measure(q, idx)
        experiments.append(circ)

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
def static_haar_fid_decay_free(haar_params_list, num_ids, backend,
                               encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector):
    |0> -- U3(h) -- free -- U3^{dag}(h),
    where U3(h) state is prepared on [encoding_qubits] and
    [num_ids] identity gates is applied on [dd_qubits].
    -->[backend] is needed to specify DD circuit time.
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
    time_circ = IBMQDdCircuit(1, ibmq_backend=backend)
    time_circ.add_free(0, num_ids)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"static_haar_fid_decay_free_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_ids_{num_ids}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
        n_bits = len(encoding_qubits)
        circ = IBMQDdCircuit(n_qubits, n_bits,
                             name=exp_tag.format(t, p, l), ibmq_backend=backend)
        circ.u3(t, p, l, encoding_qubits)
        circ.barrier(encoding_qubits)
        circ.add_free(dd_qubits, num_ids)
        circ.barrier(encoding_qubits)
        circ.u3(-t, -l, -p, encoding_qubits)
        circ.barrier(encoding_qubits)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            circ.measure(q, idx)
        circ.draw()
        # append this circuit to experiments
        experiments.append(circ)

    return (job_tag, experiments)

def static_haar_fid_decay_dd(haar_params_list, seq_name, num_reps, id_pad,
                             backend, encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments indexed by h \in haar_param_list
    (h is 3 vector) and n \in num_id_list of the form:
    |0> -- U3(t,0,0) -- DD seq -- U3^{dag}(t,0,0),
    where U3(t,0,0) state is prepared on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [id_pad] identities between pulses.
    -->[backend] is needed to specify DD circuit time.
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
    time_circ = IBMQDdCircuit(1, ibmq_backend=backend)
    getattr(time_circ, seq_method)(0, num_reps, id_pad)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"static_haar_fid_decay_{seq_name}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_idpad_{id_pad}_T_{T}ns"
    exp_tag = job_tag + "_theta_{}_phi_{}_lambda_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for params in haar_params_list:
        # extract theta, phi, and lambda to construct u3
        t, p, l = params
        # create circ add u3-->DD seq-->u3^{dag}
        n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
        n_bits = len(encoding_qubits)
        circ = IBMQDdCircuit(n_qubits, n_bits,
                             name=exp_tag.format(t, p, l), ibmq_backend=backend)
        circ.u3(t, p, l, encoding_qubits)
        circ.barrier(encoding_qubits)
        getattr(circ, seq_method)(dd_qubits, num_reps, id_pad)
        circ.barrier(encoding_qubits)
        circ.u3(-t, -l, -p, encoding_qubits)
        circ.barrier(encoding_qubits)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            circ.measure(q, idx)
        circ.draw()
        # append this circuit to experiments
        experiments.append(circ)

    return (job_tag, experiments)

def pauli_pode_fid_decay_free(offset, tau, backend,
                               encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments which probe fidelity of Pauli podes.
    |0> -- encode_pode_state(offset) -- free -- decode_pode_state(offset),
    where encoding is prepared on [encoding_qubits] and
    [num_ids] identity gates is applied on [dd_qubits].
    -->[backend] is needed to specify DD circuit time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # if encoding_qubits set to 'all,' get all backend qubits
    n_qubits = backend.get_number_qubits()
    if encoding_qubits == 'all':
        encoding_qubits = list(range(n_qubits))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many id gates
    time_circ = IBMQDdCircuit(n_qubits, ibmq_backend=backend)
    time_circ.add_free(0, tau)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"pauli_pode_offset_{offset}_fid_decay_free_encodeqs"
    job_tag += f"_{encoding_qubits}_ddqs_{dd_qubits}_tau_{tau}_T_{T}ns"
    exp_tag = job_tag + "_pode_{}"

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for pode in range(6):
        # create circuit
        #n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
        n_bits = len(encoding_qubits)
        circ = IBMQDdCircuit(n_qubits, n_bits,
                             name=exp_tag.format(pode), ibmq_backend=backend)
        circ.barrier(encoding_qubits)
        circ.encode_podal_state(encoding_qubits, pode, offset)
        circ.barrier(encoding_qubits)
        circ.add_free(dd_qubits, num_ids)
        circ.barrier(encoding_qubits)
        circ.decode_podal_state(encoding_qubits, pode, offset)
        circ.barrier(encoding_qubits)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            circ.measure(q, idx)
        circ.draw()
        # append this circuit to experiments
        experiments.append(circ)

    return (job_tag, experiments)


def pauli_pode_fid_decay_dd(offset, seq_name, num_reps, tau,
                             backend, encoding_qubits='all', dd_qubits='all'):
    """
    -->Generates list of experiments which probe fidelity of Pauli podes.
    |0> -- encode_pode_state(offset) -- DD -- decode_pode_state(offset),
    where encoding done on [encoding_qubits] and
    [seq_name] DD sequence is applied on [dd_qubits] [num_reps] times
    in repetition with [id_pad] identities between pulses.
    -->[backend] is needed to specify DD circuit time.
    -->If [encoding_qubits] = 'all,' use all backend qubits.
    -->If [dd_qubits] = 'all', then set dd_qubits = encoding_qubits.
    """
    # prepend the 'add' to seq name string
    seq_method = 'add_' + seq_name
    # if encoding_qubits set to 'all,' get all backend qubits
    n_qubits = backend.get_number_qubits()
    if encoding_qubits == 'all':
        encoding_qubits = list(range(n_qubits))
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get amount of time it takes to run this many reps of dd seq
    time_circ = IBMQDdCircuit(n_qubits, ibmq_backend=backend)
    getattr(time_circ, seq_method)(0, 1, tau)
    T = (time_circ.get_phys_time(backend) * num_reps)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = f"circ_pauli_pode_offset_{offset}_"
    job_tag += f"fid_decay_{seq_name}_encodeqs_{encoding_qubits}_"
    job_tag += f"ddqs_{dd_qubits}_reps_{num_reps}_tau_{tau}_T_{T}ns"
    exp_tag = job_tag + "_pode_{}"

    #n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)

    # create single rep of DD seq
    sing_dd_circ = IBMQDdCircuit(n_qubits, n_bits, ibmq_backend=backend)
    getattr(sing_dd_circ, seq_method)(dd_qubits, 1, tau)

    # create dd circ with number of reps desired
    dd_circ = IBMQDdCircuit(n_qubits, n_bits, ibmq_backend=backend)
    for j in range(num_reps):
        dd_circ.extend(sing_dd_circ)

    # now iterate over the haar_params, i.e the "experiment" loop
    experiments = []
    for pode in range(6):
        circ = IBMQDdCircuit(n_qubits, n_bits,
                             name=exp_tag.format(pode), ibmq_backend=backend)
        circ.encode_podal_state(encoding_qubits, pode, offset)
        circ.barrier(encoding_qubits)
        circ.extend(dd_circ)
        circ.barrier(encoding_qubits)
        circ.decode_podal_state(encoding_qubits, pode, offset)
        circ.barrier(encoding_qubits)
        # measure encoded qubits
        for idx, q in enumerate(encoding_qubits):
            circ.measure(q, idx)
        circ.draw()
        # append this circuit to experiments
        experiments.append(circ)

    return (job_tag, experiments)

############################################################
# Type 2 Fidelity Decay
# Here, the haar random states which we try to protect with
# a DD sequence are randomly generated on the fly. In this
# case, there's no chance that XY4 1 rep and XY4 2 rep are
# protecting the same states, so convergence would require
# a much larger sample size of haar random states (we think).
############################################################

############################################################
# Type 3 Fidelity over Entangled States
# Rather than try a set of haar-random states, here we simply
# prepare various entangled states and try to protect them
# with various DD sequences.
############################################################
def bell_fid_decay_free(qubit_pairs, num_ids, backend,
                        dd_qubits='all'):
    """
    -->Generates list of experiments indexed by q \in qubit_pairs_list
    (q is tuple) of the form:
    |0> -- bell_state_prep -- free -- bell_state_decode -- measure
    |0> -- bell_state_prep -- free -- bell_state_decode -- measure,
    where bell_states are prepared between [qubit_pairs] and [num_ids]
    identity gates applied to qubits in [dd_qubits].
    -->[backend] is needed to specify DD circuit time.
    -->If [dd_qubits] = 'all', then set dd_qubits = union([qubit_pairs]).
    """
    # get list of qubits from qubit_pairs
    encoding_qubits = []
    for pair in qubit_pairs:
        encoding_qubits.append(pair[0])
        encoding_qubits.append(pair[1])
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits

    # get T from building a test circuit over a single qubit
    time_circ = IBMQDdCircuit(1, ibmq_backend=backend)
    time_circ.add_free(0, num_ids)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"bell_fid_decay_free_pairs_{qubit_pairs}_ddqs_{dd_qubits}_"
    job_tag += f"ids_{num_ids}_T_{T}ns"
    exp_tag = job_tag + "_{}"

    experiments = []
    # first, add the phi plus bell state
    # want to "access" all qubits from 0 to n since connections
    # are non-linear, but only use/measure len(qubits) of them
    phi_plus_tag = exp_tag.format('phi+')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=phi_plus_tag, ibmq_backend=backend)
    circ.encode_bell_phi_plus(qubit_pairs)
    circ.barrier(encoding_qubits)
    circ.add_free(dd_qubits, num_ids)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_phi_plus(qubit_pairs, measure=True)
    experiments.append(circ)

    # next add phi minus bell state following same steps
    phi_minus_tag = exp_tag.format('phi-')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=phi_minus_tag, ibmq_backend=backend)
    circ.encode_bell_phi_minus(qubit_pairs)
    circ.barrier(encoding_qubits)
    circ.add_free(dd_qubits, num_ids)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_phi_minus(qubit_pairs, measure=True)
    experiments.append(circ)

    # add psi plus bell state
    psi_plus_tag = exp_tag.format('psi+')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=psi_plus_tag, ibmq_backend=backend)
    circ.encode_bell_psi_plus(qubit_pairs)
    circ.barrier(encoding_qubits)
    circ.add_free(dd_qubits, num_ids)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_psi_plus(qubit_pairs, measure=True)
    experiments.append(circ)

    # add psi minus bell state
    psi_minus_tag = exp_tag.format('psi-')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=psi_minus_tag, ibmq_backend=backend)
    circ.encode_bell_psi_minus(qubit_pairs)
    circ.barrier(encoding_qubits)
    circ.add_free(dd_qubits, num_ids)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_psi_minus(qubit_pairs, measure=True)
    experiments.append(circ)

    return (job_tag, experiments)

def bell_fid_decay_dd(qubit_pairs, seq_name, num_reps, id_pad, backend,
                      dd_qubits='all'):
    """
    -->Generates list of experiments indexed by q \in qubit_pairs_list
    (q is tuple) of the form:
    |0> -- bell_state_prep -- dd_seq -- bell_state_decode -- measure
    |0> -- bell_state_prep -- dd_seq -- bell_state_decode -- measure,
    where bell_states are prepared between [qubit_pairs] and
    dd_seq is applied to [dd_qubits] for [num_reps] repetitions with
    [id_pad] identities between dd pulses.
    -->[backend] is needed to specify DD circuit time.
    -->If [dd_qubits] = 'all', then set dd_qubits = union([qubit_pairs])
    """
    seq_method = 'add_' + seq_name
    # get list of qubits from qubit_pairs
    encoding_qubits = []
    for pair in qubit_pairs:
        encoding_qubits.append(pair[0])
        encoding_qubits.append(pair[1])
    # set up which qubits to run DD sequence on (not same as those
    # we measure necessarily unless 'all' is specified)
    if dd_qubits == 'all':
        dd_qubits = encoding_qubits
    # get T from building a test circuit over a single qubit
    time_circ = IBMQDdCircuit(1, ibmq_backend=backend)
    getattr(time_circ, seq_method)(0, num_reps, id_pad)
    T = time_circ.get_phys_time(backend)
    # hvaing parsed input args, set job and exp tags which associate
    # input params to circuit
    job_tag = "circ_"
    job_tag += f"bell_fid_decay_{seq_name}_pairs_{qubit_pairs}_ddqs_{dd_qubits}_"
    job_tag += f"reps_{num_reps}_idpad_{id_pad}_T_{T}ns"
    exp_tag = job_tag + "_{}"

    experiments = []
    # first, add the phi plus bell state
    # want to "access" all qubits from 0 to n since connections
    # are non-linear, but only use/measure len(qubits) of them
    phi_plus_tag = exp_tag.format('phi+')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=phi_plus_tag, ibmq_backend=backend)
    circ.encode_bell_phi_plus(qubit_pairs)
    circ.barrier(encoding_qubits)
    getattr(circ, seq_method)(dd_qubits, num_reps, id_pad)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_phi_plus(qubit_pairs, measure=True)
    experiments.append(circ)

    # next add phi minus bell state following same steps
    phi_minus_tag = exp_tag.format('phi-')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=phi_minus_tag, ibmq_backend=backend)
    circ.encode_bell_phi_minus(qubit_pairs)
    circ.barrier(encoding_qubits)
    getattr(circ, seq_method)(dd_qubits, num_reps, id_pad)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_phi_minus(qubit_pairs, measure=True)
    experiments.append(circ)

    # add psi plus bell state
    psi_plus_tag = exp_tag.format('psi+')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=psi_plus_tag, ibmq_backend=backend)
    circ.encode_bell_psi_plus(qubit_pairs)
    circ.barrier(encoding_qubits)
    getattr(circ, seq_method)(dd_qubits, num_reps, id_pad)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_psi_plus(qubit_pairs, measure=True)
    experiments.append(circ)

    # add psi minus bell state
    psi_minus_tag = exp_tag.format('psi-')
    n_qubits = max(max(encoding_qubits), max(dd_qubits)) + 1
    n_bits = len(encoding_qubits)
    circ = IBMQDdCircuit(n_qubits, n_bits,
                         name=psi_minus_tag, ibmq_backend=backend)
    circ.encode_bell_psi_minus(qubit_pairs)
    circ.barrier(encoding_qubits)
    getattr(circ, seq_method)(dd_qubits, num_reps, id_pad)
    circ.barrier(encoding_qubits)
    # this decoding automatically adds measurements
    circ.decode_bell_psi_minus(qubit_pairs, measure=True)
    experiments.append(circ)

    return (job_tag, experiments)
