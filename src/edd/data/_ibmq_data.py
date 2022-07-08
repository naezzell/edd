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
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import scipy
import yaml
import itertools
import copy

# type check import
from qiskit.result import Result


class IBMQData():
    """
    This class holds data from IBMQ experiments and allows easy manipulation
    such as bootstrapping, plotting, and saving data.
    """

    def __init__(self, raw_data=None, name='test', working_label=None):
        """ [raw_data] is Result object from submitting IBMQ job or
        this object cast as a dict and [name] just names the raw_data set."""
        self.name = name
        # cast as dict to utilize common methods (not data class ones)
        self.raw_data = raw_data
        if self.raw_data is not None:
            if isinstance(raw_data, Result):
                self.raw_data = [raw_data.to_dict()]
            elif isinstance(raw_data, dict):
                self.raw_data = [raw_data]
            elif isinstance(raw_data, list):
                self.raw_data = raw_data

        # convert data to more workable form
        if self.raw_data is not None:
            self.collate_to_results()
            self.results_to_tuple_data()
            if working_label is not None:
                self.change_working_data_by_label(working_label)
        else:
            self.results = None
            self.tuple_data = None
            self.working_data = None
            self.working_label = working_label

        # last member data is "plottable data"
        self.plot_data = None
        self.grover_data = None

        return

    def add_raw_data(self, more_raw_data):
        """
        Appends [more_raw_data] to self.raw_data. If not a list, makes
        self.raw_data into one.
        This method is useful for concatenating similar raw_data sets.
        """
        # convert more_raw_data to right type
        if not isinstance(more_raw_data, list) and not isinstance(more_raw_data, dict):
            if isinstance(more_raw_data, Result):
                more_raw_data = more_raw_data.to_dict()
            else:
                raise ValueError("added raw_data must be list, dict, or Result object")

        if self.raw_data is None:
            if isinstance(more_raw_data, list):
                self.raw_data = more_raw_data
            else:
                self.raw_data = [more_raw_data]
        else:
            if isinstance(more_raw_data, list):
                self.raw_data.extend(more_raw_data)
            else:
                self.raw_data.append(more_raw_data)
        return

    def save_raw_data(self, fname):
        """
        Saves raw_data in self.raw_data to '[fname].yml'
        """
        if fname[-4::] != '.yml':
            fname += '.yml'
        try:
            with open(f'{fname}', 'w') as outfile:
                yaml.dump(self.raw_data, outfile)
        except ValueError:
            e = "Cannot save self.raw_data to file when it is None type."
            raise ValueError(e)
        return

    def load_raw_data(self, fname):
        """
        Loads raw_datas (as dict) from [fname] into [self.raw_data]
        """
        # check if fname has .yml extension already and add if not
        if fname[-4::] != '.yml':
            fname += '.yml'

        # load in raw_data
        with open(fname, 'r') as infile:
            new_raw_data = yaml.safe_load(infile)
        # add new loaded raw_datas in with current raw_datas
        self.add_raw_data(new_raw_data)

    ##################################################
    # "Pre-wrangle" wrangle methods
    ##################################################
    def collate_to_results(self):
        self.results = []
        for result_dict in self.raw_data:
            r_dict_list = result_dict['results']
            for r_dict in r_dict_list:
                r_dict['backend_name'] = result_dict['backend_name']
                r_dict['date'] = result_dict['date'].__str__()
            self.results.extend(result_dict['results'])

    def results_to_tuple_data(self):
        self.tuple_data = []
        for result in self.results:
            back_name = result['backend_name'].split("_")[1]
            back_str = f"backName_{back_name}_"
            date_str = f"_date_{result['date']}_"
            tup = (back_str + result['header']['name'] + date_str,
                   result['header']['memory_slots'], result['data']['counts'])
            self.tuple_data.append(tup)
        return

    def format_data(self):
        self.collate_to_results()
        self.results_to_tuple_data()
        return

    def change_working_data_by_label(self, *labels):
        self.working_label = labels
        self.working_data = []
        for result in self.tuple_data:
            if np.array([x in result[0] for x in labels]).all():
                self.working_data.append(result)
        return

    ##################################################
    # Wrangle and Plotting Methods
    ##################################################
    def apply_error_mitigation(self):
        """
        If this dataset contains error-mitigation circuits, then we want
        to offset obtained results appropriately.
        """
        self.change_working_data_by_label('error_mitigate_0')
        err0_data = self.working_data[0][2]
        tot = err0_data['0x0'] + err0_data['0x1']
        m00 = err0_data['0x0'] / tot
        m01 = err0_data['0x1'] / tot
        self.change_working_data_by_label('error_mitigate_1')
        err1_data = self.working_data[0][2]
        tot = err1_data['0x0'] + err1_data['0x1']
        m10 = err1_data['0x0'] / tot
        m11 = err1_data['0x1'] / tot
        calib_mat = np.array([[m00, m01], [m10, m11]])
        # adjust raw counts for other data
        for idx in range(len(self.tuple_data)):
            tup = self.tuple_data[idx]
            count_vec = np.array([tup[2]['0x0'], tup[2]['0x1']])
            new_vec = np.matmul(calib_mat, count_vec)
            new_counts = {'0x0': int(new_vec[0]), '0x1': int(new_vec[1])}
            self.tuple_data[idx] = (tup[0], tup[1], new_counts)

    def save_plot_data(self, fname):
        """
        Given plot data, saves to file in format readable by Mathematica.
        """
        plot_data = self.plot_data
        with open(fname + '.csv', 'w') as f:
            for idx in range(len(plot_data[0])):
                f.write(f"{plot_data[0][idx]}, {plot_data[1][idx]}, {plot_data[2][idx]}\n")
        return fname

    def wrangle_theta_sweep(self):
        """
        Wrangles [self.working_data] assuming it was produced
        from a theta sweep experiment.
        """
        # iterate over the list of runs and get global information
        theta_vals = []
        theta_fids = []
        theta_2stds = []

        for result in self.working_data:
            # extract tau value from experiment label
            split_label = result[0].split('_')[2::]
            theta_idx = split_label.index('theta')
            theta_val = split_label[theta_idx + 1]
            theta_vals.append(float(theta_val))

            num_qubits_measured, count_list = result
            # iterate over counts for each of the haar states
            succ_counts = []
            fail_counts = []
            for state_data in count_list:
                succ, fail = get_success_failure_count(state_data, num_qubits_measured)
                succ_counts.append(succ)
                fail_counts.append(fail)

            # bootstrap over haar states
            boot_sample = beta_bayesian_bootstrap(succ_counts, fail_counts, succ_counts[0] + fail_counts[0])
            theta_fids.append(np.mean(boot_sample))
            theta_2stds.append(2 * np.std(boot_sample))

        # cast data lists to arrays for convenience
        theta_vals = np.array(theta_vals)
        theta_fids = np.array(theta_fids)
        theta_2stds = np.array(theta_2stds)
        fid_data = (theta_vals, theta_fids, theta_2stds)
        self.plot_data = fid_data

        return fid_data

    def plot_theta_sweep(self, title=''):
        """
        Plots [self.plot_data] assuming it was produced
        from a theta sweep experiment.
        """
        # unpack data
        thetas, fids, l_ci, u_ci, p_errs = self.plot_data
        # divide thetas by np to make easier to display
        thetas = thetas / np.pi
        # get diff between upper ci and mean value for errbar formatting
        fid_up = u_ci - fids
        fid_low = fids - l_ci

        # make thetas into more readable format
        # finally plot it and add labels
        plt.errorbar(thetas, fids, yerr=(fid_low, fid_up), fmt='.')
        plt.xlabel('theta / pi')
        plt.ylabel('fidelity')
        plt.title(title)

        return plt

    def wrangle_fid_decay(self):
        """
        Wrangles [self.working_data] assuming it was produced
        from a fidelity decay experiment.
        Outputs:
        T_vals: array of DD sequence times in microseconds
        haar_fids: array of fidelities associated to each time
        haar_2stds: array of 2 standard deviation values associated with fids
        """
        T_vals = []
        haar_fids = []
        haar_2stds = []

        contracted_data = contract_by_T(self.working_data)

        for Tval, count_data in contracted_data.items():
            T_vals.append(float(Tval))
            num_qubits_measured, count_list = count_data
            # iterate over counts for each of the haar states
            succ_counts = []
            fail_counts = []
            for state_data in count_list:
                succ, fail = get_success_failure_count(state_data, num_qubits_measured)
                succ_counts.append(succ)
                fail_counts.append(fail)

            # bootstrap over haar states
            boot_sample = beta_bayesian_bootstrap(succ_counts, fail_counts, succ_counts[0] + fail_counts[0])
            haar_fids.append(np.mean(boot_sample))
            haar_2stds.append(2 * np.std(boot_sample))

        # cast data lists to arrays for convenience
        T_vals = np.array(T_vals) / 1000 # convert to \mu s
        haar_fids = np.array(haar_fids)
        haar_2stds = np.array(haar_2stds)

        self.plot_data = (T_vals, haar_fids, haar_2stds)

        return self.plot_data

    def wrangle_mistake_fid_decay(self):
        """
        Wrangles [self.working_data] assuming it was produced
        from a fidelity decay experiment.
        Outputs:
        T_vals: array of DD sequence times in microseconds
        haar_fids: array of fidelities associated to each time
        haar_2stds: array of 2 standard deviation values associated with fids
        """
        T_vals = []
        haar_fids = []
        haar_2stds = []

        contracted_data = contract_by_fixed_data_T(self.working_data)

        for Tval, count_data in contracted_data.items():
            T_vals.append(float(Tval))
            num_qubits_measured, count_list = count_data
            count_list = fix_measurement_mistake(count_list)
            # iterate over counts for each of the haar states
            succ_counts = []
            fail_counts = []
            for state_data in count_list:
                succ, fail = get_success_failure_count(state_data, num_qubits_measured)
                succ_counts.append(succ)
                fail_counts.append(fail)

            # bootstrap over haar states
            boot_sample = beta_bayesian_bootstrap(succ_counts, fail_counts, succ_counts[0] + fail_counts[0])
            haar_fids.append(np.mean(boot_sample))
            haar_2stds.append(2 * np.std(boot_sample))

        # cast data lists to arrays for convenience
        T_vals = np.array(T_vals) / 1000 # convert to \mu s
        haar_fids = np.array(haar_fids)
        haar_2stds = np.array(haar_2stds)

        self.plot_data = (T_vals, haar_fids, haar_2stds)

        return self.plot_data

    def plot_fid_decay(self, with_fit=False, title='', legend=None):
        """
        Plots [self.plot_data] assuming it was produced
        from fidelity decay experiment.
        """
        fig, ax = plt.subplots()
        times, fids, errs = self.plot_data
        prop = ax._get_lines.prop_cycler
        color = next(prop)['color']
        if with_fit is True and legend is None:
            legend = []
        # simple plot
        if with_fit is False:
            ax.errorbar(times, fids, yerr=errs, fmt='.', color=color)
            ax.set_xlabel('time ($\\mu$s)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
            # plot with simple_exp fitting
        else:
            # get fit and plot it
            fit_params, par_params, sum_res = self.bootstrap_fit_aug_exp()
            tau = fit_params[0]
            tau_err = fit_params[1]
            a0 = par_params[0]
            a0_err = par_params[1]
            ax.errorbar(times, fids, yerr=errs, fmt='.', color=color)
            ord_t = np.array(sorted(times))
            ax.plot(ord_t, aug_exp_func([a0, tau], ord_t), color=color)
            # format the legend with the decay constant and 2 std value
            st_tau = f'{tau:.2f}'
            st_tau_err = f'{tau_err:.2f}'
            legend.append(f'$\\tau$ = {st_tau} $\pm$ {st_tau_err}')
            # add the 95% confidence intervals
            ax.fill_between(ord_t, aug_exp_func([a0+a0_err, tau+tau_err], ord_t), aug_exp_func([a0-a0_err, tau-tau_err], ord_t),
                            color=color, alpha=.2)
            # add labels
            ax.set_xlabel('time ($\mu$s)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
        if legend is not None:
            ax.legend(legend)

        return (fig, ax)

    def plot_cosfid_decay(self, with_fit=False, title='', seqs=None):
        """
        Plots [self.plot_data] assuming it was produced
        from fidelity decay experiment.
        """
        times, fids, errs = self.plot_data
        fig, ax = plt.subplots()
        prop = ax._get_lines.prop_cycler
        if seqs is None:
            seqs = ['' for _ in range(len(self.plot_data))]
        s_idx = 0
        legend = []
        color = next(prop)['color']
        # simple plot
        if with_fit is False:
            ax.errorbar(times, fids, yerr=errs, fmt='.', color=color)
            ax.set_xlabel('time ($\\mu$s)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
            legend.append(seqs[s_idx])
            # plot with simple_exp fitting
        else:
            # get fit and plot it
            fit_params, par_params, gam_params, sum_res = self.bootstrap_fit_aug_cosexp()
            tau = fit_params[0]
            tau_err = fit_params[1]
            a0 = par_params[0]
            a0_err = par_params[1]
            gam = gam_params[0]
            gam_err = gam_params[1]
            ax.errorbar(times, fids, yerr=errs, fmt='.', color=color)
            ord_t = np.array(sorted(times))
            ax.plot(ord_t, aug_cosexp_func([a0, tau, gam], ord_t), color=color)
            # format the legend with the decay constant and 2 std value
            st_tau = f'{tau:.2f}'
            st_tau_err = f'{tau_err:.2f}'
            if np.abs(gam) > 1000:
                st_gam = "$\\infty$"
                st_gam_err = "0"
            else:
                st_gam = f'{gam:.2f}'
                st_gam_err = f'{gam_err:.2f}'
            legend.append(f'{seqs[s_idx]}, $\\tau$ = {st_tau} $\pm$ {st_tau_err}, $\\gamma$ = {st_gam} $\pm$ {st_gam_err}')
            # add the 95% confidence intervals
            ax.fill_between(ord_t, aug_cosexp_func([a0+a0_err, tau+tau_err, gam+gam_err], ord_t), aug_cosexp_func([a0-a0_err, tau-tau_err, gam-gam_err], ord_t),
                            color=color, alpha=.2)
            # add labels
            ax.set_xlabel('time ($\mu$s)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
            s_idx += 1
        ax.legend(legend)

        return (fig, ax)

    def wrangle_tau_test(self):
        """
        Wrangles [self.working_data] assuming it was produced
        from a tau test.
        Outputs:
        tau_vals: array of DD sequence times in microseconds
        haar_fids: array of fidelities associated to each tau
        haar_2stds: array of 2 standard deviation values associated with fids
        """
        tau_vals = []
        haar_fids = []
        haar_2stds = []

        contracted_data = contract_by_tau(self.working_data)

        for tau, count_data in contracted_data.items():
            tau_vals.append(float(tau))
            num_qubits_measured, count_list = count_data
            # iterate over counts for each of the haar states
            succ_counts = []
            fail_counts = []
            for state_data in count_list:
                succ, fail = get_success_failure_count(state_data, num_qubits_measured)
                succ_counts.append(succ)
                fail_counts.append(fail)

            # bootstrap over haar states
            boot_sample = beta_bayesian_bootstrap(succ_counts, fail_counts, succ_counts[0] + fail_counts[0])
            haar_fids.append(np.mean(boot_sample))
            haar_2stds.append(2 * np.std(boot_sample))

        # cast data lists to arrays for convenience
        tau_vals = np.array(tau_vals)
        haar_fids = np.array(haar_fids)
        haar_2stds = np.array(haar_2stds)

        self.plot_data = (tau_vals, haar_fids, haar_2stds)

        return self.plot_data

    def plot_tau_test(self, with_fit=False, title='', seqs=None):
        """
        Plots [self.plot_data] assuming it was produced
        from tau test experiment.
        """
        times, fids, errs = self.plot_data
        fig, ax = plt.subplots()
        prop = ax._get_lines.prop_cycler
        if seqs is None:
            seqs = ['' for _ in range(len(self.plot_data))]
        s_idx = 0
        legend = []
        color = next(prop)['color']
        # simple plot
        if with_fit is False:
            ax.errorbar(times, fids, yerr=errs, fmt='.', color=color)
            ax.set_xlabel('delay time (dt)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
            legend.append(seqs[s_idx])
            # plot with simple_exp fitting
        else:
            # get fit and plot it
            fit_params, par_params, gam_params, sum_res = self.bootstrap_fit_aug_cosexp()
            tau = fit_params[0]
            tau_err = fit_params[1]
            a0 = par_params[0]
            a0_err = par_params[1]
            gam = gam_params[0]
            gam_err = gam_params[1]
            ax.errorbar(times, fids, yerr=errs, fmt='.', color=color)
            ord_t = np.array(sorted(times))
            ax.plot(ord_t, aug_cosexp_func([a0, tau, gam], ord_t), color=color)
            # format the legend with the decay constant and 2 std value
            st_tau = f'{tau:.2f}'
            st_tau_err = f'{tau_err:.2f}'
            if np.abs(gam) > 1000:
                st_gam = "$\\infty$"
                st_gam_err = "0"
            else:
                st_gam = f'{gam:.2f}'
                st_gam_err = f'{gam_err:.2f}'
            legend.append(f'{seqs[s_idx]}, $\\tau$ = {st_tau} $\pm$ {st_tau_err}, $\\gamma$ = {st_gam} $\pm$ {st_gam_err}')
            # add the 95% confidence intervals
            ax.fill_between(ord_t, aug_cosexp_func([a0+a0_err, tau+tau_err, gam+gam_err], ord_t), aug_cosexp_func([a0-a0_err, tau-tau_err, gam-gam_err], ord_t),
                            color=color, alpha=.2)
            # add labels
            ax.set_xlabel('delay time (dt)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
            s_idx += 1
        ax.legend(legend)

        return (fig, ax)

    def wrangle_idpad_test(self):
        """
        Wrangles [self.working_data] assuming it was produced
        from a tau test.
        """
        idpad_vals = []
        haar_fids = []
        haar_2stds = []

        contracted_data = contract_by_idpad(working_data)

        for id_pad, count_data in contracted_data.items():
            idpad_vals.append(id_pad)
            num_qubits_measured, count_list = count_data
            # iterate over counts for each of the haar states
            succ_counts = []
            fail_counts = []
            for state_data in count_list:
                succ, fail = get_success_failure_count(state_data, num_qubits_measured)
                succ_counts.append(succ)
                fail_counts.append(fail)

            # bootstrap over haar states
            boot_sample = beta_bayesian_bootstrap(succ_counts, fail_counts, succ_counts[0] + fail_counts[0])
            haar_fids.append(np.mean(boot_sample))
            haar_2stds.append(2 * np.std(boot_sample))

        # cast data lists to arrays for convenience
        idpad_vals = np.array(idpad_vals)
        haar_fids = np.array(haar_fids)
        haar_2stds = np.array(haar_2stds)

        self.plot_data = (tau_vals, haar_fids, haar_2stds)

        return self.plot_data

    def wrangle_grover(self):
        """
        Wrangles [self.working_data] assuming it was produced
        from a grover experiment.
        """
        marker_states = []
        haar_fids = []
        haar_2stds = []
        prob_amp_list = []

        contracted_data = contract_by_marker(self.working_data)

        for markers, count_data in contracted_data.items():
            # turn markers into list with no whitespace
            markers = markers.replace(" ", "").split(',')
            marker_states.append(markers)
            num_qubits_measured, count_list = count_data
            # count_list is list of only one element in this case (comes from
            # haar experiments where list of length # haar states)
            count_dict = hex_dict_to_bin_dict(count_list[0], num_qubits_measured)
            # flip the keys to match normal physics notation
            phys_count_dict = {}
            for key, value in count_dict.items():
                phys_count_dict[key[::-1]] = value
            succ, fail = get_grover_succ_and_fail_num(phys_count_dict, num_qubits_measured, markers)
            total_trials = succ + fail
            prob_amps = {}
            for key, value in phys_count_dict.items():
                prob_amps[key] = (value / total_trials)

            prob_amp_list.append(prob_amps)

            boot_sample = beta_bayesian_bootstrap([succ], [fail], total_trials)
            haar_fids.append(np.mean(boot_sample))
            haar_2stds.append(2 * np.std(boot_sample))

        # cast data lists to arrays for convenience
        haar_fids = np.array(haar_fids)
        haar_2stds = np.array(haar_2stds)

        self.grover_data = (marker_states, haar_fids, haar_2stds, prob_amp_list)

        return self.grover_data

    def plot_grover_heatmap(self):
        """
        Plots a heatmap of grover data akin to that in
        Figgatt et. al. Nature paper of 2017.
        """
        markers, fids, errs, amp_dicts = self.grover_data
        # get number of qubits in a measurement
        num_q = len(list(amp_dicts[0].keys())[0])
        # form list of (ordered) basis states
        basis_states = [''.join(i) for i in itertools.product('01', repeat=num_q)]

        # format data for heatmap
        y_labels = []
        prob_amps = np.zeros((len(markers), len(basis_states)))
        # format side success probabilities
        side_fids = []
        for idx in range(len(fids)):
            fid = f"{fids[idx] * 100:.1f}"
            err = f"{errs[idx] * 100:.1f}"
            side_fids.append(f"{fid}({err})")
        side_fids = np.asarray([side_fids])
        for (i, m) in enumerate(markers):
            y_lab = str(m).strip('[]').replace("'", "")
            y_labels.append(y_lab)
            for (j, state) in enumerate(basis_states):
                 prob = amp_dicts[i].get(state, 0)
                 prob_amps[(i, j)] = prob


        # format the plot data
        fig = plt.figure(figsize=(10,5))
        ax1 = plt.subplot2grid((20,20), (0,0), colspan=16, rowspan=19)
        ax2 = plt.subplot2grid((20,20), (0,16), colspan=2, rowspan=19)
        ax3 = plt.subplot2grid((20,20), (0,18), colspan=1, rowspan=19)
        # set up main plot
        sns.heatmap(prob_amps, ax=ax1, linewidth=0.5, xticklabels=basis_states, cbar=False)
        ax1.set_yticklabels(y_labels, rotation=0, horizontalalignment='right')
        ax1.set(xlabel='Detected State', ylabel='Marked State')
        # set up the side bar with marked state success probability
        sns.heatmap(np.array([fids]), ax=ax2, cmap = "YlGnBu",  annot=side_fids, fmt="", cbar=False, xticklabels=False, yticklabels=False)

        fig.colorbar(ax1.get_children()[0], cax=ax3, orientation="vertical")

        return plt

############################################################
# Fitting Utilities
############################################################
    def perform_aug_exp_fit(self, seed=None):
        """
        Takes plot_data (x, y, yerr) & fits a function of the form
        F(t) = aug_exp_func(p, t)
        such that sum{[(F(t) - y)/yerr]^2} is minimized, i.e.
        least squares weighted by errors.
        """
        return perform_aug_exp_fit(self.plot_data, seed)

    def perform_aug_cosexp_fit(self, seed=None):
        """
        Takes plot_data (x, y, yerr) & fits a function of the form
        F(t) = aug_cosexp_func(p, t)
        such that sum{[(F(t) - y)/yerr]^2} is minimized, i.e.
        least squares weighted by errors.
        """
        return perform_aug_cosexp_fit(self.plot_data, seed)

    def bootstrap_fit_aug_exp(self, boot_samps=1000, seed=None):
        """
        Bootstrap fit of [plots_data] to the aug_exp function.
        """
        plot_data = self.plot_data
        a0_list = []
        tau_list = []
        for i in range(boot_samps):
            # generate sample of plot_data
            samp_plot_data = gen_time_series_sample(plot_data)
            # fit this data and add to lists
            tau_fit, a0_fit, _ = perform_aug_exp_fit(samp_plot_data, seed)
            a0_list.append(a0_fit[0])
            tau_list.append(tau_fit[0])

        avg_a0 = np.mean(a0_list)
        a0_std = 2*np.std(a0_list)
        avg_tau = np.mean(tau_list)
        tau_std = 2*np.std(tau_list)

        # obtain sum of residuals squared
        t, fid, _ = plot_data
        rsq_sum = np.sum((aug_exp_func([avg_a0, avg_tau], t) - fid)**2)

        return [avg_tau, tau_std], [avg_a0, a0_std], rsq_sum

    def bootstrap_fit_aug_cosexp(self, boot_samps=1000, seed=None):
        """
        Bootstrap fit of [plots_data] to the aug_exp function.
        """
        plot_data = self.plot_data
        a0_list = []
        tau_list = []
        gam_list = []

        # do fit on original data
        tau_fit, a0_fit, gam_fit, _ = perform_aug_cosexp_fit(plot_data, seed)
        a0 = a0_fit[0]
        a0_list.append(a0)
        tau = tau_fit[0]
        tau_list.append(tau)
        gam = gam_fit[0]
        gam_list.append(gam)
        seed = [a0, tau, gam]

        for i in range(boot_samps):
            # generate sample of plot_data
            samp_plot_data = gen_time_series_sample(plot_data)
            # fit this data and add to lists
            tau_fit, a0_fit, gam_fit, _ = perform_aug_cosexp_fit(samp_plot_data, seed)
            a0_list.append(a0_fit[0])
            tau_list.append(tau_fit[0])
            gam_list.append(gam_fit[0])

        avg_a0 = np.mean(a0_list)
        a0_std = 2*np.std(a0_list)
        avg_tau = np.mean(tau_list)
        tau_std = 2*np.std(tau_list)
        avg_gam = np.mean(gam_list)
        gam_std = 2*np.std(gam_list)

        # obtain sum of residuals squared
        t, fid, _ = plot_data
        rsq_sum = np.sum((aug_cosexp_func([avg_a0, avg_tau, avg_gam], t) - fid)**2)

        return [avg_tau, tau_std], [avg_a0, a0_std], [avg_gam, gam_std], rsq_sum


# a better version than Bibek fit for now
def aug_exp_func(p, t):
    """
    F(t) = 1/2(p[0]*e^{-t / p[1]} + 1)
    """
    # define f(t)
    return (1/2)*(p[0]*np.exp(-t/p[1]) + 1)

def aug_cosexp_func(p, t):
    """
    F(t) = 1/2(p[0]*e^{-t / p[1]} + 1)*Cos(p[2]*t)
    """
    mid_val = p[0]*np.exp(-t/p[1])*np.cos(t/p[2])
    return (1/2)*(mid_val + 1)


def perform_aug_exp_fit(plot_data, seed=None):
    """
    Takes plot_data (x, y, yerr) & fits a function of the form
    F(t) = aug_exp_func(p, t)
    such that sum{[(F(t) - y)/yerr]^2} is minimized, i.e.
    least squares weighted by errors.
    """
    t, fid, yerr = plot_data
    ysig = (yerr / 2)

    def err_func(p, t, fid, err):
        return ((fid - aug_exp_func(p, t)) / err)**2

    if seed is None:
        p_init = [1.0, 50.0]
    else:
        p_init = seed
    out = scipy.optimize.leastsq(err_func, p_init,
                           args=(t, fid, ysig), full_output=1)
    # collect fitting parameters
    a0, tau = out[0]
    covar = out[1]
    # get sum of squared residuals
    res_sum = np.sum((aug_exp_func([a0, tau], t) - fid)**2)
    red_chi_sq = res_sum / (len(fid)- 2)
    # get fitting errors provided covar is not None
    if covar is not None:
        if covar[0][0] is not None:
            a0_err = np.sqrt(covar[0][0] * red_chi_sq)
        else:
            a0_err = 'undetermined'
        if covar[1][1] is not None:
            tau_err = np.sqrt(covar[1][1] * red_chi_sq)
        else:
            tau_err = 'undetermined'
    else:
        a0_err = 'undetermined'
        tau_err = 'undetermined' 

    return [tau, tau_err], [a0, a0_err], res_sum

def perform_aug_cosexp_fit(plot_data, seed=None):
    """
    Takes plot_data (x, y, yerr) & fits a function of the form
    F(t) = aug_cosexp_func(p, t)
    such that sum{[(F(t) - y)/yerr]^2} is minimized, i.e.
    least squares weighted by errors.
    """
    t, fid, yerr = plot_data
    ysig = (yerr / 2)

    def err_func(p, t, fid, err):
        return ((fid - aug_cosexp_func(p, t)) / err)**2

    if seed is None:
        p_init = [1.0, 50.0, 150.0]
    else:
        p_init = seed
    out = scipy.optimize.leastsq(err_func, p_init,
                           args=(t, fid, ysig), full_output=1)

    # collect fitting parameters
    a0, tau, gam = out[0]
    covar = out[1]
    # get sum of squared residuals
    res_sum = np.sum((aug_cosexp_func([a0, tau, gam], t) - fid)**2)
    red_chi_sq = res_sum / (len(fid) - 2)
    # get errors if not None
    if covar is not None:
        if covar[0][0] is not None:
            a0_err = np.sqrt(covar[0][0] * red_chi_sq)
        else:
            a0_err = 'undetermined'
        if covar[1][1] is not None:
            tau_err = np.sqrt(covar[1][1] * red_chi_sq)
        else:
            tau_err = 'undetermined'
        if covar[2][2] is not None:
            gam_err = np.sqrt(covar[2][2] * red_chi_sq)
        else:
            gam_err = 'undetermined'
    else:
        a0_err = 'undetermined'
        tau_err = 'undetermined'
        gam_err = 'undetermined'

    return [tau, tau_err], [a0, a0_err], [gam, gam_err], res_sum

def gen_time_series_sample(plot_data):
    """
    Given [plot_data] of the form [xdata, ydata, yerrs],
    generates a new sample data set of the form
    [xsamp, ysamp, yerrs] useful for bootstrapping.

    Assumes yerr is 2 \sigma.
    """
    xs, ys, errs = plot_data
    samp_ys = []
    # get 1 sigma errs instead for sampling
    errs = errs / 2
    for i in range(len(ys)):
        # generate sample y from normal dist
        s_y = np.random.normal(ys[i], errs[i])
        samp_ys.append(s_y)

    return [xs, samp_ys, errs]
###########################################################################
#                      HELPER FUNCTIONS
###########################################################################
def hex_to_bin(hexstr, num_bits):
    """
    Converts [hexstr] which is hexadecimal as string
    into binary with [num_bits] padded 0s. For example,
    0x00 --> 0000 if num_bits is 4.
    """
    base = 16 # hexadecimal is base 16
    return bin(int(hexstr, base))[2:].zfill(num_bits)


def hex_dict_to_bin_dict(hexdict, num_bits):
    """
    Converts a dictionary with hexadecimal (str) keys into binary (str) keys
    """
    return {hex_to_bin(key, num_bits): hexdict[key] for key in hexdict}

def count_0s(istr):
    """
    Counts the number of 0s in [istr] and returns that integer.
    """
    num_0s = 0
    for s in istr:
        if s == '0':
            num_0s += 1
    return num_0s

def dict_data_to_n0_array(data_dict):
    """
    Converts a dictionary of the form {'101': 2, '000': 1}-->[101, 101, 000]
    -->[1, 1, 3] where last list specifies how many zeros each bitstring
    has. (We don't actually compute second list, but helps to see it.)
    """
    # populate data_list with relative frequencies from dict
    data_list = []
    for key, value in data_dict.items():
        for i in range(value):
            # get number of zeros in this string and append to list
            data_list.append(count_0s(key))

    return np.array(data_list)

def contract_by_T(working_data):
    """
    Given that self.working_data is populated by tuple data,
    contracts data into [contracted_data] dict by combining all
    data with shared T into single data point, i.e.
    contracted_data[T] = [{counts_dict1}, {counts_dict2}, ...]
    where each counts_dict is over same T.
    """
    contracted_data = {}
    for tup_data in working_data:
        split_label = tup_data[0].split('_')
        T_idx = split_label.index('T')
        # the [0:-2] removes the trailing units of ns
        T_val = f"{float(split_label[T_idx + 1][0:-2]):.4f}"

        if T_val in contracted_data:
            # check that number of qubits measured is the same
            assert(contracted_data[T_val][0] == tup_data[1])
            # append additional data to T_val
            contracted_data[T_val][1].append(copy.deepcopy(tup_data[2]))
        else:
            contracted_data[T_val] = (tup_data[1], [copy.deepcopy(tup_data[2])])

    return contracted_data

def contract_by_fixed_data_T(working_data):
    """
    Given that self.working_data is populated by tuple data,
    contracts data into [contracted_data] dict by combining all
    data with shared T into single data point, i.e.
    contracted_data[T] = [{counts_dict1}, {counts_dict2}, ...]
    where each counts_dict is over same T.
    """
    contracted_data = {}
    for tup_data in working_data:
        split_label = tup_data[0].split('_')
        T_idx = split_label.index('T')
        # the [0:-2] removes the trailing units of ns
        T_val = f"{float(split_label[T_idx + 1][0:-2]):.4f}"

        if T_val in contracted_data:
            # append additional data to T_val
            contracted_data[T_val][1].append(copy.deepcopy(tup_data[2]))
        else:
            contracted_data[T_val] = (tup_data[1], [copy.deepcopy(tup_data[2])])

    return contracted_data

def contract_by_tau(working_data):
    """
    Given that self.working_data is populated by tuple data,
    contracts data into [contracted_data] dict by combining all
    data with shared tau into single data point, i.e.
    contracted_data[tau] = [{counts_dict1}, {counts_dict2}, ...]
    where each counts_dict is over same tau.
    """
    contracted_data = {}
    for tup_data in working_data:
        split_label = tup_data[0].split('_')
        tau_idx = split_label.index('tau')
        # the [0:-2] removes the trailing units of ns
        tau_val = f"{float(split_label[tau_idx + 1][0:-2]):.4f}"

        if tau_val in contracted_data:
            # check that number of qubits measured is the same
            assert(contracted_data[tau_val][0] == tup_data[1])
            # append additional data to tau_val
            contracted_data[tau_val][1].append(copy.deepcopy(tup_data[2]))
        else:
            contracted_data[tau_val] = (tup_data[1], [copy.deepcopy(tup_data[2])])

    return contracted_data

def contract_by_marker(working_data):
    """
    Given that self.working_data is populated by tuple data,
    contracts data into [contracted_data] dict by combining all
    data with shared marker into single data point, i.e.
    contracted_data[marker] = [{counts_dict1}, {counts_dict2}, ...]
    where each counts_dict is over same marker.
    """
    contracted_data = {}
    for tup_data in working_data:
        split_label = tup_data[0].split('_')
        mark_idx = split_label.index('grover')
        markers = split_label[mark_idx + 1].strip('[]')

        if markers in contracted_data:
            # check that number of qubits measured is the same
            assert(contracted_data[marker][0] == tup_data[1])
            # append additional data
            contracted_data[markers][1].append(copy.deepcopy(tup_data[2]))
        else:
            contracted_data[markers] = (tup_data[1], [copy.deepcopy(tup_data[2])])

    return contracted_data


def get_grover_succ_and_fail_num(dict_counts, num_qubits_measured, markers):
    if type(markers) is not list:
        markers = [markers]

    total_counts = 0
    success_num = 0
    for key, value in dict_counts.items():
        total_counts += value
        if str(key) in markers:
            success_num += value

    return success_num, (total_counts - success_num)


def get_success_failure_count(dict_counts, num_qubits_measured):
    # results have hex keys--convert them to binary keys first
    counts = hex_dict_to_bin_dict(dict_counts, num_qubits_measured)
    # obtain the number 0s from each shot of experiment and create a (degenerate) array
    # That is, if '001' shows up 50 times, populate array with 50 instances of 2
    num_0s_list = dict_data_to_n0_array(counts)
    num_0s = sum(num_0s_list)
    num_1s = (num_qubits_measured * len(num_0s_list) - num_0s)
    return num_0s, num_1s

##################################################
# Statistics (such as bootstrapping)
##################################################

def safe_beta(a,b,n):
    if a==0:
        return np.zeros(n)
    if b==0:
        return np.ones(n)
    if a!=0 and b!=0:
        return np.random.beta(a,b,size=n)

def beta_bayesian_bootstrap(successes,failures,n):
    # successes = list of the number of successes in each gauge
    # failures = list of the number of failures in each gauge
    # n is number of samples
    bb_weights=np.random.dirichlet(np.ones(len(successes)),n)
    out=np.zeros(n)
    for i in range(0,len(successes)):
        out+=bb_weights[:,i]*safe_beta(successes[i],failures[i],n)
    return out

def bootstrapci(data, n=1000, func=np.mean, ci=.95):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. Then computes error bar with
    'ci' confidence interval on data.
    """
    simulations = np.zeros(n)
    sample_size = len(data)
    xbar = func(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations[c] = func(itersample)
    diff = np.sort(simulations - xbar)
    lefterr_idx = int(np.floor(ci*n))
    righterr_idx = int(np.ceil((1 - ci)*n))

    conf_int = (xbar - diff[lefterr_idx], xbar - diff[righterr_idx])

    return conf_int

def per_err(mu, errbar):
    """
    Computes percent error with respect to mean. In particular, we calculate difference between
    (mu - errbar[0]) / mu and (mu - errbar[1]) / mu and takes the larger of the two in abs value then
    just multiply by 100.
    """
    diff1 = abs(mu - errbar[0])
    diff2 = abs(mu - errbar[1])

    perdif1 = (diff1 / mu) * 100
    perdif2 = (diff2 / mu) * 100

    if perdif1 > perdif2:
        return perdif1
    else:
        return perdif2

def calc_exper_fid(counts, num_qubits_measured):
    """
    Given an single experiment's results from IBMQ, calculates the fidelity and computes bootstrapped error
    bars. By "fidelity" we mean the total number of zeros--that is, we assume experiment is designed to output
    0 at the end. (If you encode state--decode it!)
    For example, if 3 qubit experiment with 10 shots gives results {'101': 3, '000': 5, '111': 2}, then the
    total number of zeros is 3(1) + 5(3) + 2(0) = 18, and fid = 18 / (10 * 3).

    Input
    ----------------
    exper_res -- "results" of IBMQ experiemnt run, i.e. if you run result = submit_job(...).to_dict(), then
    this expects results['results'][j], i.e. the results of jth experiment.
    num_qubits_measured -- int

    Output
    ----------------
    fid -- tuple, (fidelity, -.95ci, +.95ci, p_err) -(+).95ci is lower(upper) bound of 95% confidence
    interval via bootstrapping and err_mag is magnitude of error w.r.t. mean.

    Also prints if abs mag of error is less than input tol.
    """
    # results have hex keys--convert them to binary keys first
    counts = hex_dict_to_bin_dict(counts, num_qubits_measured)
    # obtain the number 0s from each shot of experiment and create a (degenerate) array
    # That is, if '001' shows up 50 times, populate array with 50 instances of 2
    num_0s = dict_data_to_n0_array(counts)
    # turn num0s to fideltiy by dividing by number of qubits
    fids = num_0s / num_qubits_measured
    # calculate mean
    mean_fid = np.mean(fids)
    # calculate confidence interval with 1000 bootstrap samples and .95 confidence interval
    ci = bootstrapci(fids, 1000, np.mean, .95)
    # calculate percent error
    p_err = per_err(mean_fid, ci)

    return (mean_fid, ci[0], ci[1], p_err)

def fix_measurement_mistake(count_list):
    """
    Take data where I accidently measured idle qubits
    and ignore them to create single qubit counts.
    """
    new_count_list = []
    for counts in count_list:
        new_counts = {}
        num_0 = 0
        num_1 = 0
        for key in counts:
            binary = hex_to_bin(key, 5)
            if binary[-1] == '0':
                num_0 += counts[key]
            else:
                num_1 += counts[key]
        new_counts['0x0'] = num_0
        new_counts['0x1'] = num_1
        new_count_list.append(new_counts)

    return new_count_list
