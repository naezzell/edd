import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import datetime
import os
import time
import pandas as pd
import scipy
import edd.experiments as edde
from edd.backend import IBMQBackend
from edd.data import IBMQData
from edd.pulse import IBMQDdSchedule

#############################################
# Post Experiment Analysis Utils
#############################################
def load_raw_data(file_list):
    """
    Given [file_list], loads all the .yml files
    into list of [data_sets].
    """
    data_sets = []
    for f in file_list:
        # get job index
        split_f = f.split('_')
        job_idx = split_f.index('job')
        job_n = int(split_f[job_idx + 1])
        data = IBMQData(name=f"job_{job_n}")
        data.load_raw_data(f)
        data.format_data()
        try:
            data.apply_error_mitigation()
        except:
            pass

        qubit = None
        error_mit = True
        j = 0
        while error_mit:
            data_label = data.tuple_data[j][0]
            if 'error_mitigate' in data_label:
                j += 1
                continue
            else:
                split_label = data_label.split('_')
                try:
                    ddqs_idx = split_label.index('ddqs')
                except:
                    ddqs_idx = split_label.index('ddq')
                qubit = int(split_label[ddqs_idx + 1].strip('[]'))
                error_mit = False

        txt_f = f"{f[:f.find('_raw_data')]}_properties.txt"
        with open(txt_f, 'r') as tf:
            lines = [l.strip('\n') for l in tf.readlines()]
            # get index of "qubit info"
            qi_idx = lines.index('Qubit Info')
            header = lines[qi_idx + 2].split(',')
            # get index of T1 and T2 times
            t1_idx = None
            t2_idx = None
            for i in range(len(header)):
                c_label = header[i]
                if 'T1(us)' in c_label:
                    t1_idx = i
                elif 'T2(us)' in c_label:
                    t2_idx = i
            # extract T1 and T2 time for relevant qubit
            line_idx = qi_idx + 2 + qubit + 1
            qubit_info = lines[line_idx].split(',')
            T1 = float(qubit_info[t1_idx])
            T2 = float(qubit_info[t2_idx])
            # get index of "pulse calibration"
            pc_idx = lines.index("Pulse calibration information")
            # get pulses used and current calibration
            line_used_x = lines[pc_idx + 2].split(': ')
            used_x = line_used_x[1]
            line_calib_x = lines[pc_idx + 3].split(': ')
            calib_x = line_calib_x[1]
            line_used_y = lines[pc_idx + 4].split(': ')
            used_y = line_used_y[1]
            line_calib_y = lines[pc_idx + 5].split(': ')
            calib_y = line_calib_y[1]
            # determine if calibration was same or not
            if used_x == calib_x and used_y == calib_y:
                good_calib = True
            else:
                good_calib = False

        # add T1, T2, and good_calib labels to data label
        for j in range(len(data.tuple_data)):
            tup_data = data.tuple_data[j]
            split_label = tup_data[0].split("_")
            try:
                try:
                    ddqs_idx = split_label.index('ddqs')
                except:
                    ddqs_idx = split_label.index('ddq')
            except:
                continue
            pre_ddqs = "_".join(split_label[0: ddqs_idx + 2])
            post_ddqs = "_".join(split_label[ddqs_idx + 2: ])
            mid_str = f"_T1_{T1}_T2_{T2}_goodc_{good_calib}_"
            new_str = pre_ddqs + mid_str + post_ddqs
            data.tuple_data[j] = (new_str, tup_data[1], tup_data[2])

        data_sets.append(data)

    return data_sets

def load_fixed_mistake_raw_data(file_list):
    """
    Given [file_list], loads all the .yml files
    into list of [data_sets].
    """
    data_sets = []
    for f in file_list:
        # get job index
        split_f = f.split('_')
        job_idx = split_f.index('job')
        job_n = int(split_f[job_idx + 1])
        data = IBMQData(name=f"job_{job_n}")
        data.load_raw_data(f)
        data.format_data()
        data_sets.append(data)

    return data_sets

def perform_pauli_pode_analysis(data_sets, list_of_seq, offset=0,
                               gen_plots = True, cwd="."):
    """
    Given [data_sets] and [list_of_seqs], performs
    the main rel_adv_to_XY4 pauli pode analysis.
    """
    # make data container
    rel_adv_data = {}

    # make directory for each seq if not already made
    for seq in list_of_seq:
        try:
            os.mkdir(f"{cwd}/{seq}")
        except:
            continue

    # perform the analysis
    count = 0
    for data in data_sets:
        # do some complicated stuff to extract non-XY4 sequence
        enter_else_lab = False
        # extract the sequence name from a representative tuple data
        rep_tup = data.tuple_data[0]
        # extract seq and offset
        for tup in data.tuple_data:
            label = tup[0]
            if 'offset' in label:
                if 'xy4' in label and 'c_basis' in label:
                    continue
                else:
                    enter_else_lab = True
                    split_label = label.split('_')
                    offset_idx = split_label.index('offset') + 1
                    offset = float(split_label[offset_idx])
                    seq_idx = split_label.index('decay') + 1
                    seq = split_label[seq_idx]
                    sym_idx = split_label.index('sym') + 1
                    sym = split_label[sym_idx]
                    if seq == 'super':
                        seq += f"-{split_label[seq_idx+1]}"
                    elif seq == 'qdd':
                        seq += f"-{split_label[seq_idx+1]}-{split_label[seq_idx+2]}"
                    break

        if enter_else_lab is False:
            continue
        # once seq extracted, we can actually do analysis
        else:
            # make fname for data to save
            fname = "rel_adv_to_xy4_seq_{}_sym_{}_pode_{}_offset_{}_count_{}"
            # perform analysis state by state
            state_data = {}
            for pode in range(6):
                # analyze non-XY4 sequence performance
                pode_lab = f"pode_{pode}"
                data.change_working_data_by_label(seq, pode_lab)
                seq_pdata = data.wrangle_fid_decay()
                data.save_plot_data(f"{cwd}/{seq}/{fname.format(seq, sym, pode, offset, count)}")
                seq_fit = bootstrap_pub_fit(seq_pdata)
                seq_lam = seq_fit[0][0]
                seq_err = seq_fit[0][1]
                seq_gam = seq_fit[2][0]
                seq_gam_err = seq_fit[2][1]

                # now analyze XY4 sequence performance
                data.change_working_data_by_label('xy4', pode_lab)
                xy4_pdata = data.wrangle_fid_decay()
                data.save_plot_data(f"{cwd}/{seq}/{fname.format('xy4', sym, pode, offset, count)}")
                xy4_fit = bootstrap_pub_fit(xy4_pdata)
                xy4_lam = xy4_fit[0][0]
                xy4_err = xy4_fit[0][1]
                xy4_gam = xy4_fit[2][0]
                xy4_gam_err = xy4_fit[2][1]

                # now get relative performance to XY4
                rel_adv = xy4_lam / seq_lam
                rel_adv_err = np.sqrt((xy4_err / xy4_lam)**2 + (seq_err / seq_lam)**2) * rel_adv
                # append ratio to list for that state
                state_data[pode] = [rel_adv, rel_adv_err, seq_gam, seq_gam_err]

                if gen_plots is True:
                    title = fname.format(seq, sym, pode, offset, count)
                    fig, ax = plot_pub_decay([seq_pdata, xy4_pdata], True, title=title.replace("_", "-"), seqs=[seq.replace("_", "-"), 'xy4'])
                    fig.savefig(f"{cwd}/{seq}/{title}.png")

            # append state_data to total data set for this seq
            if seq not in rel_adv_data:
                rel_adv_data[seq] = [state_data]
            else:
                rel_adv_data[seq].append(state_data)

            count += 1

    return rel_adv_data

def extract_and_save_plot_datas(data_sets, data_dir, cwd='.'):
    """
    Given [data_sets] and [list_of_seqs], generates
    plot data from raw data.
    """
    # keep track of file names
    fnames = []

    # make directory to store data unless already exists
    try:
        os.mkdir(f"{cwd}/{data_dir}")
    except:
        print("Directory already exists")

    # generate plot data files
    for count, data in enumerate(data_sets):
        # do some complicated stuff to extract non-XY4 sequence
        enter_else_lab = False
        # extract the sequence name from a representative tuple data
        rep_tup = data.tuple_data[0]
        # extract seq and offset
        for tup in data.tuple_data:
            label = tup[0]
            if 'offset' in label:
                enter_else_lab = True
                split_label = label.split('_')
                backname_idx = split_label.index("backName") + 1
                back_name = split_label[backname_idx]
                basis_idx = split_label.index("pulse") + 1
                basis = split_label[basis_idx]
                offset_idx = split_label.index('offset') + 1
                offset = float(split_label[offset_idx])
                dtype_idx = split_label.index("dtype") + 1
                delay_type = split_label[dtype_idx]
                seq_idx = split_label.index('decay') + 1
                seq = split_label[seq_idx]
                sym_idx = split_label.index('sym') + 1
                sym = split_label[sym_idx]
                try:
                    qubit_idx = split_label.index('ddqs')
                except:
                    qubit_idx = split_label.index('ddq')
                qubit = split_label[qubit_idx][1:][0:-1]
                goodc_idx = split_label.index('goodc') + 1
                goodc = split_label[goodc_idx]
                t1_idx = split_label.index("T1") + 1
                T1 = float(split_label[t1_idx])
                t2_idx = split_label.index("T2") + 1
                T2 = float(split_label[t2_idx])
                date_idx = split_label.index("date") + 1
                date = split_label[date_idx]
                if seq == 'super':
                    seq += f"-{split_label[seq_idx+1]}"
                elif seq == 'qdd':
                    seq += f"-{split_label[seq_idx+1]}-{split_label[seq_idx+2]}"
                if 'c_basis' in label and "xy4" in label:
                    seq = "ref-xy4"
                break

        if enter_else_lab is False:
            continue
        # once seq extracted, we can actually extract data
        else:
            # make fname for data to save
            fname = f"fidDecay_backend_{back_name}_qubit_{qubit}_"
            fname += f"basis_{basis}_goodc_{goodc}_T1_{T1:0.2f}_T2_{T2:0.2f}_"
            fname += f"dtype_{delay_type}_offset_{offset}_"
            fname += "seq_{}_sym_{}_pode_{}_"
            fname += f"date_{date}"
            # perform analysis state by state
            state_data = {}
            for pode in range(6):
                # analyze non-XY4 sequence performance
                pode_lab = f"pode_{pode}"
                if seq == "ref-xy4":
                    data.change_working_data_by_label("xy4".replace("-", "_"), pode_lab)
                else:
                    data.change_working_data_by_label(seq.replace("-", "_"), pode_lab)
                #print(data.working_data[0])
                seq_pdata = data.wrangle_fid_decay()
                sfname = f"{cwd}/{data_dir}/{fname.format(seq, sym, pode)}"
                fnames.append(sfname)
                data.save_plot_data(sfname)

    return fnames


def extract_and_save_xy4_comp_plot_datas(data_sets, data_dir, cwd='.'):
    """
    Given [data_sets] and [list_of_seqs], generates
    plot data from raw data.
    """
    # keep track of file names
    fnames = []

    # make directory to store data unless already exists
    try:
        os.mkdir(f"{cwd}/{data_dir}")
    except:
        print("Directory already exists")

    # generate plot data files
    for count, data in enumerate(data_sets):
        # do some complicated stuff to extract non-XY4 sequence
        enter_else_lab = False
        # extract the sequence name from a representative tuple data
        rep_tup = data.tuple_data[0]
        # extract seq and offset
        for tup in data.tuple_data:
            label = tup[0]
            if 'offset' in label:
                if 'xy4' in label and 'c_basis' in label:
                    continue
                else:
                    enter_else_lab = True
                    split_label = label.split('_')
                    backname_idx = split_label.index("backName") + 1
                    back_name = split_label[backname_idx]
                    offset_idx = split_label.index('offset') + 1
                    offset = float(split_label[offset_idx])
                    dtype_idx = split_label.index("dtype") + 1
                    delay_type = split_label[dtype_idx]
                    seq_idx = split_label.index('decay') + 1
                    seq = split_label[seq_idx]
                    sym_idx = split_label.index('sym') + 1
                    sym = split_label[sym_idx]
                    qubit_idx = split_label.index("ddqs") + 1
                    qubit = split_label[qubit_idx][1:][0:-1]
                    t1_idx = split_label.index("T1") + 1
                    T1 = float(split_label[t1_idx])
                    t2_idx = split_label.index("T2") + 1
                    T2 = float(split_label[t2_idx])
                    date_idx = split_label.index("date") + 1
                    date = split_label[date_idx]
                    if seq == 'super':
                        seq += f"-{split_label[seq_idx+1]}"
                    elif seq == 'qdd':
                        seq += f"-{split_label[seq_idx+1]}-{split_label[seq_idx+2]}"
                    break

        if enter_else_lab is False:
            continue
        # once seq extracted, we can actually extract data
        else:
            # make fname for data to save
            fname = f"rel_adv_to_xy4_backend_{back_name}_qubit_{qubit}_"
            fname += f"T1_{T1:0.2f}_T2_{T2:0.2f}_"
            fname += f"dtype_{delay_type}_offset_{offset}_"
            fname += "seq_{}_sym_{}_pode_{}_"
            fname += f"date_{date}"
            # perform analysis state by state
            state_data = {}
            for pode in range(6):
                # analyze non-XY4 sequence performance
                pode_lab = f"pode_{pode}"
                data.change_working_data_by_label(seq.replace("-", "_"), pode_lab)
                #print(data.working_data[0])
                seq_pdata = data.wrangle_fid_decay()
                sfname = f"{cwd}/{data_dir}/{fname.format(seq, sym, pode)}"
                fnames.append(sfname)
                data.save_plot_data(sfname)

                # now analyze XY4 sequence performance
                data.change_working_data_by_label('xy4', 'c_basis', pode_lab)
                #print(data.working_data[0])
                xy4_pdata = data.wrangle_fid_decay()
                sfname = f"{cwd}/{data_dir}/{fname.format('ref-xy4', False, pode)}"
                fnames.append(sfname)
                data.save_plot_data(sfname)

    return fnames


def extract_and_save_fixed_mistake_plot_datas(data_sets, list_of_seq,
                                              data_dir,offset=0,
                                              cwd='.'):
    """
    Given [data_sets] and [list_of_seqs], generates
    plot data from raw data.
    """
    # keep track of file names
    fnames = []

    # make individual directories for each sequence
    for seq in list_of_seq:
        try:
            os.mkdir(f"{cwd}/{data_dir}/{seq}")
        except:
            continue

    # generate plot data files
    for count, data in enumerate(data_sets):
        # do some complicated stuff to extract non-XY4 sequence
        enter_else_lab = False
        # extract the sequence name from a representative tuple data
        rep_tup = data.tuple_data[0]
        # extract seq and offset
        for tup in data.tuple_data:
            label = tup[0]
            if 'offset' in label:
                if 'xy4' in label and 'c_basis' in label:
                    continue
                else:
                    enter_else_lab = True
                    split_label = label.split('_')
                    offset_idx = split_label.index('offset') + 1
                    offset = float(split_label[offset_idx])
                    seq_idx = split_label.index('decay') + 1
                    seq = split_label[seq_idx]
                    sym_idx = split_label.index('sym') + 1
                    sym = split_label[sym_idx]
                    if seq == 'super':
                        seq += f"_{split_label[seq_idx+1]}"
                    elif seq == 'qdd':
                        seq += f"_{split_label[seq_idx+1]}_{split_label[seq_idx+2]}"
                    break

        if enter_else_lab is False:
            continue
        # once seq extracted, we can actually extract data
        else:
            # make fname for data to save
            fname = "rel_adv_to_xy4_seq_{}_sym_{}_pode_{}_offset_{}_count_{}"
            # perform analysis state by state
            state_data = {}
            for pode in range(6):
                # analyze non-XY4 sequence performance
                pode_lab = f"pode_{pode}"
                data.change_working_data_by_label(seq, pode_lab)
                #print(data.working_data[0])
                seq_pdata = data.wrangle_mistake_fid_decay()
                sfname = f"{cwd}/{data_dir}/{seq}/{fname.format(seq, sym, pode, offset, count)}"
                fnames.append(sfname)
                data.save_plot_data(sfname)

                # now analyze XY4 sequence performance
                data.change_working_data_by_label('xy4', 'c_basis', pode_lab)
                #print(data.working_data[0])
                xy4_pdata = data.wrangle_mistake_fid_decay()
                sfname = f"{cwd}/{data_dir}/{seq}/{fname.format('xy4', False, pode, offset, count)}"
                fnames.append(sfname)
                data.save_plot_data(sfname)

    return fnames


def perform_pauli_pode_seq1_to_seq2_analysis(data_sets, seq1, seq2,
                                             offset=0, gen_plots = True,
                                             cwd="."):
    """
    Given [data_sets] and [list_of_seqs], performs
    the main rel_adv_to_XY4 pauli pode analysis.
    """
    # make data container
    rel_adv_data = []

    fprefix = f"{cwd}/{seq1}_to_{seq2}"

    # make directory for each seq if not already made
    try:
        os.mkdir(fprefix)
    except:
        pass

    # perform the analysis
    count = 0
    for data in data_sets:
        # make fname for data to save
        fname = f"rel_adv_{seq1}_to_{seq2}"
        fname += "_pode_{}_offset_{}_count_{}"
        # perform analysis state by state
        state_data = {}
        for pode in range(6):
            # analyze non-XY4 sequence performance
            pode_lab = f"pode_{pode}"
            data.change_working_data_by_label(seq1, pode_lab)
            seq1_pdata = data.wrangle_fid_decay()
            data.save_plot_data(f"{fprefix}/{fname.format(pode, offset, count)}")
            seq1_fit = data.bootstrap_fit_aug_cosexp()
            seq1_lam = seq1_fit[0][0]
            seq1_err = seq1_fit[0][1]
            seq1_gam = seq1_fit[2][0]
            seq1_gam_err = seq1_fit[2][1]

            # now analyze XY4 sequence performance
            data.change_working_data_by_label(seq2, pode_lab)
            seq2_pdata = data.wrangle_fid_decay()
            data.save_plot_data(f"{fprefix}/{fname.format(pode, offset, count)}")
            seq2_fit = data.bootstrap_fit_aug_cosexp()
            seq2_lam = seq2_fit[0][0]
            seq2_err = seq2_fit[0][1]
            seq2_gam = seq2_fit[2][0]
            seq2_gam_err = seq2_fit[2][1]

            # now get relative performance to XY4
            rel_adv = seq2_lam / seq1_lam
            rel_adv_err = np.sqrt((seq1_err / seq1_lam)**2 + (seq2_err / seq2_lam)**2) * rel_adv
            # append ratio to list for that state
            state_data[pode] = [rel_adv, rel_adv_err, seq1_gam, seq1_gam_err]

            if gen_plots is True:
                title = fname.format(pode, offset, count)
                fig, ax = plot_cosfid_decay([seq1_pdata, seq2_pdata], True, title=title.replace("_", "-"),
                                 seqs=[seq1.replace("_", "-"), seq2.replace("_", "-")])
                fig.savefig(f"{fprefix}/{title}.png")

        rel_adv_data.append(state_data)

        count += 1

    return rel_adv_data

def perform_pode_max_seq_vs_min_xy4_analysis(data_sets, list_of_seq, offset=0,
                                             gen_plots = True, cwd="."):
    """
    Given [data_sets] and [list_of_seqs], performs
    the main rel_adv_to_XY4 pauli pode analysis.
    """
    # make data container
    rel_adv_data = {}

    # make directory for each seq if not already made
    for seq in list_of_seq:
        try:
            os.mkdir(f"{cwd}/{seq}")
        except:
            continue

    # perform the analysis
    count = 0
    for data in data_sets:
        # do some complicated stuff to extract non-XY4 sequence
        enter_else_lab = False
        # extract the sequence name from a representative tuple data
        rep_tup = data.tuple_data[0]
        # extract seq and offset
        for tup in data.tuple_data:
            label = tup[0]
            if 'offset' in label:
                if 'xy4' in label and 'tau_0dt' in label:
                    continue
                else:
                    enter_else_lab = True
                    split_label = label.split('_')
                    offset_idx = split_label.index('offset') + 1
                    offset = float(split_label[offset_idx])
                    seq_idx = split_label.index('decay') + 1
                    seq = split_label[seq_idx]
                    if seq == 'super':
                        seq += f"_{split_label[seq_idx+1]}"
                    elif seq == 'qdd':
                        seq += f"_{split_label[seq_idx+1]}_{split_label[seq_idx+2]}"
                    break

        if enter_else_lab is False:
            continue
        # once seq extracted, we can actually do analysis
        else:
            # make fname for data to save
            fname = "rel_adv_to_min_xy4_max_seq_{}_pode_{}_offset_{}_count_{}"
            # perform analysis state by state
            state_data = {}
            for pode in range(6):
                # analyze non-XY4 sequence performance
                pode_lab = f"pode_{pode}"
                data.change_working_data_by_label(seq, 'reps_1', pode_lab)
                seq_pdata = data.wrangle_fid_decay()
                data.save_plot_data(f"{cwd}/LD-xy4/{fname.format(seq, pode, offset, count)}")
                seq_fit = data.bootstrap_fit_aug_cosexp()
                seq_tau = seq_fit[0][0]
                seq_err = seq_fit[0][1]
                seq_gam = seq_fit[2][0]
                seq_gam_err = seq_fit[2][1]

                # now analyze XY4 sequence performance
                data.change_working_data_by_label('xy4', 'tau_0dt', pode_lab)
                xy4_pdata = data.wrangle_fid_decay()
                data.save_plot_data(f"{cwd}/{seq}/{fname.format('xy4', pode, offset, count)}")
                xy4_fit = data.bootstrap_fit_aug_cosexp()
                xy4_tau = xy4_fit[0][0]
                xy4_err = xy4_fit[0][1]
                xy4_gam = xy4_fit[2][0]
                xy4_gam_err = xy4_fit[2][1]

                # now get relative performance to XY4
                rel_adv = seq_tau / xy4_tau
                rel_adv_err = np.sqrt((xy4_err / xy4_tau)**2 + (seq_err / seq_tau)**2) * rel_adv
                # append ratio to list for that state
                state_data[pode] = [rel_adv, rel_adv_err, seq_gam, seq_gam_err]

                if gen_plots is True:
                    title = fname.format(seq, pode, offset, count)
                    fig, ax = plot_cosfid_decay([seq_pdata, xy4_pdata], True, title=title.replace("_", "-"),
                                     seqs=[f"LD {seq.replace('_', '-')}", 'SD xy4'])
                    fig.savefig(f"{cwd}/{seq}/{title}.png")

            # append state_data to total data set for this seq
            if seq not in rel_adv_data:
                rel_adv_data[seq] = [state_data]
            else:
                rel_adv_data[seq].append(state_data)

            count += 1

    return rel_adv_data


def calc_med_with_err(values, errs):
    """
    Given [values] +/- [errs], finds the median
    and propogates the error if necessary.
    """
    # handle odd case first
    if len(values) % 2 == 1:
        med_idx = np.argsort(values)[len(values)//2]
        med = values[med_idx]
        med_err = errs[med_idx]
    else:
        sort_idxs = np.argsort(values)
        idx_of_med_idx = len(sort_idxs) // 2
        med_idx1 = sort_idxs[idx_of_med_idx]
        med_idx0 = sort_idxs[idx_of_med_idx - 1]
        med = (values[med_idx0] + values[med_idx1]) / 2
        weight = np.sqrt((errs[med_idx0]/values[med_idx0])**2 + (errs[med_idx1]/values[med_idx1])**2)
        med_err = med * weight

    return med, med_err

def make_rel_adv_summary(rel_adv_data, fname, pd=''):
    """
    Takes [rel_adv_data] and summarizes results into
    [fname].txt.
    """
    # write header
    with open (fname, 'w') as f:
        header = "seq, "
        for j in range(6):
            header += f"p{j}_ra, p{j}_ra_2sigma, p{j}_gam, p{j}_gam_2sig "
        header += "min, 2sig_min, max, 2sig_max, med, 2sig_med, mean, 2sig_mean\n"
        f.write(header)

    # write lines for each sequence
    for seq in rel_adv_data:
        # set up ratio list of each pode
        ratios = [[] for _ in range(6)]
        errors = [[] for _ in range(6)]
        gams = [[] for _ in range(6)]
        gam_errs = [[] for _ in range(6)]
        data = rel_adv_data[seq]
        # iterate over trials
        for trial in data:
            # iterate over podes
            for j in range(6):
                ratios[j].append(trial[j][0])
                errors[j].append(trial[j][1])
                gams[j].append(np.abs(trial[j][2]))
                gam_errs[j].append(trial[j][3])


        # set up list of means/ errs
        means = []
        tot_errs = []
        gam_means = []
        gam_tot_errs = []
        for j in range(6):
            # get mean/ error over trials for fidelity
            # calculate mean
            m = np.mean(ratios[j])
            means.append(m)
            # calculate error
            std = 2*np.std(ratios[j])
            prop_uncert = m * np.sqrt(np.sum((np.array(errors[j]) / np.array(ratios[j]))**2))
            tot_e = np.sqrt(std**2 + prop_uncert**2)
            tot_errs.append(prop_uncert)
            # get mean/ error over trials for gamma
            g_m = np.mean(gams[j])
            gam_means.append(g_m)
            # calculate error
            std = 2*np.std(gams[j])
            prop_uncert = g_m * np.sqrt(np.sum((np.array(gam_errs[j]) / np.array(gams[j]))**2))
            tot_e = np.sqrt(std**2 + prop_uncert**2)
            gam_tot_errs.append(tot_e)

        # calculate order statistics
        # min
        min_idx = np.argmin(means)
        minn = means[min_idx]
        minn_err = tot_errs[min_idx]
        # max
        max_idx = np.argmax(means)
        maxx = means[max_idx]
        maxx_err = tot_errs[max_idx]
        # median
        med, med_err = calc_med_with_err(means, tot_errs)
        # mean
        mean = np.mean(means)
        weight = 0.0
        for j in range(6):
            weight += (tot_errs[j] / means[j])**2
        mean_err = mean * np.sqrt(weight)

        with open(fname, 'a') as f:
            line = f"{pd}{seq},"
            for j in range(6):
                line += f"{means[j]:0.4f},{tot_errs[j]:0.4f},{gam_means[j]:0.4f},{gam_tot_errs[j]:0.4f},"
            line += f"{minn},{minn_err},"
            line += f"{maxx},{maxx_err},"
            line += f"{med},{med_err},"
            line += f"{mean},{mean_err}\n"
            f.write(line)

    return


def load_rel_adv_summary(fname):
    """
    Loads .txt file summary of rel advantage experiment
    into a list of lists.
    """
    seq_list = []
    p0_data = []
    p1_data = []
    p2_data = []
    p3_data = []
    p4_data = []
    p5_data = []
    med_data = []
    with open(fname, 'r') as f:
        # skip header
        f.readline()
        # get data for each sequence
        for line in f.readlines():
            data = line.rstrip().split(',')
            # add seq
            seq_list.append(data[0].replace('_', '-'))
            # add p0 data
            p0_data.append([float(x) for x in data[1:5]])
            p1_data.append([float(x) for x in data[5:9]])
            p2_data.append([float(x) for x in data[9:13]])
            p3_data.append([float(x) for x in data[13:17]])
            p4_data.append([float(x) for x in data[17:21]])
            p5_data.append([float(x) for x in data[21:25]])
            med_data.append([float(data[-4]), float(data[-3])])

    all_data = [seq_list, p0_data, p1_data, p2_data, p3_data, p4_data, p5_data, med_data]
    return all_data


def combine_rel_adv_data(*data_sets):
    """
    Combines relative advantage [data_sets]
    into one large data set.
    """
    big_dset = [[] for _ in range(8)]
    for dset in data_sets:
        for j in range(8):
            big_dset[j].extend(dset[j])

    return big_dset

def argsort_seq(o_seq, u_seq):
    """
    Given an ordered set [o_seq] of strings,
    find indices that put unordered set [u_seq]
    into same order.
    """
    indices = []
    for seq in o_seq:
        idx = u_seq.index(seq)
        indices.append(idx)

    return np.array(indices)

def get_data_subset(data, seqs):
    """
    Extracts only data of [seqs].
    """
    sub_data = [[] for _ in range(len(data))]

    for j in range(len(data[0])):
        if data[0][j] in seqs:
            sub_data[0].append(data[0][j])
            for k in range(1, len(data)):
                sub_data[k].append(data[k][j])

    ord_data = [[] for _ in range(len(sub_data))]
    # now re-order in accordance with seqs order
    idx_order = argsort_seq(seqs, sub_data[0])
    for j in range(len(sub_data[0])):
        for k in range(len(sub_data)):
            ord_data[k].append(sub_data[k][idx_order[j]])

    return ord_data

def sep_data_with_osc(seqs, pj_data):
    """
    Separates data into set of those with negligible oscillations
    and those with apparent oscillations.
    """
    x = []
    xg = []
    y = []
    y_err = []
    yg = []
    yg_err = []
    for idx, vals in enumerate(pj_data):
        if np.abs(vals[2]) > 1000:
            if np.abs(vals[0]) > 10:
                x.append(seqs[idx])
                y.append(vals[0])
                y_err.append(0)
            else:
                x.append(seqs[idx])
                y.append(vals[0])
                y_err.append(vals[1])
        else:
            xg.append(seqs[idx])
            yg.append(vals[0])
            yg_err.append(vals[1])

    return x, y, y_err, xg, yg, yg_err

def sep_med_data(med_data):
    """
    Separates median data from tuple list of the form
    [(med1, med_err1), ...] into [[med1, med2...], [err1, err2...]]
    """
    y = []
    y_err = []
    for tup in med_data:
        y.append(tup[0])
        y_err.append(tup[1])

    return y, y_err

def get_xticks_from_seq(seq, sub_seq):
    """
    Create x-tick (i.e. plot x values) list
    gives a sub_seq of seq that respects original
    ordering/ spacing.
    """
    x_ticks = []
    for j in range(len(seq)):
        if seq[j] in sub_seq:
            x_ticks.append(j)

    return np.array(x_ticks)



#############################################
# Plotting Utilties
#############################################
def plot_podal_ra_results(data, figsize=1200, xmax=4):
    """
    Plots average improvement over XY4 for each sequence
    over the 6 Pauli states + the median performance.
    """
    fig, ax = plt.subplots(figsize=set_size(figsize))
    #fig.set_size_inches(18.5, 10.5)
    prop = ax._get_lines.prop_cycler
    # first, "decompress" the data for convenience
    seqs, p0, p1, p2, p3, p4, p5, avg = data
    xvals = np.array(list(range(len(seqs))))
    datas = [p0, p1, p2, p3, p4, p5, avg]
    for j, pj in enumerate(datas):
        offset = 0.1*(j - (len(datas)/2))
        # create label
        if j == 0:
            label = '0'
        elif j == 1:
            label = '1'
        elif j == 2:
            label = '+'
        elif j == 3:
            label = '-'
        elif j == 4:
            label = '+i'
        elif j == 5:
            label = '-i'
        else:
            label = 'median'
        c = next(prop)['color']
        if j < 6:
            x, y, y_err, xg, yg, yg_err = sep_data_with_osc(seqs, pj)
            x_ticks = get_xticks_from_seq(seqs, x)
            xg_ticks = get_xticks_from_seq(seqs, xg)
            ax.errorbar(x_ticks+offset, y, yerr=y_err, fmt='.', marker='o', color=c, label=label)
            if xg != []:
                ax.errorbar(xg_ticks+offset, yg, yerr=yg_err, fmt='.', marker='x', color=c)
        else:
            y, y_err = sep_med_data(pj)
            ax.errorbar(xvals+offset, y, yerr=y_err, fmt='.', marker='v', color=c, label=label)

    ax.legend()

    ax.set_xticks(xvals)
    for j in range(len(seqs)):
        if 'super' in seqs[j]:
            seqs[j] = seqs[j].replace('super', 's')

    ax.set_xticklabels(seqs)
    ax.set_ylabel("Relative advantage to XY4")
    ax.set_ylim(0, xmax)
    c = next(prop)['color']
    ax.axhline(y=1, color=c, linestyle='-')

    return fig, ax

def plot_podal_max_tau_ra_results(data, figsize=1200, xmax=4):
    """
    Plots average improvement over XY4 for each sequence
    over the 6 Pauli states + the median performance.
    Here, sequences have maximum \tau and are repeated only
    once, but XY4 has min \tau and is repeated many times.
    """
    fig, ax = plt.subplots(figsize=set_size(figsize))
    #fig.set_size_inches(18.5, 10.5)
    prop = ax._get_lines.prop_cycler
    # first, "decompress" the data for convenience
    seqs, p0, p1, p2, p3, p4, p5, avg = data
    xvals = np.array(list(range(len(seqs))))
    datas = [p0, p1, p2, p3, p4, p5, avg]
    for j, pj in enumerate(datas):
        offset = 0.1*(j - (len(datas)/2))
        # create label
        if j == 0:
            label = '0'
        elif j == 1:
            label = '1'
        elif j == 2:
            label = '+'
        elif j == 3:
            label = '-'
        elif j == 4:
            label = '+i'
        elif j == 5:
            label = '-i'
        else:
            label = 'median'
        c = next(prop)['color']
        if j < 6:
            x, y, y_err, xg, yg, yg_err = sep_data_with_osc(seqs, pj)
            x_ticks = get_xticks_from_seq(seqs, x)
            xg_ticks = get_xticks_from_seq(seqs, xg)
            ax.errorbar(x_ticks+offset, y, yerr=y_err, fmt='.', marker='o', color=c, label=label)
            if xg != []:
                ax.errorbar(xg_ticks+offset, yg, yerr=yg_err, fmt='.', marker='x', color=c)
        else:
            y, y_err = sep_med_data(pj)
            ax.errorbar(xvals+offset, y, yerr=y_err, fmt='.', marker='v', color=c, label=label)

    ax.legend()

    ax.set_xticks(xvals)
    for j in range(len(seqs)):
        if 'super' in seqs[j]:
            seqs[j] = seqs[j].replace('super', 's')
        seqs[j] = f"LD {seqs[j]}"

    print(seqs)

    ax.set_xticklabels(seqs)
    ax.set_ylabel("Relative advantage to XY4")
    ax.set_ylim(0, xmax)
    c = next(prop)['color']
    ax.axhline(y=1, color=c, linestyle='-')

    return fig, ax

def gen_pub_conf_curve(a0_fit, lam_fit, gam_fit, smooth_t):
    """
    Given fitting parameters and times of interest,
    generates a 2 \sigma confidence band.
    """
    # extract fitting params
    a0, a0_err = a0_fit
    lam, lam_err = lam_fit
    gam, gam_err = gam_fit

    # set up data containers
    func_vals = [[] for _ in range(len(smooth_t))]

    for _ in range(1000):
        s_a0 = np.random.normal(a0, a0_err / 2)
        s_lam = np.random.normal(lam, lam_err / 2)
        s_gam = np.random.normal(gam, gam_err / 2)

        s_func_vals = pub_func([s_a0, s_lam, s_gam], smooth_t)
def gen_pub_conf_curve(a0_fit, lam_fit, gam_fit, smooth_t):
    """
    Given fitting parameters and times of interest,
    generates a 2 \sigma confidence band.
    """
    # extract fitting params
    a0, a0_err = a0_fit
    lam, lam_err = lam_fit
    gam, gam_err = gam_fit

    # set up data containers
    func_vals = [[] for _ in range(len(smooth_t))]

    for _ in range(1000):
        s_a0 = np.random.normal(a0, a0_err / 2)
        s_lam = np.random.normal(lam, lam_err / 2)
        s_gam = np.random.normal(gam, gam_err / 2)

        s_func_vals = pub_func([s_a0, s_lam, s_gam], smooth_t)

        for j in range(len(s_func_vals)):
            func_vals[j].append(s_func_vals[j])

    st_devs = map(np.std, func_vals)
    st_devs = np.array([2 * sig for sig in st_devs])

    # generate the min and max values of func
    avg_vals = pub_func([a0, lam, gam], smooth_t)
    min_vals = avg_vals - st_devs
    max_vals = avg_vals + st_devs

    return min_vals, max_vals

def plot_pub_decay(plot_datas, with_fit=False, title='', seqs=None, size=400):
    """
    Plots [self.plot_data] assuming it was produced
    from fidelity decay experiment.
    """
    fig, ax = plt.subplots(figsize=set_size(size))
    prop = ax._get_lines.prop_cycler
    mark_gen = itertools.cycle(('o', 's', 'v'))
    if seqs is None:
        seqs = ['' for _ in range(len(plot_datas))]
    s_idx = 0
    legend = []
    for p_data in plot_datas:
        times, fids, errs = p_data
        time_ord = np.argsort(times)
        times = times[time_ord]
        fids = fids[time_ord]
        errs = errs[time_ord]
        color = next(prop)['color']
        marker = next(mark_gen)
        # simple plot
        if with_fit is False:
            ax.errorbar(times, fids, yerr=errs, color=color, fmt=f"{marker}-")
            ax.set_xlabel('time ($\\mu$s)')
            ax.set_ylabel('fidelity')
            ax.set_title(title)
            legend.append(seqs[s_idx])
            # plot with simple_exp fitting
        else:
            # get fit and plot it
            a0_fit, lam_fit, gam_fit, _ = bootstrap_pub_fit(p_data)
            a0, a0_err = a0_fit
            lam, lam_err = lam_fit
            gam, gam_err = gam_fit
            ax.errorbar(times, fids, yerr=errs, color=color, fmt=f"{marker}")
            smooth_t = np.linspace(min(times), max(times), 100)
            ax.plot(smooth_t, pub_func([a0, lam, gam], smooth_t), color=color)
            # format the legend with the decay constant and 2 std value
            st_lam = f'{(1000 * lam):.2f}'
            st_lam_err = f'{(1000*lam_err):.2f}'
            if np.abs(gam) < 1/1000:
                st_gam = "0"
                legend.append(f'{seqs[s_idx]}, $\\lambda$ = {st_lam} $\pm$ {st_lam_err} (MHz), $\\gamma$ = {st_gam} (MHz)')
            else:
                st_gam = f'{(1000*gam):.2f}'
                st_gam_err = f'{(1000*gam_err):.2f}'
                legend.append(f'{seqs[s_idx]}, $\\lambda$ = {st_lam} $\pm$ {st_lam_err} (MHz), $\\gamma$ = {st_gam} $\pm$ {st_gam_err} (MHz)')
            # add the 95% confidence intervals
            min_vals, max_vals = gen_pub_conf_curve(a0_fit, lam_fit, gam_fit, smooth_t)
            ax.fill_between(smooth_t, min_vals, max_vals, color=color, alpha=.2)
        # add labels
        ax.set_xlabel('time ($\mu$s)')
        ax.set_ylabel('fidelity')
        ax.set_title(title)
        s_idx += 1
    ax.legend(legend)

    return (fig, ax)

def plot_cosfid_decay(plot_datas, with_fit=False, title='', seqs=None):
    """
    Plots [self.plot_data] assuming it was produced
    from fidelity decay experiment.
    """
    fig, ax = plt.subplots(figsize=set_size(400))
    prop = ax._get_lines.prop_cycler
    if seqs is None:
        seqs = ['' for _ in range(len(plot_datas))]
    s_idx = 0
    legend = []
    for p_data in plot_datas:
        times, fids, errs = p_data
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
            fit_params, par_params, gam_params, sum_res = bootstrap_fit_aug_cosexp(p_data)
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
            legend.append(f'{seqs[s_idx]}, $\\lambda$ = {st_tau} $\pm$ {st_tau_err}, $\\gamma$ = {st_gam} $\pm$ {st_gam_err}')
            # add the 95% confidence intervals
            small = aug_cosexp_func([a0+a0_err, tau+tau_err, gam], ord_t)
            big = aug_cosexp_func([a0-a0_err, tau-tau_err, gam], ord_t)
            ax.fill_between(ord_t, aug_cosexp_func([a0+a0_err, tau+tau_err, gam], ord_t), aug_cosexp_func([a0-a0_err, tau-tau_err, gam], ord_t),
                            color=color, alpha=.2)
        # add labels
        ax.set_xlabel('time ($\mu$s)')
        ax.set_ylabel('fidelity')
        ax.set_title(title)
        s_idx += 1
    ax.legend(legend)

    return (fig, ax)

def load_plot_data(fname):
    """
    Given [fname], read file in and extract
    plot data in desired format.
    """
    times = []
    fids = []
    errs = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            T, fid, err = [float(x) for x in line.split(',')]
            times.append(T)
            fids.append(fid)
            errs.append(err)

    return [np.array(times), np.array(fids), np.array(errs)]

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in

def plot_fid_decay(plot_datas, with_fit=False, title='', legend=None):
    """
    Plots [self.plot_data] assuming it was produced
    from fidelity decay experiment.
    """
    fig, ax = plt.subplots(figsize=set_size(400))
    prop = ax._get_lines.prop_cycler
    for p_data in plot_datas:
        times, fids, errs = p_data
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
            fit_params, par_params, sum_res = bootstrap_fit_aug_exp(p_data)
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
            legend.append(f'$\\lambda$ = {st_tau} $\pm$ {st_tau_err}')
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

#############################################
# Reproducibility Statistics Utilities
#############################################
def order_files(files):
    """
    Orders files by job #.
    """
    job_nums = []
    for f in files:
        split_f = f.split('_')
        job_idx = split_f.index('job')
        job_n = int(split_f[job_idx + 1])
        job_nums.append(job_n)

    new_order = np.argsort(job_nums)
    new_files = []
    for i in new_order:
        new_files.append(files[i])

    return new_files

def get_data_over_calibs(file_list):
    """
    Given a file list of data sets, organize the
    data into a dictionary of the form
    {'calib 1 date': [job0, job1, ..., jobn],
    'calib 2 date': [job0, job1, ..., jobn]}
    """
    data_over_calib = {}

    for f in file_list:
        # get job index
        split_f = f.split('_')
        job_idx = split_f.index('job')
        job_n = int(split_f[job_idx + 1])
        data = IBMQData(name=f"job_{job_n}")
        data.load_raw_data(f)
        data.format_data()

        # get txt file and extract calib date
        txt_f = f"{f[:f.find('_raw_data')]}_properties.txt"
        with open(txt_f, 'r') as tf:
            update = tf.readlines()[5]
            date = update[18:]
            if date not in data_over_calib:
                data_over_calib[date] = [data]
            else:
                data_over_calib[date].append(data)

    return data_over_calib

def perform_calib_fid_analysis(data_over_calib, file_list, label='free'):
    """
    Given a dictionary with data over calibration cycles
    and the files (in same order) from which it was made,
    performs analysis to extract fidelities across each job.
    Gives the results in the form of four dictionaries
    1. split_job_fid_dict
    2. per_calib_fid_dict
    3. job_len_dict
    4. complete_times_dict
    """
    split_job_fid_dict = {}
    per_calib_fid_dict = {}
    job_len_dict = {}
    complete_times_dict = {}

    tot_idx = 0
    for key in data_over_calib:
        key_fid_list = []
        job_len_list = []
        complete_times = []
        for data in data_over_calib[key]:
            dat_file = file_list[tot_idx]
            tot_idx += 1
            job_idx = dat_file.split('_').index('job')
            c_time = dat_file.split('_')[job_idx + 3][0:-3]
            complete_times.append(c_time)
            fid_list = []
            job_len = 0
            data.change_working_data_by_label(label)
            for tup in data.working_data:
                job_len += 1
                counts = tup[2]
                fid = counts['0x0'] / (counts['0x0'] + counts['0x1'])
                fid_list.append(fid)

            key_fid_list.append(fid_list)
            job_len_list.append(job_len)

        split_job_fid_dict[key] = key_fid_list
        flat_fid_list = [item for sublist in key_fid_list for item in sublist]
        per_calib_fid_dict[key] = flat_fid_list
        job_len_dict[key] = job_len_list
        complete_times_dict[key] = complete_times

    return split_job_fid_dict, per_calib_fid_dict, job_len_dict, complete_times_dict

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

    std = np.std(simulations)
    return np.mean(data), 2*std

def boot_fid(fid, shots=8000):
    """
    Given a [fid], bootstraps the
    error given that [shots] shots
    were taken to obtain [fid].
    """
    n0 = int(np.ceil(shots * fid))
    n1 = shots - n0
    fake_data = []
    for j in range(n0):
        fake_data.append(1)
    for j in range(n1):
        fake_data.append(0)
    random.shuffle(fake_data)

    return bootstrapci(fake_data)

#############################################
# Fitting Utilties
#############################################
##########################
# Exp + Cos fit
#########################
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

###########################################
# pub(lication) fit
###########################################
def pub_func(p, t):
    """
    F(t) = 1/2(p[0]*e^{-p[1] t} + 1)*Cos(p[2]*t)
    """
    mid_val = p[0]*np.exp(-p[1]*t)*np.cos(p[2]*t)
    return (1/2)*(mid_val + 1)

def perform_pub_fit(plot_data, seed=None):
    """
    Takes plot_data (x, y, yerr) & fits a function of the form
    F(t) = pub_func(p, t)
    such that sum{[(F(t) - y)/yerr]^2} is minimized, i.e.
    least squares weighted by errors.
    """
    t, fid, yerr = plot_data
    ysig = (yerr / 2)

    def err_func(p, t, fid, err):
        return sum(((fid - pub_func(p, t)) / err)**2)

    if seed is None:
        p_init = [1.0, 1 / 10, 1 / 40]
    else:
        p_init = seed

    bnds = ((0.0, 1.0), (0.0, 10.0), (0.0, 10.0))
    out = scipy.optimize.minimize(err_func, p_init, args=(t, fid, ysig), bounds=bnds)

    # collect fitting parameters
    a0, lam, gam = out.x
    # get sum of squared residuals
    res_sum = np.sum((pub_func([a0, lam, gam], t) - fid)**2)
    red_chi_sq = res_sum / (len(fid) - 2)

    return a0, lam, gam, res_sum

def bootstrap_pub_fit(plot_data, boot_samps=1000, seed=None):
    """
    Bootstrap fit of [plots_data] to the aug_exp function.
    """
    a0_list = []
    lam_list = []
    gam_list = []

    # do fit on original data
    a0, lam, gam, _ = perform_pub_fit(plot_data, seed)
    a0_list.append(a0)
    lam_list.append(lam)
    gam_list.append(gam)
    seed = [a0, lam, gam]

    for i in range(boot_samps):
        # generate sample of plot_data
        samp_plot_data = gen_time_series_sample(plot_data)
        # fit this data and add to lists
        a0, lam, gam, _ = perform_pub_fit(samp_plot_data, seed)
        a0_list.append(a0)
        lam_list.append(lam)
        gam_list.append(gam)

    avg_a0 = np.mean(a0_list)
    a0_std = 2*np.std(a0_list)
    avg_lam = np.mean(lam_list)
    lam_std = 2*np.std(lam_list)
    avg_gam = np.mean(gam_list)
    gam_std = 2*np.std(gam_list)

    # obtain sum of residuals squared
    t, fid, _ = plot_data
    rsq_sum = np.sum((pub_func([avg_a0, avg_lam, avg_gam], t) - fid)**2)

    return [avg_a0, a0_std], [avg_lam, lam_std], [avg_gam, gam_std], rsq_sum

###########################################
# aug_cosexp fit
###########################################

def aug_cosexp_func(p, t):
    """
    F(t) = 1/2(p[0]*e^{-t / p[1]} + 1)*Cos(p[2]*t)
    """
    mid_val = p[0]*np.exp(-t/p[1])*np.cos(t/p[2])
    return (1/2)*(mid_val + 1)

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
        p_init = [1.0, 50.0, 50.0]
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

def bootstrap_fit_aug_cosexp(plot_data, boot_samps=1000, seed=None):
    """
    Bootstrap fit of [plots_data] to the aug_exp function.
    """
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

##########################
# Exp Fit
#########################
def aug_exp_func(p, t):
    """
    F(t) = 1/2(p[0]*e^{-t / p[1]} + 1)
    """
    # define f(t)
    return (1/2)*(p[0]*np.exp(-t/p[1]) + 1)

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
    else:
        a0_err = 'undetermined'
        tau_err = 'undetermined'

    return [tau, tau_err], [a0, a0_err], res_sum

def bootstrap_fit_aug_exp(plot_data, boot_samps=1000, seed=None):
    """
    Bootstrap fit of [plots_data] to the aug_exp function.
    """
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


#############################################
# Job Submission Utilties
#############################################
def get_max_tau(seq, max_T, backend, tau_min=0, tau_max=800, tol=1):
    """
    Finds the maximum value of [tau] that a
    sequence can have before it runs longer than
    [max_T].
    """
    sched = IBMQDdSchedule(backend, 'c_basis')
    # get mid point tau
    tau_mid = int(np.floor((tau_max + tau_min) / 2))
    # time it
    getattr(sched, f"add_{seq}")(0, 1, tau_mid)
    T = sched.get_phys_time() / 1000 # get in micro-s
    diff = max_T - T
    if np.abs(diff) < tol:
        return tau_mid
    elif tau_mid == 1:
        return tau_mid
    elif T < max_T:
        return get_max_tau(seq, max_T, backend, tau_mid, tau_max, tol=tol)
    else:
        return get_max_tau(seq, max_T, backend, tau_min, tau_mid, tol=tol)


def extract_T(experiment):
    """
    Extract the T value for an experiment in nano seconds.
    """
    split_label = experiment.name.split('_')
    T_idx = split_label.index('T')
    T_val = float(split_label[T_idx + 1][0:-2])

    return T_val

def chunk_experiments(experiments, chunk_size):
    """
    Chunks list of [experiments] into lists of size [chunk_size].
    """
    num_lists = int(np.ceil(len(experiments) / chunk_size))

    for j in range(num_lists):
        yield experiments[chunk_size*j:chunk_size*(j+1)]

def group_experiments_by_duration(experiments, num_bins):
    """
    Bins experiments by DD sequence time. Chooses bins to
    try and make all bins contain same number of elements.
    """
    binned_experiments = [[] for _ in range(num_bins)]
    # first, extract the times
    times = []
    for exper in experiments:
        T_val = extract_T(exper)
        times.append(T_val)

    # bin the times (maintains order of data)
    bin_info = pd.qcut(times, num_bins)
    interval_list = bin_info.categories

    # iterate through the experiments and add them to correct bin
    for (o_idx, bi) in enumerate(bin_info):
        for (i_idx, interval) in enumerate(interval_list):
            if bi.overlaps(interval):
                binned_experiments[i_idx].append(experiments[o_idx])
                break

    return binned_experiments

def gen_fds_on_fly_and_submit(backend, rep_dict, sched_settings, trials,
                              max_tries_on_fail=3, max_job_size=75):
    """
    Generates fixed-delay schedules on the fly and then submits them.
    """
    waittime = 1200
    # get data containers ready
    back_name = backend.props['backend_name']
    result_list = []
    failed_experiments = []

    # create "binned experiments list" which consists of list of list
    # each sublist contains ~75 experiments to submit in one job
    exps_to_run = []
    seqs = rep_dict.keys()
    for _ in range(trials):
        exps_to_run.extend(seqs)

    # unravel sched settings for schedule creation on the fly
    offset, sym, delay, d_label, basis, encoding_qubit, dd_qubit = sched_settings
    qubit = encoding_qubit

    # run first attempt of experiments
    active_jobs = []
    for idx, seq in enumerate(exps_to_run):
        print(f"job idx being tried: {idx} w/ seq {seq}")
        job_attempted = False
        while job_attempted is False:
            # check if any queue slots are availible
            avail_jobs = backend.backend.remaining_jobs_count()
            print(f"avail jobs: {avail_jobs}")
            if avail_jobs > 0:
                job_attempted = True
                try:
                    print(f"submitting job with idx: {idx} w/ seq {seq}")
                    # submit job with random ordering of individual circuits run
                    scheds_to_submit = []
                    for reps in rep_dict[seq]:
                        desc, scheds = edde.pulse.pauli_pode_fid_decay_dd(offset, seq, sym, reps, delay, d_label, backend, basis, encoding_qubit, dd_qubit)
                        scheds_to_submit.extend(scheds)
                    error_mit_0 = IBMQDdSchedule(backend, basis, name='error_mitigate_0')
                    error_mit_0.add_error_mitigation_0(qubit)
                    scheds_to_submit.append(error_mit_0)
                    error_mit_1 = IBMQDdSchedule(backend, basis, name='error_mitigate_1')
                    error_mit_1.add_error_mitigation_1(qubit)
                    scheds_to_submit.append(error_mit_1)
                    job = backend.submit_job(scheds_to_submit, f"job_{idx}", num_shots=8192, shuffle=True)
                    # extract calibrated pules and add to active jobs list
                    x_pulse = str(error_mit_0.basis[f'X_{dd_qubit}'])
                    y_pulse = str(error_mit_0.basis[f'Y_{dd_qubit}'])
                    active_jobs.append((idx, job, x_pulse, y_pulse))
                except:
                    print(f"job with idx: {idx} failed on first try w/ seq {seq}")
                    failed_experiments.append((idx, seq))
            # otherwise, wait a few minutes and try again
            else:
                print(f"No availible queue slots for this account. Going to wait {waittime}s and try again.")
                time.sleep(waittime)
                # check on status of jobs and extract/save results if applicable
                idxs_to_remove = []
                for (a_idx, a_job) in enumerate(active_jobs):
                    if a_job[1].done() is True:
                        print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                        # first, load in backend to extract most recent "data sheet"
                        backend.change_backend(back_name)
                        # save useful job info
                        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                        backend.save_backend_props(backend_fname + "properties.txt")
                        # add info about pulses used and current pulses calibrated
                        sched = IBMQDdSchedule(backend, basis, name='get_pulses')
                        curr_xp = str(sched.basis[f'X_{dd_qubit}'])
                        curr_yp = str(sched.basis[f'Y_{dd_qubit}'])
                        with open(backend_fname + "properties.txt", 'a') as f:
                            f.write("\nPulse calibration information\n")
                            f.write("---\n")
                            f.write(f"used X: {a_job[2]}\n")
                            f.write(f"currently calibrated X: {curr_xp}\n")
                            f.write(f"used Y: {a_job[3]}\n")
                            f.write(f"currently calibrated Y: {curr_yp}\n")
                        # save result
                        result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                        result_list.append(result)
                        data = IBMQData(result)
                        data.save_raw_data(backend_fname + "raw_data")
                        idxs_to_remove.append(a_idx)
                # remove the finished jobs from active_job list
                for j in sorted(idxs_to_remove, reverse=True):
                    del active_jobs[j]

    # run second attempt of experiments provided that they didn't succeed the first time
    trial = 1
    while trial < max_tries_on_fail:
        trial += 1
        if len(failed_experiments) > 0:
            successful_idx = []
            for pos_idx, seq in enumerate(failed_experiments):
                job_attempted = False
                while job_attempted is False:
                    avail_jobs = backend.backend.remaining_jobs_count()
                    if avail_jobs > 0:
                        job_attempted = True
                        try:
                            print(f"submitting job with idx: {idx}")
                            # submit job with random ordering of individual circuits run
                            scheds_to_submit = []
                            for reps in rep_dict[seq]:
                                desc, scheds = edde.pulse.pauli_pode_fid_decay_dd(offset, seq, sym, reps, delay, d_label, backend, basis, encoding_qubit, dd_qubit)
                                scheds_to_submit.extend(scheds)
                            error_mit_0 = IBMQDdSchedule(backend, basis, name='error_mitigate_0')
                            error_mit_0.add_error_mitigation_0(qubit)
                            scheds_to_submit.append(error_mit_0)
                            error_mit_1 = IBMQDdSchedule(backend, basis, name='error_mitigate_1')
                            error_mit_1.add_error_mitigation_1(qubit)
                            scheds_to_submit.append(error_mit_1)
                            job = backend.submit_job(scheds_to_submit, f"job_{idx}", num_shots=8192, shuffle=True)
                            # extract calibrated pules and add to active jobs list
                            x_pulse = str(error_mit_0.basis[f'X_{dd_qubit}'])
                            y_pulse = str(error_mit_0.basis[f'Y_{dd_qubit}'])
                            active_jobs.append((idx, job, x_pulse, y_pulse))
                            successful_idx.append(idx)
                        except:
                            print(f"job with idx: {idx} failed on {pos_idx+1} try")
                            continue
                    # otherwise, wait a few minutes and try again
                    else:
                        print(f"No availible queue slots for this account. Going to wait {waittime}s and try again.")
                        time.sleep(waittime)
                        # check on status of jobs and extract/save results if applicable
                        idxs_to_remove = []
                        for (a_idx, a_job) in enumerate(active_jobs):
                            if a_job[1].done() is True:
                                print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                                # first, load in backend to extract most recent "data sheet"
                                backend.change_backend(back_name)
                                # save useful job info
                                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                                backend.save_backend_props(backend_fname + "properties.txt")
                                # add info about pulses used and current pulses calibrated
                                sched = IBMQDdSchedule(backend, basis, name='get_pulses')
                                curr_xp = str(sched.basis[f'X_{dd_qubit}'])
                                curr_yp = str(sched.basis[f'Y_{dd_qubit}'])
                                with open(backend_fname + "properties.txt", 'a') as f:
                                    f.write("\nPulse calibration information\n")
                                    f.write("---\n")
                                    f.write(f"used X: {a_job[2]}\n")
                                    f.write(f"currently calibrated X: {curr_xp}\n")
                                    f.write(f"used Y: {a_job[3]}\n")
                                    f.write(f"currently calibrated Y: {curr_yp}\n")
                                # save result
                                result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                                result_list.append(result)
                                data = IBMQData(result)
                                data.save_raw_data(backend_fname + "raw_data")
                                idxs_to_remove.append(a_idx)
                        # remove the finished jobs from active_job list
                        for j in sorted(idxs_to_remove, reverse=True):
                            del active_jobs[j]

                # remove successful experiments (reverse order prevent in-place bugs)
                for idx in sorted(successful_idx, reverse=True):
                    del failed_experiments[idx]

    # for remaining active jobs, wait on results to come in
    count = 0
    while len(active_jobs) != 0:
        print(f"All jobs submitted. Awaiting retrieval. (This message is sent every {waittime}s.)")
        if count > 0:
            time.sleep(waittime)
        # check on status of jobs and extract/save results if applicable
        idxs_to_remove = []
        for (a_idx, a_job) in enumerate(active_jobs):
            if a_job[1].done() is True:
                print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                # first, load in backend to extract most recent "data sheet"
                backend.change_backend(back_name)
                # save useful job info
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                backend.save_backend_props(backend_fname + "properties.txt")
                # add info about pulses used and current pulses calibrated
                sched = IBMQDdSchedule(backend, basis, name='get_pulses')
                curr_xp = str(sched.basis[f'X_{dd_qubit}'])
                curr_yp = str(sched.basis[f'Y_{dd_qubit}'])
                with open(backend_fname + "properties.txt", 'a') as f:
                    f.write("\nPulse calibration information\n")
                    f.write("---\n")
                    f.write(f"used X: {a_job[2]}\n")
                    f.write(f"currently calibrated X: {curr_xp}\n")
                    f.write(f"used Y: {a_job[3]}\n")
                    f.write(f"currently calibrated Y: {curr_yp}\n")
                # save result
                result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                result_list.append(result)
                data = IBMQData(result)
                data.save_raw_data(backend_fname + "raw_data")
                idxs_to_remove.append(a_idx)
        # remove the finished jobs from active_job list
        for j in sorted(idxs_to_remove, reverse=True):
            del active_jobs[j]

        count += 1

    return result_list, failed_experiments


def gen_haar_fds_on_fly_and_submit(backend, rep_dict, seq_settings,
                                   generic_settings,
                                   trials, max_tries_on_fail=3,
                                   max_job_size=75):
    """
    Generates fixed-delay schedules on the fly and then submits them.
    """
    waittime = 600
    # get data containers ready
    back_name = backend.props['backend_name']
    result_list = []
    failed_experiments = []

    # create "binned experiments list" which consists of list of list
    # each sublist contains ~75 experiments to submit in one job
    exps_to_run = []
    seqs = rep_dict.keys()
    for _ in range(trials):
        exps_to_run.extend(seqs)

    # unravel sched settings for schedule creation on the fly
    N, d_label, encoding_qubits, dd_qubits = generic_settings
    qubit = encoding_qubits

    # run first attempt of experiments
    active_jobs = []
    for idx, seq in enumerate(exps_to_run):
        print(f"job idx being tried: {idx} w/ seq {seq}")
        job_attempted = False
        while job_attempted is False:
            # check if any queue slots are availible
            avail_jobs = backend.backend.remaining_jobs_count()
            print(f"avail jobs: {avail_jobs}")
            if avail_jobs > 0:
                job_attempted = True
                try:
                    print(f"submitting job with idx: {idx} w/ seq {seq}")
                    # submit job with random ordering of individual circuits run
                    reps, pad = rep_dict[seq]
                    basis, sym, delay = seq_settings[seq]
                    desc, scheds = edde.pulse.haar_fid_decay_dd(N, seq, sym, reps, pad, delay, d_label, backend, basis, encoding_qubits, dd_qubits)
                    job = backend.submit_job(scheds, f"job_{idx}", num_shots='max', shuffle=True)
                    # extract calibrated pules and add to active jobs list
                    x_pulse = str(scheds[0].basis[f'X_{dd_qubits}'])
                    y_pulse = str(scheds[0].basis[f'Y_{dd_qubits}'])
                    active_jobs.append((idx, job, x_pulse, y_pulse))
                except:
                    print(f"job with idx: {idx} failed on first try w/ seq {seq}")
                    failed_experiments.append((idx, seq))
            # otherwise, wait a few minutes and try again
            else:
                print(f"No availible queue slots for this account. Going to wait {waittime}s and try again.")
                time.sleep(waittime)
                # check on status of jobs and extract/save results if applicable
                idxs_to_remove = []
                for (a_idx, a_job) in enumerate(active_jobs):
                    if a_job[1].done() is True:
                        print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                        # first, load in backend to extract most recent "data sheet"
                        backend.change_backend(back_name)
                        # save useful job info
                        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                        backend.save_backend_props(backend_fname + "properties.txt")
                        # add info about pulses used and current pulses calibrated
                        sched = IBMQDdSchedule(backend, basis, name='get_pulses')
                        curr_xp = str(sched.basis[f'X_{dd_qubits}'])
                        curr_yp = str(sched.basis[f'Y_{dd_qubits}'])
                        with open(backend_fname + "properties.txt", 'a') as f:
                            f.write("\nPulse calibration information\n")
                            f.write("---\n")
                            f.write(f"used X: {a_job[2]}\n")
                            f.write(f"currently calibrated X: {curr_xp}\n")
                            f.write(f"used Y: {a_job[3]}\n")
                            f.write(f"currently calibrated Y: {curr_yp}\n")
                        # save result
                        result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                        result_list.append(result)
                        data = IBMQData(result)
                        data.save_raw_data(backend_fname + "raw_data")
                        idxs_to_remove.append(a_idx)
                # remove the finished jobs from active_job list
                for j in sorted(idxs_to_remove, reverse=True):
                    del active_jobs[j]

    # run second attempt of experiments provided that they didn't succeed the first time
    trial = 1
    while trial < max_tries_on_fail:
        trial += 1
        if len(failed_experiments) > 0:
            successful_idx = []
            for pos_idx, seq in enumerate(failed_experiments):
                job_attempted = False
                while job_attempted is False:
                    avail_jobs = backend.backend.remaining_jobs_count()
                    if avail_jobs > 0:
                        job_attempted = True
                        try:
                            print(f"submitting job with idx: {idx}")
                            # submit job with random ordering of individual circuits run
                            reps, pad = rep_dict[seq]
                            basis, sym, delay = seq_settings[seq]
                            desc, scheds = edde.pulse.haar_fid_decay_dd(N, seq, sym, reps, pad, delay, d_label, backend, basis, encoding_qubits, dd_qubits)
                            job = backend.submit_job(scheds, f"job_{idx}", num_shots='max', shuffle=True)
                            # extract calibrated pules and add to active jobs list
                            x_pulse = str(scheds[0].basis[f'X_{dd_qubits}'])
                            y_pulse = str(scheds[0].basis[f'Y_{dd_qubits}'])
                            active_jobs.append((idx, job, x_pulse, y_pulse))
                            successful_idx.append(idx)
                        except:
                            print(f"job with idx: {idx} failed on {pos_idx+1} try")
                            continue
                    # otherwise, wait a few minutes and try again
                    else:
                        print(f"No availible queue slots for this account. Going to wait {waittime}s and try again.")
                        time.sleep(waittime)
                        # check on status of jobs and extract/save results if applicable
                        idxs_to_remove = []
                        for (a_idx, a_job) in enumerate(active_jobs):
                            if a_job[1].done() is True:
                                print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                                # first, load in backend to extract most recent "data sheet"
                                backend.change_backend(back_name)
                                # save useful job info
                                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                                backend.save_backend_props(backend_fname + "properties.txt")
                                # add info about pulses used and current pulses calibrated
                                sched = IBMQDdSchedule(backend, basis, name='get_pulses')
                                curr_xp = str(sched.basis[f'X_{dd_qubits}'])
                                curr_yp = str(sched.basis[f'Y_{dd_qubits}'])
                                with open(backend_fname + "properties.txt", 'a') as f:
                                    f.write("\nPulse calibration information\n")
                                    f.write("---\n")
                                    f.write(f"used X: {a_job[2]}\n")
                                    f.write(f"currently calibrated X: {curr_xp}\n")
                                    f.write(f"used Y: {a_job[3]}\n")
                                    f.write(f"currently calibrated Y: {curr_yp}\n")
                                # save result
                                result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                                result_list.append(result)
                                data = IBMQData(result)
                                data.save_raw_data(backend_fname + "raw_data")
                                idxs_to_remove.append(a_idx)
                        # remove the finished jobs from active_job list
                        for j in sorted(idxs_to_remove, reverse=True):
                            del active_jobs[j]

                # remove successful experiments (reverse order prevent in-place bugs)
                for idx in sorted(successful_idx, reverse=True):
                    del failed_experiments[idx]

    # for remaining active jobs, wait on results to come in
    count = 0
    while len(active_jobs) != 0:
        print(f"All jobs submitted. Awaiting retrieval. (This message is sent every {waittime}s.)")
        if count > 0:
            time.sleep(waittime)
        # check on status of jobs and extract/save results if applicable
        idxs_to_remove = []
        for (a_idx, a_job) in enumerate(active_jobs):
            if a_job[1].done() is True:
                print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                # first, load in backend to extract most recent "data sheet"
                backend.change_backend(back_name)
                # save useful job info
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                backend.save_backend_props(backend_fname + "properties.txt")
                # add info about pulses used and current pulses calibrated
                sched = IBMQDdSchedule(backend, basis, name='get_pulses')
                curr_xp = str(sched.basis[f'X_{dd_qubits}'])
                curr_yp = str(sched.basis[f'Y_{dd_qubits}'])
                with open(backend_fname + "properties.txt", 'a') as f:
                    f.write("\nPulse calibration information\n")
                    f.write("---\n")
                    f.write(f"used X: {a_job[2]}\n")
                    f.write(f"currently calibrated X: {curr_xp}\n")
                    f.write(f"used Y: {a_job[3]}\n")
                    f.write(f"currently calibrated Y: {curr_yp}\n")
                # save result
                result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                result_list.append(result)
                data = IBMQData(result)
                data.save_raw_data(backend_fname + "raw_data")
                idxs_to_remove.append(a_idx)
        # remove the finished jobs from active_job list
        for j in sorted(idxs_to_remove, reverse=True):
            del active_jobs[j]

        count += 1

    return result_list, failed_experiments

def loop_binned_experiments(backend, binned_experiments, max_tries=3, max_job_size=75):
    """
    Submits jobs that are binned together in same queue. At any given time,
    can submit up to 5 jobs at once--but ensures that it's never the case
    that a single job contains two different bin sets--even if this
    isn't maximizing number of circuits per job. As a queue slot
    frees up, automatically queues up new jobs.
    """
    waittime = 600
    # get data containers ready
    back_name = backend.props['backend_name']
    result_list = []
    failed_experiments = []

    # randomly shuffle the bin order
    #random.shuffle(binned_experiments)

    # reduce each job submission down to [max_job_size] circuits/schedules
    exps_to_run = []
    for exp in binned_experiments:
        chunked_exp = list(chunk_experiments(exp, max_job_size))
        exps_to_run.extend(chunked_exp)

    # run first attempt of experiments
    active_jobs = []
    for idx, exp_group in enumerate(exps_to_run):
        print(f"job idx being tried: {idx}")
        job_attempted = False
        while job_attempted is False:
            # check if any queue slots are availible
            avail_jobs = backend.backend.remaining_jobs_count()
            print(f"avail jobs: {avail_jobs}")
            if avail_jobs > 0:
                job_attempted = True
                try:
                    print(f"submitting job with idx: {idx}")
                    # submit job with random ordering of individual circuits run
                    job = backend.submit_job(exp_group, f"job_{idx}", num_shots=8000, shuffle=True)
                    active_jobs.append((idx, job))
                except:
                    print(f"job with idx: {idx} failed on first try")
                    failed_experiments.append((idx, exp))
            # otherwise, wait a few minutes and try again
            else:
                print(f"No availible queue slots for this account. Going to wait {waittime}s and try again.")
                time.sleep(waittime)
                # check on status of jobs and extract/save results if applicable
                idxs_to_remove = []
                for (a_idx, a_job) in enumerate(active_jobs):
                    if a_job[1].done() is True:
                        print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                        # first, load in backend to extract most recent "data sheet"
                        backend.change_backend(back_name)
                        # save useful job info
                        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                        backend.save_backend_props(backend_fname + "properties.txt")
                        # save result
                        result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                        result_list.append(result)
                        data = IBMQData(result)
                        data.save_raw_data(backend_fname + "raw_data")
                        idxs_to_remove.append(a_idx)
                # remove the finished jobs from active_job list
                for j in sorted(idxs_to_remove, reverse=True):
                    del active_jobs[j]

    # run second attempt of experiments provided that they didn't succeed the first time
    trial = 1
    while trial < max_tries:
        trial += 1
        if len(failed_experiments) > 0:
            successful_idx = []
            for pos_idx, exp_tup in enumerate(failed_experiments):
                job_attempted = False
                while job_attempted is False:
                    avail_jobs = backend.backend.remaining_jobs_count()
                    if avail_jobs > 0:
                        job_attempted = True
                        try:
                            print(f"submitting job with idx: {idx}")
                            # submit job with random ordering of individual circuits run
                            job = backend.submit_job(exp_group, f"job_{idx}", num_shots=8000, shuffle=True)
                            active_jobs.append((idx, job))
                            successful_idx.append(idx)
                        except:
                            print(f"job with idx: {idx} failed on {pos_idx+1} try")
                            continue
                    # otherwise, wait a few minutes and try again
                    else:
                        print(f"No availible queue slots for this account. Going to wait {waittime}s and try again.")
                        time.sleep(waittime)
                        # check on status of jobs and extract/save results if applicable
                        idxs_to_remove = []
                        for (a_idx, a_job) in enumerate(active_jobs):
                            if a_job[1].done() is True:
                                print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                                # first, load in backend to extract most recent "data sheet"
                                backend.change_backend(back_name)
                                # save useful job info
                                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                                backend.save_backend_props(backend_fname + "properties.txt")
                                # save result
                                result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                                result_list.append(result)
                                data = IBMQData(result)
                                data.save_raw_data(backend_fname + "raw_data")
                                idxs_to_remove.append(a_idx)
                        # remove the finished jobs from active_job list
                        for j in sorted(idxs_to_remove, reverse=True):
                            del active_jobs[j]

                # remove successful experiments (reverse order prevent in-place bugs)
                for idx in sorted(successful_idx, reverse=True):
                    del failed_experiments[idx]

    # for remaining active jobs, wait on results to come in
    count = 0
    while len(active_jobs) != 0:
        print(f"All jobs submitted. Awaiting retrieval. (This message is sent every {waittime}s.)")
        if count > 0:
            time.sleep(waittime)
        # check on status of jobs and extract/save results if applicable
        idxs_to_remove = []
        for (a_idx, a_job) in enumerate(active_jobs):
            if a_job[1].done() is True:
                print(f"Data for job with idx {a_job[0]} retrieved and will be saved.")
                # first, load in backend to extract most recent "data sheet"
                backend.change_backend(back_name)
                # save useful job info
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                backend_fname = back_name + "_" + f"job_{a_job[0]}" +  "_" + now + "_"
                backend.save_backend_props(backend_fname + "properties.txt")
                # save result
                result = backend.backend.retrieve_job(a_job[1].job_id()).result()
                result_list.append(result)
                data = IBMQData(result)
                data.save_raw_data(backend_fname + "raw_data")
                idxs_to_remove.append(a_idx)
        # remove the finished jobs from active_job list
        for j in sorted(idxs_to_remove, reverse=True):
            del active_jobs[j]

        count += 1

    return result_list, failed_experiments

def cancel_most_recent_job(ibmq_backend):
    """
    Cancels the most recent job sent to [ibmq_backend].
    """
    ibmq_backend.backend.active_jobs()[0].cancel()
    return

def cancel_all_jobs(ibmq_backend):
    """
    Cancels all jobs on [ibmq_backend].
    """
    while ibmq_backend.backend.remaining_jobs_count() < 5:
        cancel_most_recent_job(ibmq_backend)
    return

