# edd
Experimental dynamical decoupling (EDD) is a repo dedicated to testing out various DD sequences on NISQ-era quantum computers. Further details of what this means are contained in the associated paper: "Dynamical decoupling for superconducting qubits: a performance survey" by Nic Ezzell, Bibek Pokharel, Lina Tewala, Gregory Quiroz, and Daniel A. Lidar. 

In this README, we shall discuss (i) how to download and install edd (ii) the package structure and useage (iii) a dicussion of which parts of the repo are relevant to our paper and how to obtain/ analyze our data and (iv) some special cavetas/ considerations when using edd with IBM devices.

## (i) How to download and install edd
Simply clone the repo in your preferred directory and run `pip install -e .` within your desired python virtual environment/ conda environment while located in the edd directory with the setup.py file. If this doesn't make sense to you, please refer to the more detailed instructions in "setup_commands.txt." 

## (ii) Package structure and usage
We modeled our package structure after `numpy`, but by no means did we try to make things "production" quality. In other words, don't be surprised to find defunct functions or weird conventions. 

### Summary of structure

There are three main components to our code. 
- `src/edd` -- hosts our custom modules that interface directly with the IBM quantum devices, i.e. to write, send, and process quantum memory experiments
- `dataAndAnalysis` -- contains scripts to wrangle and analyze the experiment data alongside the raw data itself
- everything else -- contains requirements and instructions for a local install via pip

Using the code in `src/edd` is like using `numpy` or `qiskit` itself, so let us explain it further. We will discuss the structure of the `dataAndAnalysis` in Section (iii). 


This code in the `src/edd` directory has several sub-directories
- `backend` -- contains the `IBMQBackend` class for sending/receiving jobs from IBM
- `pulse` -- contains the `IBMQDdSchedule` class which inherits from `qiskit.pulse.Schedule` class. Contains the DD sequence definitions in terms of pulse instructions. Allows one to add DD sequences to it as methods. 
- `circuit` -- contains `IBMQDdCircuit` class which inherits from qiskit.QuantumCircuit but otherwise should behave roughly the same as `pulse`. However, we ended up not using this class for the paper, so no promises that it's on par with `pulse`
- `experiments` -- contains scripts to run various experiments we were interested in at various points. For example, in `_pulse_experiments.py` we have a function called `pauli_pode_fid_decay_dd` which is the quantum memory experiment on the Pauli eigenstates. Also, `haar_fid_decay_dd` contains script to run the same for Haar random states. 
- `data` -- contains `IBMQData` class which is initialized with job data from IBM devices and can re-format and summarize it with useful statistics
- `workflow` -- contains scripts useful for automating the job-submission, data extraction, and analysis for experiments used in the paper. For example, `gen_fds_on_fly_and_submit` was used to submit many circuits for this purpose. 
- `states` -- contains "u3_list.npy" which encodes the u3 parameters which define the static set of Haar random states we test over

### Import patterns
Basic usage follows a `numpy` like convention. For example, a typical set of imports looks like
```
from edd.backend import IBMQBackend
from edd.pulse import IBMQDdSchedule
from edd.data import IBMQData
import edd.experiments as edde
import edd.workflow as wf
```
### Loading in IBM Backend
When using the `IBMQDdSchedule` class, note that all pulses are defined relative to a backend, so you'll want to first load in a pulse-compatible backend. To do this, you can just run something like
```
backend_name = "ibmq_example"
hub = "ibm-q-research-or-whatever"
group = "perhaps-your-uni"
project = "the-funds"
token = "your-token"
backend = IBMQBackend(backend, hub, group, project, token)
```
where the hub, group, project, and token info can be obtained by going to `IBM quantum > account settings > providers`. Then click the three vertical dots and select `copy Qiskit provider code.` If typing this is annoys you, you can hardcode your account as an `__init__` option. An important functon to be aware of when running experiments is
```
print(backend.get_readable_props_str())
```
which gives a human readable summary of the backend properties which can be saved to a txt file easily. For `jakarta`, for example, the result as of this tutorial are
```
Backend Properties
---
Experiment Date: 2022-07-07 12:06:44
Backend Name: ibmq_jakarta
Version: 1.0.34
Last Update Date: 2022-07-07 08:21:07-07:00

Gate Info
---
name, gate_error(), gate_length(ns)
id, 0.0003199501648000584, 35.55555555555556
id, 0.00020741584099435536, 35.55555555555556
id, 0.00020747484772500605, 35.55555555555556
id, 0.0002035022703507296, 35.55555555555556
id, 0.0010525762631233982, 35.55555555555556
id, 0.0002491638513245879, 35.55555555555556
id, 0.00020354933384976543, 35.55555555555556
rz, 0, 0
rz, 0, 0
rz, 0, 0
rz, 0, 0
rz, 0, 0
rz, 0, 0
rz, 0, 0
sx, 0.0003199501648000584, 35.55555555555556
sx, 0.00020741584099435536, 35.55555555555556
sx, 0.00020747484772500605, 35.55555555555556
sx, 0.0002035022703507296, 35.55555555555556
sx, 0.0010525762631233982, 35.55555555555556
sx, 0.0002491638513245879, 35.55555555555556
sx, 0.00020354933384976543, 35.55555555555556
x, 0.0003199501648000584, 35.55555555555556
x, 0.00020741584099435536, 35.55555555555556
x, 0.00020747484772500605, 35.55555555555556
x, 0.0002035022703507296, 35.55555555555556
x, 0.0010525762631233982, 35.55555555555556
x, 0.0002491638513245879, 35.55555555555556
x, 0.00020354933384976543, 35.55555555555556
cx, 0.017376502926994886, 504.88888888888886
cx, 0.017376502926994886, 540.4444444444445
cx, 0.007410021627299507, 384
cx, 0.007410021627299507, 419.55555555555554
cx, 0.006437586645274163, 277.3333333333333
cx, 0.006437586645274163, 312.88888888888886
cx, 0.008834249450477588, 291.55555555555554
cx, 0.008834249450477588, 327.1111111111111
cx, 0.00980800779616306, 248.88888888888889
cx, 0.00980800779616306, 284.44444444444446
cx, 0.007000849427907713, 234.66666666666666
cx, 0.007000849427907713, 270.22222222222223
reset, 7342.222222222222
reset, 7342.222222222222
reset, 7342.222222222222
reset, 7342.222222222222
reset, 7342.222222222222
reset, 7342.222222222222
reset, 7342.222222222222

Qubit Info
---
qubit, T1(us), T2(us), frequency(GHz), anharmonicity(GHz), readout_error(), prob_meas0_prep1(), prob_meas1_prep0(), readout_length(ns)
0, 179.27108623920228, 46.48443007226778, 5.236537333392189, -0.339883615358574, 0.03959999999999997, 0.05479999999999996, 0.0244, 5351.11111111111
1, 136.6107664533625, 28.38193214382445, 5.014431945688961, -0.3432005583724651, 0.035599999999999965, 0.03739999999999999, 0.0338, 5351.11111111111
2, 115.41307756584426, 26.005733303914802, 5.108468919342932, -0.3416150041672664, 0.024499999999999966, 0.0388, 0.010199999999999987, 5351.11111111111
3, 130.6752912614701, 43.29522100023257, 5.178135251335165, -0.3411171247904715, 0.017800000000000038, 0.0268, 0.00880000000000003, 5351.11111111111
4, 43.33845082911979, 50.386887681694404, 5.213062099531775, -0.3392533874360392, 0.1964999999999999, 0.29479999999999995, 0.0982, 5351.11111111111
5, 69.68705534461373, 49.38499263027378, 5.063262326256089, -0.3412893561600795, 0.040100000000000025, 0.050799999999999956, 0.0294, 5351.11111111111
6, 99.3803716167542, 23.117838725006226, 5.300667969846487, -0.3383638923290693, 0.049900000000000055, 0.0364, 0.06340000000000001, 5351.11111111111
```
which as you can see is fairly comprehensive. 

### Created a DD pulse Schedule and visualizing it
 Now you can initialize a `IBMQDdSchedule` object, add DD pulses, and print the pulse sequence. For example,
```
# basis is an advanced key option to change the way 'x' and 'y' pulses are defined. Default is 'g_basis'
# name is just a tag that shows up when printing the schedule and whatnot
dd_sched = IBMQDdSchedule(backend, basis_version = "g_basis", name = "github_example")
dd_sched.add_xy4(qubit = 1, num_reps = 2, d = 0, sym = False)
dd_sched.add_measurement(1, 0)
print(dd_sched.sched)
```
which produces a result like
```
Schedule((0, Play(Drag(duration=160, amp=(-0.004039903559079659+0.20185383960793365j), sigma=40, beta=-1.017901529566831, name='Y'), DriveChannel(1), name='Y')), (160, Play(Drag(duration=160, amp=(0.20187820062948367+0j), sigma=40, beta=-1.0153527787687813, name='X'), DriveChannel(1), name='X')), (320, Play(Drag(duration=160, amp=(-0.004039903559079659+0.20185383960793365j), sigma=40, beta=-1.017901529566831, name='Y'), DriveChannel(1), name='Y')), (480, Play(Drag(duration=160, amp=(0.20187820062948367+0j), sigma=40, beta=-1.0153527787687813, name='X'), DriveChannel(1), name='X')), (640, Play(Drag(duration=160, amp=(-0.004039903559079659+0.20185383960793365j), sigma=40, beta=-1.017901529566831, name='Y'), DriveChannel(1), name='Y')), (800, Play(Drag(duration=160, amp=(0.20187820062948367+0j), sigma=40, beta=-1.0153527787687813, name='X'), DriveChannel(1), name='X')), (960, Play(Drag(duration=160, amp=(-0.004039903559079659+0.20185383960793365j), sigma=40, beta=-1.017901529566831, name='Y'), DriveChannel(1), name='Y')), (1120, Play(Drag(duration=160, amp=(0.20187820062948367+0j), sigma=40, beta=-1.0153527787687813, name='X'), DriveChannel(1), name='X')), (1280, Acquire(22400, AcquireChannel(1), MemorySlot(0))), (1280, Play(GaussianSquare(duration=22400, amp=(0.135263552870952+0.14115247523414948j), sigma=64, width=22144, name='M_m1'), MeasureChannel(1), name='M_m1')), (23680, Delay(1680, MeasureChannel(1))), name="github_example")
```
Pay special attention to the above notation of `dd_sched.sched` and the comment above it. To actually get the schedule object, we need to ask for it. In a way, the `IBMQDdSchedule` object is more of a wrapper than proper inheritance of `Schedule`, but again, this is just a technical detail to get changes to stay. In other words, you can continually add more pulses without previous additions getting thrown away. Anyway, the text is hard to understand, but luckily, you can invoke all the `Schedule` methods such as `draw`. So, 
```
dd_sched.draw()
```
<img width="1024" alt="Screen Shot 2022-07-07 at 11 59 53 AM" src="https://user-images.githubusercontent.com/29308150/177851523-a8bfe7f6-9489-474b-9ada-a72c18a8d34e.png">

### Submitting the schedule as a job
At this point, you can submit the pulse schedule as a job using the `backend.submit_job` function like
```
job = backend.submit_job(dd_sched, "qiskit_test", num_shots = 1000)
```
When the job is finished (you can check on it with `job.status()` or by looing at the online queue), you can pass the result object to the data class to perform data analysis. For example, 
```
result = job.result()
data = IBMQData(result)
```
However, the specific analysis we choose is only relevant to our partiuclar experiment. We will get into more details when we discuss how we used our package for our paper. But just for a glimpe into the ultimate simplicity (most of the complication is just keep track of which sequence/ paramters/ etc. were used), consider that
```
counts = result.get_counts()
```
gives the answer `counts = {'0': 981, '1': 19}` on this day which encodes an empirial Uhlmann fidelity of `981 / 1000 = 98.1%` which is consistent with expectations after merely applying 2 repetitions of XY4. 



## (iii) Relevant parts to paper
The job submission scripts, data analysis scripts, raw data, and machine calibration information is contained in `dataAndAnalysis`. Each relevant part here is zipped to prevent the repository from being too large. The zipped files consist of
- `eddDataWranling` -- contains scripts to wrangle data from experiments into a form to be analyzed
- `armonkDataAnalysis` -- contains processed ibmq_armonk data, scripts to submit jobs and anlayze the data, and all armonk graphs in paper
- `bogotaDataAnalysis` -- same as armonk above
- `jakartaDataAnalysis` -- same as armonk above
- `methodologyDataAnalysis` -- contains scripts for justifying our methodology (i.e. Appendix C)


`eddDataWrangling` file summary
- pauliData_wrangle.ipynb -- wrangle Pauli experiment data
- haarData_wrangle.ipynb -- wrangles Haar convergence experiment data
- haarDelayData_wrangle.ipynb -- wrangles Haar delay experiment data
- average_T1_and_T2.ipynb -- wrangles all machine calibration text dumps into an average T1 and T2 across all experiments for each device
as well as a directory for each device

`armonkDataAnaysis` file summary (similar for bogota and jakarta)
- These device specific directories contain the experiment submission sripts, raw data, machine calibration files, and processed data. For example purposes, we will discuss the Armonk folder structure, but the other folders follow the same patterns. 
Within this directory, there are three sub-directories whose structure is otherwise self-contained 
- PauliData -- contains Pauli experiment submission script `job_submit.ipynb`, data, analysis, and plots put in paper
- HaarData -- contains Haar convergence experiment submission script, data, analysis, and plots put in paper
- HaarDelayData -- contains Haar delay experiment submission script, data, analysis, and plots put in paper

`methodologyDataAnalysis` file summary (justification of methods via Appendix C)
- `armonk_methodology_part1.nb` -- analysis and graph generation of subsections 1 -- 5 in appendix C
- `armonk_methodology_part2.nb` --  analysis and graph generation of subsections 6 onward in appendix C
- `methodology_figures/` -- collection of all methodology figures used in Appendix C

### Bottom line summary
We will discuss what each important file/ directory in the above PauliData, HaarData, and HaarDelayData is. But the bottom line is 
- edd_pub_data/Armonk/PauliData/csvPauliData contains the Pauli experiment fidelity decay data in CSV format: `T, empirical Uhlmann fidelity, 2 sigma bootstrapped error on fidelity`
- edd_pub_data/Armonk/PauliData/armonkPauliAnalysis.nb is the Mathematica script we used to generate publication plots
- edd_pub_data/Armonk/HaarDelayData/csvHaarFreeData contains the Haar delay experiment data in CSV format: `d, empirical Uhlmann fidelity, 2 sigma bootstrapped error on fidelity`
- edd_pub_data/Armonk/HaarDelayData/ArmonkHaarDelayDataAnalysis.nb is the Mathematica script we used to generate publication plots 


#### The PauliData folder
The PauliData folder consists of
- job_submit.ipynb -- job submission script
- rawPauliData/ -- folder containing raw Pauli experiment data along with machine calibration files
- csvPauliData/ -- folder containing summary statistics of Pauli data as CSV files with three columns: T, fid, 2 \sigma bootstrapped err. Files are descriptively named like `fidDecay_backend_armonk_qubit_0_basis_g_goodc_False_T1_142.83_T2_236.68_dtype_min_offset_0.0_seq_qdd-2-7_sym_False_pode_3_date_2021-09-23 23:59:24-07:00.csv`
- armonkPauliAnalysis.nb -- Mathematica script to visualze data/ make plots
- armonkPauliAnalysis_figures/ -- folder containing plots we made several of which appear in the paper
- armonkPauliDataDict -- folder containing a compressed, re-loadable (think Python pickle of a dictionary or JSON file) Association containing Armonk Pauli data for loading into armonkPauliAnalysis.nb
- armonkPauliIntDict -- same as DataDict just above but with Hermite polynomial interpolations of each fidelity decay

### The HaarDelayData and HaarDelay folders
The structure of these two folders is very similar to the PauliData folder, so we do not re-iterate things here. 

## (iv) Details for experts/ gotchas for those who want to reproduce results
If you are an IBM expert or someone interested in reproducing our results down to a T, then you will want to be aware of a few specific details regarding our code which we discuss now.

#### The choice of "pulse basis"
The "traditional way" (funny to say this) to think about using NISQ devices is in terms of unitary logic gates in a quantum circuit. Note, however, that one will be unable to reproduce all our results by using the `QuantumCircuit` class alone. The empirical difference was discussed in "Appendix B: Circuit vs OpenPulse APIs" in our paper, so please check this out first. Another way to see the difference is to define a common sequence like XY4 with our method and then compare it to the standard circuit derived method: 
<img width="1037" alt="Screen Shot 2022-07-07 at 3 23 57 PM" src="https://user-images.githubusercontent.com/29308150/177881422-3d077ba1-ca3b-4935-b9e9-21d142dafdc3.png">

The "g_basis" is a custom basis choice whereas "c_basis" corresponds to the process of defining XY4 with the QuantumCircuit API, transpiling it into native gates, and using the qiskit `build_schedule` function to construct the corresponding `Schedule` object. In other words, the "c_basis" is the following:  
<img width="1043" alt="Screen Shot 2022-07-07 at 3 32 21 PM" src="https://user-images.githubusercontent.com/29308150/177882309-262ddfc8-84c7-4e21-9332-bdb17380c951.png">
which again yields Y in terms of X and virtual Z. 

The code to generate the "g_basis" is contained in `src/edd/pulse/_ibmq_pulse.py` in the function `create_from_greg_basis`. First, we define `x` pulses in exactly the same way as they are defined as an X gate. But rather than define Y in terms of X and virtual Y, we define it from the Y used in the now deprecated `u2` pulse,
```
y90 = defaults.get('u2', qubit, P0=0, P1=0).instructions[index][1].pulse
dur = y90.parameters['duration']
amp = y90.parameters['amp']
sigma = y90.parameters['sigma']
beta = y90.parameters['beta']
y180 = Drag(dur, 2*amp, sigma, beta, name='Y')
```
where Drag is a type of pulse specification defined in . In addition, we can define rotations about axis between X and Y by simply rotating the amplitude fo the Drag pulse, i.e.
```
def rotate(complex_signal: np.ndarray, delta_phi: float) -> np.ndarray:
    '''Adds a phase to a complex signal.'''
    phi = np.angle(complex_signal) + delta_phi
    return np.abs(complex_signal) * np.exp(1j*phi)
    
x30 = Drag(dur, rotate(amp, np.pi/6), sigma, beta, name='X30')
x120 = Drag(dur, rotate(amp, (2*np.pi)/3), sigma, beta, name='X120')
```
where `x30` is a `Pi` pulse rotating about the axis 30 degrees from the X axis counter-clockwise. These rotated pulses are important in defining the `UR`, for example. 

This may seem like an unusual choice, but we justify it in the aforementioned appendix. 


#### A delay instruction gotchI
If you are an NISQ experimentalist, you will likely be familiar with a so-called "acquire constraint." To explain this in IBM terms, let's first introduce the notion of IBM's `dt`. Basically, `dt` is the smallest unit of time for which a pulse wave-form can be defined. In other worse, an X pulse--in an digital to analog kind of way--consists of an ampltiude value at `t = 0, dt, 2 dt, ..., 160dt`, as it's width is `160dt`. In the above pulse plots, for example, the total evolution time was listed in this `dt` and was `160 * 4 = 640 dt`. One can acquire `dt` in nano-seconds for their given backend using
```
# evaluates to 0.222222 currently
dt = backend.get_dt("ns")
```
Now, even though pulses are defined in increments in `dt`, the total sum of all wave-forms must actually be a multiple of `16dt` which is known as the acquire constraint. To query what the acquire constraint while using the device, use
```
### evaluates to 16 right now
backend.get_acquire_alignment()
```

As far as I can tell, this is not a well documented constraint of devices, as using `OpenPulse` is still fairly new. This has important implications for how to run DD experiments when varying the pulse interval. In short, the total interval delays (the free evolution periods between pulses) must also be a mutltiple of 16. Not respecting this constraint will lead to vary biare results as we described in a note in an early draft (but redacted for being too int he weeds) which may help clarify using equations/ empirical results: 
<img width="435" alt="Screen Shot 2022-07-07 at 3 55 40 PM" src="https://user-images.githubusercontent.com/29308150/177884776-e63ae822-31d6-41b3-bfcd-c0581e1a8c77.png">
