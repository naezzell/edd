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

import random
import numpy as np
from edd.backend import IBMQBackend
from edd.pulse import IBMQDdSchedule

# type check imports
from qiskit.pulse import Schedule
from qiskit.pulse import Play
from qiskit.pulse import Delay
from matplotlib.figure import Figure

import unittest

class IBMQScheduleTest(unittest.TestCase):
    """ Tests that IBMQPulse class works smoothly. """

    ##################################################
    # Basic Tests
    ##################################################

    def test_init(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        # check that member data is init properly
        self.assertIsInstance(sched.ibmq_backend, IBMQBackend)
        self.assertIsInstance(sched.basis, dict)
        self.assertIsInstance(sched.dt, float)
        self.assertIsInstance(sched.sched, Schedule)

    def test_get_schedule(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        # check that schedule is actually a Schedule object
        self.assertIsInstance(sched.get_schedule(), Schedule)

    def test_get_pulse_list(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        # check that it returns a list
        pulse_list = sched.get_pulse_list()
        self.assertIsInstance(pulse_list, list)
        # add an Play element to pulse and check type
        sched.add_id(0)
        pulse_list = sched.get_pulse_list()
        self.assertIsInstance(pulse_list[0][1], Play)
        # add Delay element to pulse and check type
        sched.add_pause(0, 500)
        pulse_list = sched.get_pulse_list()
        self.assertIsInstance(pulse_list[1][1], Delay)

    def test_get_pulse_names(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_x(0)
        pulse_names = sched.get_pulse_names()
        self.assertIsInstance(pulse_names[0], str)

    def test_get_duration(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_free(0, 500)
        durr = sched.get_duration()
        self.assertIsInstance(durr, int)

    def test_get_phys_time(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_free(0, 500)
        time = sched.get_phys_time()
        self.assertIsInstance(time, float)

    def test_draw(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        fig = sched.draw()
        self.assertIsInstance(fig, Figure)

    def test_reset(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_free(0, 500)
        sched.reset()
        durr = sched.get_duration()
        self.assertEquals(durr, 0)

    ##################################################
    # Simple Pulse Method Tests
    ##################################################
    def test_add_id(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_id(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'I')

    def test_add_x(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_x(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'X')

    def test_add_xb(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_xb(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'Xb')

    def test_add_x90(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_x90(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'X90')

    def test_add_x90b(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_x90b(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'X90b')

    def test_add_y(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_y(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'Y')

    def test_add_yb(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_yb(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'Yb')

    def test_add_y90(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_y90(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'Y90')

    def test_add_y90b(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_y90b(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'Y90b')

    def test_add_measurement(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_measurement(0)
        pulse_name = sched.get_pulse_names()[0]
        self.assertEquals(pulse_name, 'M_m0')

    def test_add_u3(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_u3(0, 1, 1, 1)
        pulse_name1 = sched.get_pulse_names()[0]
        pulse_name2 = sched.get_pulse_names()[1]
        self.assertEquals(pulse_name1, 'X90p_d0')
        self.assertEquals(pulse_name2, 'X90m_d0')

    ##################################################
    # Test DD Sequence Methods
    ##################################################
    def test_add_pause(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_pause(0, 500)
        pulse = sched.get_pulse_list()[0][1]
        self.assertIsInstance(pulse, Delay)
        self.assertEquals(sched.get_duration(), 500)

    def test_add_free(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        sched.add_free(0, 5)
        pulse_names = sched.get_pulse_names()
        pulse_order = ['I' for x in range(5)]
        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_hahn(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_hahn(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = [None, 'X', None]
            pulse_order.extend(rep)
        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_cpmg(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_cpmg(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = [None, 'X', None, 'X', None]
            pulse_order.extend(rep)
        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_xy4(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_xy4(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['Y', None, 'X', None, 'Y', None, 'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_xy4_s(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_xy4_s(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()
        # construct manual pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = [None, 'Y', None, 'X', None, 'Y', None, 'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_cdd_n(self):
        armonk = IBMQBackend('ibmq_armonk')
        n = 1
        tau = 100
        # check that n = 1 works
        cdd_1_sched = IBMQDdSchedule(armonk)
        cdd_1_sched.add_cdd_n(0, n, tau)
        pulse_names = cdd_1_sched.get_pulse_names()
        # construct manual pulse order list
        pulse_order = ['Y', None, 'X', None, 'Y', None, 'X', None]
        self.assertSequenceEqual(pulse_names, pulse_order)

        # check that n = 2 works
        n = 2
        cdd_2_sched = IBMQDdSchedule(armonk)
        cdd_2_sched.add_cdd_n(0, n, tau)
        pulse_names = cdd_2_sched.get_pulse_names()
        # construct expected pulse order list
        pulse_order2 = []
        pulse_order2.extend(['Y', None])
        pulse_order2.extend(pulse_order)
        pulse_order2.extend(['X', None])
        pulse_order2.extend(pulse_order)
        pulse_order2.extend(['Y', None])
        pulse_order2.extend(pulse_order)
        pulse_order2.extend(['X', None])
        pulse_order2.extend(pulse_order)
        self.assertSequenceEqual(pulse_names, pulse_order2)

    def test_add_rga2x(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga2x(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['Xb', None, 'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga2y(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga2y(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['Yb', None, 'Y', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga4(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga4(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['Yb', None, 'X', None, 'Yb', None, 'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga4p(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga4p(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['Yb', None, 'Xb', None, 'Yb', None,\
                   'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga8a(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga8a(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['I', None, 'Xb', None, 'Y', None,\
                   'Xb', None, 'I', None, 'X', None,\
                   'Yb', None, 'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga8c(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga8c(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        pulse_order = []
        for _ in range(num_reps):
            rep = ['X', None, 'Y', None, 'X', None,\
                   'Y', None, 'Y', None, 'X', None,\
                   'Y', None, 'X', None]
            pulse_order.extend(rep)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga16b(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga16b(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        rga4p = IBMQDdSchedule(armonk)
        rga4p.add_rga4p(0, 1, tau)
        rga4p_pulses = rga4p.get_pulse_names()
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga4p_pulses)
            pulse_order.extend(['Xb', None])
            pulse_order.extend(rga4p_pulses)
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga4p_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga4p_pulses)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga32a(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga32a(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        rga8a = IBMQDdSchedule(armonk)
        rga8a.add_rga8a(0, 1, tau)
        rga8a_pulses = rga8a.get_pulse_names()
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8a_pulses)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga32c(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga32c(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        rga4 = IBMQDdSchedule(armonk)
        rga4.add_rga4(0, 1, tau)
        rga4_pulses = rga4.get_pulse_names()
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(['X', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga4_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga4_pulses)

        self.assertSequenceEqual(pulse_names, pulse_order)


    def test_add_rga64a(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga64a(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        rga8a = IBMQDdSchedule(armonk)
        rga8a.add_rga8a(0, 1, tau)
        rga8a_pulses = rga8a.get_pulse_names()
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(['I', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['Xb', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['Xb', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['I', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga8a_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8a_pulses)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga64c(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga64c(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        rga8c = IBMQDdSchedule(armonk)
        rga8c.add_rga8c(0, 1, tau)
        rga8c_pulses = rga8c.get_pulse_names()
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['Y', None])
            pulse_order.extend(rga8c_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga8c_pulses)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_rga256a(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_rga256a(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()

        # construct manually pulse order list
        rga64a = IBMQDdSchedule(armonk)
        rga64a.add_rga64a(0, 1, tau)
        rga64a_pulses = rga64a.get_pulse_names()
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga64a_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga64a_pulses)
            pulse_order.extend(['Yb', None])
            pulse_order.extend(rga64a_pulses)
            pulse_order.extend(['X', None])
            pulse_order.extend(rga64a_pulses)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_comp_kdd(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'x_basis', name='test')
        tau = 100
        # test the phi = 0 case first
        phi = 0
        sched.add_comp_kdd(0, phi, tau)
        pulse_names = sched.get_pulse_names()
        pulse_order = [None, 'X30', None, 'X', None, 'Y', None,
                       'X', None, 'X30', None]
        self.assertSequenceEqual(pulse_names, pulse_order)
        # test the phi = pi/2 case next
        phi = np.pi / 2
        sched.reset()
        sched.add_comp_kdd(0, phi, tau)
        pulse_names = sched.get_pulse_names()
        pulse_order = [None, 'X120', None, 'Y', None, 'Xb', None,
                       'Y', None, 'X120', None]
        self.assertSequenceEqual(pulse_names, pulse_order)
        # test a more general case with phi = random float
        phi = random.random()
        sched.reset()
        sched.add_comp_kdd(0, phi, tau)
        pulse_names = sched.get_pulse_names()
        # create manual pulse name schedule
        p1_ang = np.pi/6 + phi
        p1_name = f'[pi]_({p1_ang:.6f})'
        p2_ang = phi
        p2_name = f'[pi]_({p2_ang:.6f})'
        p3_ang = np.pi/2 + phi
        p3_name = f'[pi]_({p3_ang:.6f})'
        pulse_order = [None, p1_name, None, p2_name, None, p3_name, None,
                       p2_name, None, p1_name, None]
        self.assertSequenceEqual(pulse_names, pulse_order)


    def test_add_kdd(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'x_basis', name='test')
        num_reps = 5
        tau = 100
        sched.add_kdd(0, num_reps, tau)
        pulse_names = sched.get_pulse_names()
        # compare to expected answer
        x_pulse_order = [None, 'X30', None, 'X', None, 'Y', None,
                       'X', None, 'X30', None]
        y_pulse_order = [None, 'X120', None, 'Y', None, 'Xb', None,
                       'Y', None, 'X120', None]
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(y_pulse_order)
            pulse_order.extend(x_pulse_order)
            pulse_order.extend(y_pulse_order)
            pulse_order.extend(x_pulse_order)

        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_add_ur(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'x_basis', name='test')
        num_reps = 5
        tau = 100
        # try the n = 4 case first
        sched.add_ur(0, 4, num_reps, tau)
        pulse_names = sched.get_pulse_names()
        # compare to expected answer
        ur4_pulses = [None, 'X', None, 'Xb', None, 'Xb',
                      None, 'X', None]
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(ur4_pulses)
        self.assertSequenceEqual(pulse_names, pulse_order)

        # now try the n = 16 case
        sched.reset()
        sched.add_ur(0, 16, num_reps, tau)
        pulse_names = sched.get_pulse_names()
        p0 = 'X'
        p1 = f'[pi]_({(1*np.pi / 4):.6f})'
        p2 = f'[pi]_({(3*np.pi / 4):.6f})'
        p3 = f'Yb'
        p4 = f'Y'
        p5 = f'[pi]_({(7*np.pi / 4):.6f})'
        p6 = f'[pi]_({(5*np.pi / 4):.6f})'
        p7 = f'Xb'
        ur16_pulses = [None, p0, None, p1, None, p2, None, p3, None,
                       p4, None, p5, None, p6, None, p7, None,
                       p7, None, p6, None, p5, None, p4, None,
                       p3, None, p2, None, p1, None, p0, None]
        pulse_order = []
        for _ in range(num_reps):
            pulse_order.extend(ur16_pulses)
        self.assertSequenceEqual(pulse_names, pulse_order)

    def test_udd_x(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        n = 1
        T = 10000
        # first, let's just check that pulse order is correct
        times = sched.add_udd_x(0, n, T)
        pulse_names = sched.get_pulse_names()
        # compare to expected answer
        pulse_order = [None, 'X', None, 'X']
        self.assertSequenceEqual(pulse_names, pulse_order)
        # now check that times are correct
        x_width = sched.basis['X'].duration
        theory_times = np.array([5000, 10000]) - x_width
        array_equals = (theory_times == times).all()
        self.assertEquals(array_equals, True)

        # now try n = 2, an even number
        n = 2
        sched.reset()
        times = sched.add_udd_x(0, n, T)
        pulse_names = sched.get_pulse_names()
        pulse_order = [None, 'X', None, 'X', None]
        self.assertSequenceEqual(pulse_names, pulse_order)
        theory_times = np.array([2500, 7500]) - x_width
        array_equals = (theory_times == times).all()
        self.assertEquals(array_equals, True)

    def test_udd_y(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        n = 1
        T = 10000
        # first, let's just check that pulse order is correct
        times = sched.add_udd_y(0, n, T)
        pulse_names = sched.get_pulse_names()
        # compare to expected answer
        pulse_order = [None, 'Y', None, 'Y']
        self.assertSequenceEqual(pulse_names, pulse_order)
        # now check that times are correct
        y_width = sched.basis['Y'].duration
        theory_times = np.array([5000, 10000]) - y_width
        array_equals = (theory_times == times).all()
        self.assertEquals(array_equals, True)

        # now try n = 2, an even number
        n = 2
        sched.reset()
        times = sched.add_udd_y(0, n, T)
        pulse_names = sched.get_pulse_names()
        pulse_order = [None, 'Y', None, 'Y', None]
        self.assertSequenceEqual(pulse_names, pulse_order)
        theory_times = np.array([2500, 7500]) - y_width
        array_equals = (theory_times == times).all()
        self.assertEquals(array_equals, True)

    def test_qdd(self):
        armonk = IBMQBackend('ibmq_armonk')
        sched = IBMQDdSchedule(armonk, 'g_basis', name='test')
        # NOTE: inner order can never be odd or pulses overlap
        n = 1
        m = 2
        T = 100000
        # first, let's just check that pulse order is correct
        times = sched.add_qdd(0, n, m, T)
        pulse_names = sched.get_pulse_names()
        # compare to expected answer
        pulse_order = [None, 'X', None, 'X', None, 'Y', None,
                       'X', None, 'X', None, 'Y']
        self.assertSequenceEqual(pulse_names, pulse_order)
        # now check that times are correct
        x_width = sched.basis['X'].duration
        y_width = sched.basis['Y'].duration
        y_times = np.array([50000, 100000]) - y_width
        x_times = np.array([12500, 37500, 62500, 87500]) - x_width
        theory_times = []
        theory_times.append(x_times[0])
        theory_times.append(x_times[1])
        theory_times.append(y_times[0])
        theory_times.append(x_times[2])
        theory_times.append(x_times[3])
        theory_times.append(y_times[1])
        theory_times = np.array(theory_times)
        array_equals = (theory_times == times).all()
        self.assertEquals(array_equals, True)

if __name__=="__main__":
    unittest.main()
