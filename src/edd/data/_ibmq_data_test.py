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

import os.path

from edd.data._ibmq_data import IBMQData
from edd.backend import IBMQBackend
from edd.circuit import IBMQDdCircuit
from edd.experiments import theta_sweep_frev

import unittest

class IBMQDataTest(unittest.TestCase):
    """ Tests that IBMQData class works smoothly. """

    # First, we test if we can initialize with no input data
    def test_init_no_input(self):
        test_data = IBMQData()
        self.assertIs(test_data.data, None)
        self.assertEqual(test_data.name, 'test')

    def test_init_with_result(self):
        ourense = IBMQBackend('ibmq_ourense')
        # build up some arbitrary simple circuit
        circ = IBMQDdCircuit(5)
        circ.x(0)
        circ.measure_all()
        # submit test job to get "Results" class object
        results = ourense.submit_test(circ, 'test')
        # init IBMQData with Results object
        test_data = IBMQData(results, 'x_gate_data')
        self.assertDictEqual(test_data.data, results.to_dict())
        self.assertEqual(test_data.name, 'x_gate_data')

    def test_init_with_dict_result(self):
        ourense = IBMQBackend('ibmq_ourense')
        # build up some arbitrary simple circuit
        circ = IBMQDdCircuit(5)
        circ.x(0)
        circ.measure_all()
        # submit test job to get "Results" class object
        results = ourense.submit_test(circ, 'test')
        # init IBMQData with dictionary data
        test_data = IBMQData(results.to_dict(), 'x_gate_data')
        self.assertDictEqual(test_data.data, results.to_dict())
        self.assertEqual(test_data.name, 'x_gate_data')

    def test_add_data(self):
        ourense = IBMQBackend('ibmq_ourense')
        # build up some arbitrary simple circuit
        circ = IBMQDdCircuit(5)
        circ.x(0)
        circ.measure_all()
        # submit test job to get "Results" class object
        results = ourense.submit_test(circ, 'test')
        # init IBMQData with Result object
        test_data = IBMQData(results, 'x_gate_data')
        # add same data set a second time as Results object
        test_data.add_data(results)
        self.assertIsInstance(test_data.data, list)
        self.assertDictEqual(results.to_dict(), test_data.data[0])
        self.assertDictEqual(results.to_dict(), test_data.data[1])
        # add data a third time as a dict
        test_data.add_data(results.to_dict())
        self.assertIsInstance(test_data.data, list)
        self.assertDictEqual(results.to_dict(), test_data.data[0])
        self.assertDictEqual(results.to_dict(), test_data.data[1])
        self.assertDictEqual(results.to_dict(), test_data.data[2])

    def test_save_data(self):
        ourense = IBMQBackend('ibmq_ourense')
        # build up some arbitrary simple circuit
        circ = IBMQDdCircuit(5)
        circ.x(0)
        circ.measure_all()
        # submit test job to get "Results" class object
        results = ourense.submit_test(circ, 'test')
        # init IBMQData with Results object
        test_data = IBMQData(results, 'x_gate_data')
        # save the data--this saves as .yml file
        fname = 'test_x_gate_data.yml'
        test_data.save_data(fname)
        # check if file exists now
        file_exists = os.path.exists(fname)

        self.assertIs(file_exists, True)

    def test_load_data(self):
        ourense = IBMQBackend('ibmq_ourense')
        # build up some arbitrary simple circuit
        circ = IBMQDdCircuit(5)
        circ.x(0)
        circ.measure_all()
        # submit test job to get "Results" class object
        results = ourense.submit_test(circ, 'test')
        # init IBMQData with Result object
        test_data = IBMQData(results, 'x_gate_data')
        # save the data
        fname = 'test_x_gate_data.yml'
        test_data.save_data(fname)
        
        # load the data and test whether it is added
        test_data.load_data(fname)
        self.assertIsInstance(test_data.data, list)
        self.assertDictEqual(results.to_dict(), test_data.data[0])
        self.assertDictEqual(results.to_dict(), test_data.data[1])
        # load data again to see if added again
        test_data.load_data(fname)
        self.assertIsInstance(test_data.data, list)
        self.assertDictEqual(results.to_dict(), test_data.data[0])
        self.assertDictEqual(results.to_dict(), test_data.data[1])
        self.assertDictEqual(results.to_dict(), test_data.data[2])

    def test_full_wrangle_theta_sweep(self):
        ourense = IBMQBackend('ibmq_ourense')
        # run a quick theta sweep and use to init data set
        results = theta_sweep_frev(ourense, 5, [0.209, .418])
        test_data = IBMQData(results, 'theta_sweep_test')
        # perform theta_sweep full analysis (which tests all parts)
        (data, fname, plot) = test_data.full_wrangle_theta_sweep()

        # assertions for data
        self.assertIsInstance(data[0][0], str)
        self.assertIsInstance(data[0][1], str)
        self.assertIsInstance(data[1], tuple)
        self.assertIsInstance(data[1][0][0], float)
        self.assertIsInstance(data[1][1][0], float)
        self.assertIsInstance(data[1][2][0], float)
        self.assertIsInstance(data[1][3][0], float)
        self.assertIsInstance(data[1][4][0], float)

        # assertions for fname
        self.assertIsInstance(fname, str)
        file_exists = os.path.exists(fname)
        self.assertIs(file_exists, True)

        # assertions for plt
        pltname = fname[0:-3] + 'png'
        file_exists = os.path.exists(pltname)
        self.assertIs(file_exists, True)

if __name__=="__main__":
    unittest.main()
