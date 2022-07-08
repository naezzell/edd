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

"""
typically this is a bad import, but I think since it keeps
'unpacked' namespace local to this package, it's okay to import
all the 'utility' experiment functions in this way
"""
#TODO: Figure out if the above statement is actually true
import edd.experiments._circuit_experiments as circ
import edd.experiments._pulse_experiments as pulse
