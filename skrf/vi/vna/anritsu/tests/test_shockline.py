import pyvisa
rm = pyvisa.ResourceManager('@py')
rm.list_resources()
tst = rm.open_resource('TCPIP::10.255.79.73::INSTR')

from skrf.vi.vna.anritsu import shockline

import importlib
importlib.reload(shockline)

vna = shockline.ShockLine('TCPIP::anritsu-vna.local::5001::SOCKET')


