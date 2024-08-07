
import sys

import pytest

import skrf

try:
    from skrf.vi.vna import anritsu
    from skrf.vi.vna.anritsu.shockline import SweepMode, SweepType
except ImportError:
    pass

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

@pytest.fixture
def mocked_ff(mocker):
    mocker.patch('skrf.vi.vna.anritsu.ShockLine.__init__', return_value=None)
    mocker.patch('skrf.vi.vna.anritsu.ShockLine.write')
    mocker.patch('skrf.vi.vna.anritsu.ShockLine.write_values')
    mocker.patch('skrf.vi.vna.anritsu.ShockLine.query')
    mocker.patch('skrf.vi.vna.anritsu.ShockLine.query_values')
    mock = anritsu.ShockLine('TEST')
    mock.model = "TEST"

    # This gets done in init, but we are mocking init to prevent super().__init__, so just call here
    mock.create_channel(1, 'Channel 1')

    yield mock

def test_frequency_query(mocker, mocked_ff):
    mocked_ff.query.side_effect = [
        '100', '200', '11'
    ]
    test = mocked_ff.ch1.frequency
    assert test == skrf.Frequency(100, 200, 11, unit='hz')

def test_frequency_write(mocker, mocked_ff):
    test_f = skrf.Frequency(100, 200, 11, unit='hz')
    mocked_ff.ch1.frequency = test_f
    calls = [
        mocker.call("SENS1:FREQ:STAR 100"),
        mocker.call("SENS1:FREQ:STOP 200"),
        mocker.call("SENS1:SWE:POIN 11"),
    ]
    mocked_ff.write.assert_has_calls(calls)

# def test_create_channel(mocker, mocked_ff):
    # mocked_ff.create_channel(2, 'Channel 2')
    # assert hasattr(mocked_ff, 'ch2')
    # assert mocked_ff.ch2.cnum == 2
    # assert mocked_ff.ch2.name == "Channel 2"

def test_active_channel_query(mocker, mocked_ff):
    mocked_ff.query.return_value = 1
    test = mocked_ff.active_channel
    assert isinstance(test, anritsu.ShockLine.Channel)
    assert test.cnum == 1

def test_clear_averaging(mocker, mocked_ff):
    mocked_ff.ch1.clear_averaging()
    mocked_ff.write.assert_called_once_with('SENS1:AVER:CLE')


