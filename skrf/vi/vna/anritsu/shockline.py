from __future__ import annotations

import itertools
import re
import sys
import time
import types
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterable,
    Sequence,
)

import numpy as np
import pyvisa.util as util

import skrf
from skrf.vi import vna
from skrf.vi.validators import BooleanValidator, EnumValidator, FloatValidator, FreqValidator, IntValidator
from skrf.vi.vna import VNA, ValuesFormat


class SCPIError(Exception):
    def __init__(self, description: str) -> None:
        super().__init__(
            f'ShockLine error: {description}'
        )
        self.description = description

class SweepType(Enum):
    LINEAR = 'LIN'
    LOG = 'LOG'
    POWER = 'POW'
    CW = 'CW'
    SEGMENT = 'SEGM'
    PHASE = 'PHAS'

class SweepMode(Enum):
    HOLD = 'HOLD'
    CONTINUOUS = 'CONT'
    SINGLE = 'SING'

class TriggerSource(Enum):
    INTERNAL = 'INT'
    EXTERNAL = 'EXT'

class AveragingMode(Enum):
    POINT = 'POIN'
    SWEEP = 'SWE'

class DataFormat(Enum):
    ASCII = 'ASC'
    DOUBLE = 'REAL'
    SINGLE = 'REAL32'

def _read_ascii_values(
    self,
    converter: util.ASCII_CONVERTER = 'f',
    separator: str | Callable[[str], Iterable[str]] = lambda s: re.split(r'\n|,', s),
    container: type | Callable[[Iterable], Sequence] = list,
) -> Sequence[Any]:
    '''Anritsu-style read_ascii_values
    '''
    try:
        mark = self.read_bytes(1).decode()
        if mark != r'#':
            raise SCPIError(f'Unexpected header mark: {mark}')
        digit = self.read_bytes(1).decode()
        if not digit.isdigit():
            raise SCPIError(f'Unexpected header digit: {digit}')
        count = self.read_bytes(int(digit)).decode()
        if not count.isdigit():
            raise SCPIError(f'Unexpected header count: {count}')
        block = self.read_bytes(int(count)).decode()
        _ = self.read()
    finally:
        self.clear()
    return util.from_ascii_block(block, converter, separator, container)

def _query_ascii_values(
        self,
        message: str,
        converter: util.ASCII_CONVERTER = "f",
        separator: str | Callable[[str], Iterable[str]] = lambda s: re.split(r'\n|,', s),
        container: type | Callable[[Iterable], Sequence] = list,
        delay: float | None = None,
) -> Sequence[Any]:
    '''Anritsu-style query_ascii_values
    '''
    self.write(message)
    if delay is None:
        delay = self.query_delay
    if delay > 0.0:
        time.sleep(delay)
    return self.read_ascii_values(converter, separator, container)

class ShockLine(VNA):
    '''
    Anritsu ShockLine.

    Shockline Models
    ==========
    MS46524B

    '''
    # TODO:
    # - active_calset_name: SENS<self:cnum>:CORR:CSET:ACT? NAME
    # - create_calset: SENS<self:cnum>:CORR:CSET:CRE <name>
    # - calset_data: SENS<self:cnum>:CORR:CSET:DATA?  <eterm>,<port a>,<port b> '<receiver>'


    _models = {
        'default': {'nports': 4, 'unsupported': []},
        'MS46524B': {'nports': 4, 'unsupported': ['nports']},
    }

    class Channel(vna.Channel):
        def __init__(self, parent, cnum: int, cname: str):
            super().__init__(parent, cnum, cname)


        freq_start = VNA.command(
            get_cmd='SENS<self:cnum>:FREQ:STAR?',
            set_cmd='SENS<self:cnum>:FREQ:STAR <arg>',
            doc='''The start frequency [Hz]''',
            validator=FreqValidator(),
        )

        freq_stop = VNA.command(
            get_cmd='SENS<self:cnum>:FREQ:STOP?',
            set_cmd='SENS<self:cnum>:FREQ:STOP <arg>',
            doc='''The stop frequency [Hz]''',
            validator=FreqValidator(),
        )

        freq_span = VNA.command(
            get_cmd='SENS<self:cnum>:FREQ:SPAN?',
            set_cmd='SENS<self:cnum>:FREQ:SPAN <arg>',
            doc='''The frequency span [Hz].''',
            validator=FreqValidator(),
        )

        freq_center = VNA.command(
            get_cmd='SENS<self:cnum>:FREQ:CENT?',
            set_cmd='SENS<self:cnum>:FREQ:CENT <arg>',
            doc='''The frequency span [Hz].''',
            validator=FreqValidator(),
        )

        npoints = VNA.command(
            get_cmd='SENS<self:cnum>:SWE:POIN?',
            set_cmd='SENS<self:cnum>:SWE:POIN <arg>',
            doc='''The number of frequency points. Sets the frequency step as a
                side effect
            ''',
            validator=IntValidator(),
        )

        if_bandwidth = VNA.command(
            get_cmd='SENS<self:cnum>:BWID?',
            set_cmd='SENS<self:cnum>:BWID <arg>',
            doc='''The IF bandwidth [Hz]''',
            validator=FreqValidator(),
        )

        sweep_time = VNA.command(
            get_cmd='SENS<self:cnum>:SWE:TIME?',
            set_cmd='SENS<self:cnum>:SWE:TIME <arg>',
            doc='''The time in seconds for a single sweep [s]''',
            validator=FloatValidator(decimal_places=2),
        )

        sweep_type = VNA.command(
            get_cmd='SENS<self:cnum>:SWE:TYPE?',
            set_cmd='SENS<self:cnum>:SWE:TYPE <arg>',
            doc='''The type of sweep (linear, log, etc)''',
            validator=EnumValidator(SweepType),
        )

        sweep_mode = VNA.command(
            get_cmd='SENS<self:cnum>:HOLD:FUNC?',
            set_cmd='SENS<self:cnum>:HOLD:FUNC <arg>',
            doc='''This channel's trigger mode''',
            validator=EnumValidator(SweepMode),
        )

        averaging_on = VNA.command(
            get_cmd='SENS<self:cnum>:AVER:STATE?',
            set_cmd='SENS<self:cnum>:AVER:STATE <arg>',
            doc='''Whether averaging is on or off''',
            validator=BooleanValidator(),
        )

        averaging_count = VNA.command(
            get_cmd='SENS<self:cnum>:AVER:COUN?',
            set_cmd='SENS<self:cnum>:AVER:COUN <arg>',
            doc='''The number of measurements combined for an average''',
            validator=IntValidator(1, 65536),
        )

        averaging_mode = VNA.command(
            get_cmd='SENS<self:cnum>:AVER:TYPE?',
            set_cmd='SENS<self:cnum>:AVER:TYPE <arg>',
            doc='''How measurements are averaged together''',
            validator=EnumValidator(AveragingMode),
        )

        ntraces = VNA.command(
            get_cmd='CALC<self:cnum>:PAR:COUNT?',
            set_cmd='CALC<self:cnum>:PAR:COUNT <arg>',
            doc='''The number of traces.
            ''',
            validator=IntValidator(),
        )

        @property
        def traces(self) -> Sequence[str]:
            return [self.query(f':CALC{self.cnum}:PAR{t}:DEF?')
                    for t in range(1,self.ntraces+1)]

        @traces.setter
        def traces(self, trs: Sequence[str]) -> None:
            self.ntraces = len(trs)
            for t, p in enumerate(trs,start=1):
                self.write(f':CALC{self.cnum}:PAR{t}:DEF {p}')

        @property
        def stores(self) -> Sequence[str]:
            return [self.query(f':CALC{self.cnum}:PAR{t}:MEM:MML?')
                    for t in range(1,self.ntraces+1)]

        @stores.setter
        def stores(self, trs: Sequence[str]) -> None:
            for t, p in enumerate(trs,start=1):
                self.write(f':CALC{self.cnum}:PAR{t}:MEM:MML {p}')

        @property
        def freq_step(self) -> int:
            # Not all instruments support SENS:FREQ:STEP
            f = self.frequency
            return int(f.step)

        @freq_step.setter
        def freq_step(self, f: int | float | str) -> None:
            validator = FreqValidator()
            f = validator.validate_input(f)
            freq = self.frequency
            self.npoints = len(range(int(freq.start), int(freq.stop) + f, f))

        @property
        def frequency(self) -> skrf.Frequency:
            f = skrf.Frequency(
                start=self.freq_start,
                stop=self.freq_stop,
                npoints=self.npoints,
                unit='hz',
            )
            return f

        @frequency.setter
        def frequency(self, f: skrf.Frequency) -> None:
            self.freq_start = f.start
            self.freq_stop = f.stop
            self.npoints = f.npoints

        def sweep(self) -> None:
            config = {
                'sweep_mode': self.sweep_mode,
                'timeout': self.parent._resource.timeout,
            }
            self.parent._resource.clear()
            self.sweep_mode = SweepMode.SINGLE
            while self.sweep_mode == SweepMode.SINGLE:
                time.sleep(0.1)
            self.parent._resource.clear()
            self.parent._resource.timeout = config.pop('timeout')
            for k, v in config.items():
                setattr(self, k, v)

        def get_sdata(self, a: int | str, b: int | str) -> skrf.Network:
            # Get trace
            param = f'S{a}{b}'
            tnum = 1+self.traces.index(param)
            # Acquire data
            self.sweep_mode = SweepMode.SINGLE
            while self.sweep_mode == SweepMode.SINGLE:
                time.sleep(0.1)
            self.parent.check_errors()
            # Read data
            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            sdata = self.query_values(f':CALC{self.cnum}:PAR{tnum}:DATA:SDATA?', complex_values=True)
            self.parent.query_format = orig_query_fmt
            self.parent.check_errors()
            # Build network
            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = sdata
            return ntwk

        def get_snp_network(
            self,
            ports: Sequence | None = None,
        ) -> skrf.Network:
            if ports is None:
                ports = list(range(1, self.parent.nports + 1))
            # Set up traces
            orig_traces = self.traces
            self.traces = [f"S{b}{a}" for a, b in itertools.product(ports, repeat=2)]
            # Acquire data
            self.sweep_mode = SweepMode.SINGLE
            while self.sweep_mode == SweepMode.SINGLE:
                time.sleep(0.1)
            self.parent.check_errors()
            # Read data
            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            sdata = [self.query_values(f':CALC{self.cnum}:PAR{tnum}:DATA:SDATA?', complex_values=True)
                     for tnum in range(1, self.ntraces + 1)]
            self.parent.query_format = orig_query_fmt
            self.traces = orig_traces
            self.parent.check_errors()
            # Build network
            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = np.stack(sdata,axis=1).reshape(-1,len(ports),len(ports))
            return ntwk

    def __init__(self, address: str, backend: str = '@py') -> None:
        super().__init__(address, backend)

        self._resource.read_termination = '\n'
        self._resource.write_termination = '\n'

        self.create_channel(1, 'Channel 1')

        self.model = self.id.split(',')[1]
        if self.model not in self._models:
            print(
                f'WARNING: This model ({self.model}) has not been tested with '
                'scikit-rf.',
                file=sys.stderr,
            )

    trigger_source = VNA.command(
        get_cmd='TRIG:SOUR?',
        set_cmd='TRIG:SOUR <arg>',
        doc='''The source of the sweep trigger signal''',
        validator=EnumValidator(TriggerSource),
    )

    nerrors = VNA.command(
        get_cmd='SYST:ERR:COUN?',
        set_cmd=None,
        doc='''The number of errors since last cleared (see
            :func:`PNA.clear_errors`)''',
        validator=IntValidator(),
    )

    @property
    def query_format(self) -> vna.ValuesFormat:
        fmt = self.query(':FORM:DATA?')
        if fmt == 'ASC':
            self._values_fmt = vna.ValuesFormat.ASCII
        elif fmt == 'REAL32':
            self._values_fmt = vna.ValuesFormat.BINARY_32
        elif fmt == 'REAL':
            self._values_fmt = vna.ValuesFormat.BINARY_64
        return self._values_fmt

    @query_format.setter
    def query_format(self, fmt: vna.ValuesFormat) -> None:
        if fmt == vna.ValuesFormat.ASCII:
            self._values_fmt = vna.ValuesFormat.ASCII
            self.write(':FORM:DATA ASC')
        elif fmt == vna.ValuesFormat.BINARY_32:
            self._values_fmt = vna.ValuesFormat.BINARY_32
            self.write(':FORM:BORD SWAP')
            self.write(':FORM:DATA REAL32')
        elif fmt == vna.ValuesFormat.BINARY_64:
            self._values_fmt = vna.ValuesFormat.BINARY_64
            self.write(':FORM:BORD SWAP')
            self.write(':FORM:DATA REAL')

    @property
    def nports(self) -> int:
        return int(self.query('SYST:PORT:COUN?'))

    def read_values(self, **kwargs) -> None:  # noqa: B027
        pass

    def _setup_scpi(self) -> None:
        self.__class__.wait_for_complete = lambda self: self.query('*OPC?')
        self.__class__.status = property(lambda self: self.query('*STB?'))
        self.__class__.options = property(lambda self: self.query('*OPT?'))
        self.__class__.id = property(lambda self: self.query('*IDN?'))
        self.__class__.clear_errors = lambda self: self.write('*CLS')
        # ShockLine sends propietary strings
        def errcheck(self) -> None:
            err = self.query('SYST:ERR?')
            if err == 'No Error':
                return
            else:
                raise SCPIError(err)
        self.__class__.check_errors = errcheck
        # ShockLine uses ASCII headers: #<digit><decimal>
        self._resource.read_ascii_values = types.MethodType(_read_ascii_values, self._resource)
        self._resource.query_ascii_values = types.MethodType(_query_ascii_values, self._resource)

