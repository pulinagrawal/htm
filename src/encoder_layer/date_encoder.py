"""Date encoder 2.0 for HTM, ported to Python from the C++ implementation.

This module provides a DateEncoder class that encodes various temporal features
(season, day of week, weekend, custom days, holiday, time of day) into a Sparse
Distributed Representation (SDR) for use in Hierarchical Temporal Memory (HTM) systems.

The encoder uses ScalarEncoder instances for each enabled feature, concatenating
their outputs into a single SDR. The configuration is controlled via the
DateEncoderParameters dataclass.

Usage:
    params = DateEncoderParameters(
        season_width=10,
        day_of_week_width=5,
        weekend_width=3,
        holiday_width=4,
        time_of_day_width=6,
        custom_width=3,
        custom_days=["mon,wed,fri"],
        verbose=True,
    )
    encoder = DateEncoder(params)
    output = SDR(dimensions=[encoder.size])
    encoder.encode(datetime.now(), output)
    print("Output size:", output.size)
    print("Active indices:", output.get_sparse())
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, override

import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from encoder_layer.base_encoder import BaseEncoder



class DateEncoder(BaseEncoder[datetime | pd.Timestamp | time.struct_time | None]):
    """
    Python port of the HTM DateEncoder, using the existing ScalarEncoder + SDR.
    Encodes up to 6 attributes of a timestamp into one SDR:

      - season       (day-of-year)
      - dayOfWeek
      - weekend
      - customDays
      - holiday
      - timeOfDay
    """

    SEASON = 0
    DAYOFWEEK = 1
    WEEKEND = 2
    CUSTOM = 3
    HOLIDAY = 4
    TIMEOFDAY = 5

    def __init__(
        self, parameters: DateEncoderParameters, dimensions: List[int] | None = None
    ) -> None:
        """
        Initialize the DateEncoder with the given parameters.

        Args:
            parameters: DateEncoderParameters instance specifying encoding options.
            dimensions: Optional SDR dimensions (unused, for compatibility).

        Raises:
            ValueError: If custom_days is specified but empty, or if no widths are provided.
        """
        self._parameters = copy.deepcopy(parameters)
        """DateEncoderParameters: Configuration parameters for the encoder."""
        self._customDays: set[int] = set()
        """Set of integer day indices for custom days."""
        self._bucketMap: Dict[int, int] = {}
        """Mapping from feature index to bucket position."""
        self._buckets: List[float] = []
        """List of bucket values for each feature."""
        self._size: int = 0
        """Total number of bits DateEncoder."""
        self._rdse_used = parameters.rdse_used
        """Flag indicating if RDSE is used."""

        # Declare one encoder per feature
        self._season_encoder = None
        """Encoder for season (day of year)."""
        self._dayofweek_encoder = None
        """Encoder for day of week."""
        self._weekend_encoder = None
        """Encoder for weekend flag."""
        self._customdays_encoder = None
        """Encoder for custom day groups."""
        self._holiday_encoder = None
        """Encoder for holidays."""
        self._timeofday_encoder = None
        """Encoder for time of day."""

        self._initialize(self._parameters)

        super().__init__(dimensions, self._size)

    # ------------------------------------------------------------------ #
    # Initialization (mirrors C++ initialize())
    # ------------------------------------------------------------------ #

    def _initialize(self, parameters: DateEncoderParameters) -> None:
        """Configure scalar encoders according to the supplied parameters."""

        args = parameters
        size = 0
        self._bucketMap.clear()
        self._buckets.clear()

        # -------- Season --------
        if args.season_width != 0:
            if self._rdse_used:

                p = RDSEParameters(
                    size=10,
                    active_bits=args.season_width,
                    sparsity=0.0,
                    radius=args.season_radius,
                    resolution=0.0,
                    category=False,
                    seed=42,
                )
                self._season_encoder = RandomDistributedScalarEncoder(p)
            else:

                from encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
                p = ScalarEncoderParameters(
                    minimum=0,
                    maximum=366,
                    clip_input=False,
                    periodic=True,
                    category=False,
                    active_bits=args.season_width,
                    sparsity=0.0,
                    size=0,
                    radius=args.season_radius,
                    resolution=0.0,
                )
                self._season_encoder = ScalarEncoder(p)

            self._bucketMap[self.SEASON] = len(self._buckets)
            self._buckets.append(0.0)
            size += self._season_encoder.size

        # -------- Day of week --------
        if args.day_of_week_width != 0:
            if self._rdse_used:
                p = RDSEParameters(
                    size=100,
                    active_bits=args.day_of_week_width,
                    sparsity=0.0,
                    radius=args.day_of_week_radius,
                    resolution=0.0,
                    category=False,
                    seed=43,
                )
                self._dayofweek_encoder = RandomDistributedScalarEncoder(p)

            else:

                p = ScalarEncoderParameters(
                    minimum=0,
                    maximum=7,
                    clip_input=False,
                    periodic=True,
                    category=False,
                    active_bits=args.day_of_week_width,
                    sparsity=0.0,
                    size=0,
                    radius=args.day_of_week_radius,
                    resolution=0.0,
                )
                self._dayofweek_encoder = ScalarEncoder(p)

            self._bucketMap[self.DAYOFWEEK] = len(self._buckets)
            self._buckets.append(0.0)
            size += self._dayofweek_encoder.size

        # -------- Weekend --------
        if args.weekend_width != 0:
            if self._rdse_used:
                p = RDSEParameters(
                    size=10,
                    active_bits=args.weekend_width,
                    sparsity=0.0,
                    radius=0.0,
                    resolution=0.0,
                    category=True,  # binary category 0/1
                    seed=44,
                )
                self._weekend_encoder = RandomDistributedScalarEncoder(p)
            else:
                p = ScalarEncoderParameters(
                    minimum=0,
                    maximum=1,
                    clip_input=False,
                    periodic=False,
                    category=True,  # binary category 0/1
                    active_bits=args.weekend_width,
                    sparsity=0.0,
                    size=0,
                    radius=0.0,
                    resolution=0.0,
                )
                self._weekend_encoder = ScalarEncoder(p)

            self._bucketMap[self.WEEKEND] = len(self._buckets)
            self._buckets.append(0.0)
            size += self._weekend_encoder.size

        # -------- Custom days --------
        if args.custom_width != 0:
            if not args.custom_days:
                raise ValueError(
                    "DateEncoder: custom_days must contain at least one pattern string."
                )

            # Map strings to Python tm_wday (0=Mon..6=Sun)
            daymap = {
                "mon": 0,
                "tue": 1,
                "wed": 2,
                "thu": 3,
                "fri": 4,
                "sat": 5,
                "sun": 6,
            }
            for spec in args.custom_days:
                s = spec.lower()
                parts = [x.strip() for x in s.split(",") if x.strip()]
                for day in parts:
                    if len(day) < 3:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    key = day[:3]
                    if key not in daymap:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    self._customDays.add(daymap[key])

            if self._rdse_used:
                p = RDSEParameters(
                    size=10,
                    active_bits=args.custom_width,
                    sparsity=0.0,
                    radius=0.0,
                    resolution=0.0,
                    category=True,  # boolean category
                    seed=45,
                )
                self._customdays_encoder = RandomDistributedScalarEncoder(p)
            else:
                p = ScalarEncoderParameters(
                    minimum=0,
                    maximum=1,
                    clip_input=False,
                    periodic=False,
                    category=True,  # boolean category
                    active_bits=args.custom_width,
                    sparsity=0.0,
                    size=0,
                    radius=0.0,
                    resolution=0.0,
                )
                self._customdays_encoder = ScalarEncoder(p)

            self._bucketMap[self.CUSTOM] = len(self._buckets)
            self._buckets.append(0.0)
            size += self._customdays_encoder.size

        # -------- Holiday --------
        if args.holiday_width != 0:
            for day in args.holiday_dates:
                if len(day) not in (2, 3):
                    raise ValueError(
                        "DateEncoder: holiday_dates entries must be [mon,day] or [year,mon,day]."
                    )
            if self._rdse_used:
                p = RDSEParameters(
                    size=10,
                    active_bits=args.holiday_width,
                    sparsity=0.0,
                    radius=1.0,
                    resolution=0.0,
                    category=False,
                    seed=46,
                )
                self._holiday_encoder = RandomDistributedScalarEncoder(p)
            else:
                p = ScalarEncoderParameters(
                    minimum=0,
                    maximum=2,
                    clip_input=False,
                    periodic=True,
                    category=False,
                    active_bits=args.holiday_width,
                    sparsity=0.0,
                    size=0,
                    radius=1.0,
                    resolution=0.0,
                )

                self._holiday_encoder = ScalarEncoder(p)

            self._bucketMap[self.HOLIDAY] = len(self._buckets)
            self._buckets.append(0.0)
            size += self._holiday_encoder.size

        # -------- Time of day --------
        if args.time_of_day_width != 0:
            if self._rdse_used:
                p = RDSEParameters(
                    size=100,
                    active_bits=args.time_of_day_width,
                    sparsity=0.0,
                    radius=args.time_of_day_radius,
                    resolution=0.0,
                    category=False,
                    seed=47,
                )
                self._timeofday_encoder = RandomDistributedScalarEncoder(p)
            else:
                p = ScalarEncoderParameters(
                    minimum=0,
                    maximum=24,
                    clip_input=False,
                    periodic=True,
                    category=False,
                    active_bits=args.time_of_day_width,
                    sparsity=0.0,
                    size=0,
                    radius=args.time_of_day_radius,
                    resolution=0.0,
                )
                self._timeofday_encoder = ScalarEncoder(p)

            self._bucketMap[self.TIMEOFDAY] = len(self._buckets)
            self._buckets.append(0.0)
            size += self._timeofday_encoder.size

        self._size = size

    @property
    def size(self) -> int:
        """Total number of bits in the DateEncoder output."""
        return self._size

    @override
    def encode(
        self, input_value: datetime | pd.Timestamp | time.struct_time | None
    ) -> List[int]:
        """
        Encode a timestamp-like value into a sparse representation.

        input_value:
          - None          -> current local time
          - int/float     -> UNIX epoch seconds
          - datetime      -> datetime (naive treated as local)
          - struct_time   -> used directly

        Returns:
            List of active bit indices representing the encoded date.
        """
        if input_value is None:
            t = time.localtime()
        elif isinstance(input_value, (int, float)):
            t = time.localtime(float(input_value))
        elif isinstance(input_value, datetime):
            ts = input_value.timestamp()
            t = time.localtime(ts)
        elif isinstance(input_value, time.struct_time):
            t = input_value
        else:
            raise TypeError(f"Unsupported type for DateEncoder.encode: {type(input_value)}")

        # Collect per-attribute encodings to later concatenate
        encodings: List[tuple[List[int], int]] = []  # (sparse indices, size)

        # --- Season: day of year (0-based) ---
        if self._season_encoder is not None:
            day_of_year = float(t.tm_yday - 1)  # tm_yday is 1..366
            sparse = self._season_encoder.encode(day_of_year)
            # bucket index: floor(day / radius)
            bucket_idx = math.floor(day_of_year / self._season_encoder._radius)
            self._buckets[self._bucketMap[self.SEASON]] = float(bucket_idx)

            encodings.extend(sparse)

        # --- Day of week (Monday=0..Sunday=6, same as header comment) ---
        if self._dayofweek_encoder is not None:
            # C++: dayOfWeek = (tm_wday + 6) % 7, with tm_wday 0=Sun..6=Sat
            # Python tm_wday: 0=Mon..6=Sun
            # So emulate C++ tm_wday first:
            c_tm_wday = (t.tm_wday + 1) % 7  # now 0=Sun..6=Sat
            day_of_week = float((c_tm_wday + 6) % 7)
            sparse = self._dayofweek_encoder.encode(day_of_week)
            radius = max(self._dayofweek_encoder._radius, 1e-9)
            bucket_val = day_of_week - math.fmod(day_of_week, radius)
            self._buckets[self._bucketMap[self.DAYOFWEEK]] = bucket_val

            encodings.extend(sparse)

        else:
            # still compute c_tm_wday for weekend/custom use
            c_tm_wday = (t.tm_wday + 1) % 7

        # --- Weekend flag (Fri 18:00 .. Sun 23:59) ---
        if self._weekend_encoder is not None:
            # C++ logic uses C tm_wday (0=Sun..6=Sat)
            if c_tm_wday == 0 or c_tm_wday == 6 or (c_tm_wday == 5 and t.tm_hour > 18):
                val = 1.0
            else:
                val = 0.0
            sparse = self._weekend_encoder.encode(val)
            self._buckets[self._bucketMap[self.WEEKEND]] = val

            encodings.extend(sparse)

        # --- Custom days ---
        if self._customdays_encoder is not None:
            # customDays_ holds Python tm_wday (0=Mon..6=Sun)
            custom_val = 1.0 if t.tm_wday in self._customDays else 0.0
            sparse = self._customdays_encoder.encode(custom_val)
            self._buckets[self._bucketMap[self.CUSTOM]] = custom_val

            encodings.extend(sparse)

        # --- Holiday ramp ---
        if self._holiday_encoder is not None:
            val = self._holiday_value(t)
            sparse = self._holiday_encoder.encode(val)
            self._buckets[self._bucketMap[self.HOLIDAY]] = math.floor(val)

            encodings.extend(sparse)

        # --- Time of day ---
        if self._timeofday_encoder is not None:
            tod = t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0
            sparse = self._timeofday_encoder.encode(tod)
            radius = max(self._timeofday_encoder._radius, 1e-9)
            bucket_val = tod - math.fmod(tod, radius)
            self._buckets[self._bucketMap[self.TIMEOFDAY]] = bucket_val

            encodings.extend(sparse)

        if not encodings:
            raise RuntimeError("DateEncoder misconfigured: no sub-encoders enabled.")

        # Concatenate encodings into a single list of sparse indices
        return encodings

    def _holiday_value(self, t: time.struct_time) -> float:
        """Return the holiday ramp value for the provided timestamp."""
        seconds_per_day = 86400.0
        input_ts = time.mktime(t)

        for h in self._parameters.holiday_dates:
            if len(h) == 3:
                year, mon, day = h
            else:
                year = t.tm_year
                mon, day = h
            h_ts = self.mktime(year, mon, day)

            if input_ts > h_ts:
                diff = input_ts - h_ts
                if diff < seconds_per_day:
                    return 1.0
                elif diff < 2.0 * seconds_per_day:
                    return 1.0 + (diff - seconds_per_day) / seconds_per_day
            else:
                diff = h_ts - input_ts
                if diff < seconds_per_day:
                    return 1.0 - diff / seconds_per_day

        return 0.0

    @staticmethod
    def mktime(year: int, mon: int, day: int, hr: int = 0, minute: int = 0, sec: int = 0) -> float:
        """Convenience to generate unix epoch seconds like the C++ static mktime."""
        dt = datetime(year, mon, day, hr, minute, sec)
        return time.mktime(dt.timetuple())


@dataclass
class DateEncoderParameters:
    """Configuration parameters for DateEncoder.

    Each field controls the encoding of a specific temporal feature.
    Set the corresponding width to a nonzero value to enable encoding for that feature.

    Attributes:
        season_width: Number of active bits for season (day of year).
        season_radius: Radius for season encoding (days).
        day_of_week_width: Number of active bits for day of week.
        day_of_week_radius: Radius for day of week encoding.
        weekend_width: Number of active bits for weekend flag.
        holiday_width: Number of active bits for holiday encoding.
        holiday_dates: List of holidays as [month, day] or [year, month, day].
        time_of_day_width: Number of active bits for time of day.
        time_of_day_radius: Radius for time of day encoding (hours).
        custom_width: Number of active bits for custom day groups.
        custom_days: List of custom day group strings (e.g., ["mon,wed,fri"]).


        /**
         * The DateEncoderParameters structure is used to pass configuration parameters to
         * the DateEncoder. These Six (6) members define the total number of bits in the output.
         *     Members:  season, dayOfWeek, weekend, holiday, timeOfDay, customDays
         *
         * Each member is a separate attribute of a date/time that can be activated
         * by providing a width parameter and sometimes a radius parameter.
         * Each is implemented separately using a ScalarEncoder and the results
         * are concatinated together.
         *
         * The width attribute determines the number of bits to be used for each member.
         * and 0 means don't use.  The width is like a weighting to indicate the relitive importance
         * of this member to the overall data value.
         *
         * The radius attribute indicates the size of the bucket; the quantization size.
         * All values in the same bucket generate the same pattern.
         *
         * To avoid problems with leap year, consider a year to have 366 days.
         * The timestamp will be converted to components such as time and dst based on
         * local timezone and location (see localtime()).
         *
         */
    """

    # Season: day of year (0..366), default radius 91.5 days (~4 seasons)
    season_width: int = 0
    """Number of active bits for season (day of year). how many bits to apply to season
      /**
        *  Member: season -  The portion of the year. Unit is day.  Default radius is
        *                    91.5 days which gives 4 seasons per year.
        */
   """

    season_radius: float = 91.5
    """Radius for season encoding, in days (default ~4 seasons).
    days per season
    """

    # Day of week: Monday=0, Tuesday=1, ... (C++ maps from tm_wday)
    day_of_week_width: int = 0
    """Number of active bits for day of week, how many bits to apply to day of week.

    """

    day_of_week_radius: float = 1.0
    """Radius for day of week encoding, every day is a separate bucket."""

    # Weekend flag (0/1, Fri 6pm through Sun midnight)
    weekend_width: int = 0
    """Number of active bits for weekend flag."""

    # Holiday: boolean-ish with ramp, default dates = [[12, 25]] (month, day)
    holiday_width: int = 0
    """Number of active bits for holiday encoding."""

    holiday_dates: List[List[int]] = field(default_factory=lambda: [[12, 25]])
    """List of holidays as [month, day] or [year, month, day]."""

    # Time of day: 0..24 hours
    time_of_day_width: int = 0
    """Number of active bits for time of day."""

    time_of_day_radius: float = 4.0
    """Radius for time of day encoding, in hours."""

    # Custom day groups (e.g. ["mon,wed,fri"])
    custom_width: int = 0
    """Number of active bits for custom day groups."""

    custom_days: List[str] = field(default_factory=list)
    """List of custom day group strings (e.g., ["mon,wed,fri"])."""

    rdse_used: bool = True
    """Enable RDSE usage for date encoder."""

    encoder_class = DateEncoder


if __name__ == "__main__":

    date_params = DateEncoderParameters(
        day_of_week_radius=1,
        day_of_week_width=10,
        time_of_day_radius=1,
        time_of_day_width=10,
        weekend_width=20, 
    )
    encoder = DateEncoder(date_params)
    output = encoder.encode(datetime(2026, 1, 26, 14, 30, 0))
    print("Output size:", encoder._size)
    print("Active indices:", output)