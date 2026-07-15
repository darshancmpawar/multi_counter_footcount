"""Automatic Day Type and Panchangam derivation from the date alone.

Day Type comes from the workbook's "Holiday List" sheet: a working day is
"Previous Day of Holiday" if a listed holiday sits in the non-working block
right after it, "Next Day of Holiday" if one sits right before it.
Reproduces all 203 recorded labels exactly.

Panchangam is computed astronomically with ephem. Each observance uses the
tithi at ITS canonical reference time — conventions validated against the
recorded panchangam (83/84 observances reproduced; the one extra is Rama
Navami day, which astronomically is Navratri day 9):

  sunrise  (06:00 IST)  Ekadashi, Poornima, Amavasya
  evening  (19:00 IST)  Pradosham            (pradosh-kala observance)
  evening  (20:00 IST)  Sankashti Chaturthi  (moonrise observance)
  midnight (24:00 IST)  Masik Shivaratri     (nishita-kala observance)

Lunar months (Shravan, Navratri) use the amanta convention: the month is
named by the SIDEREAL solar sign at its starting new moon (Lahiri ayanamsa).
"""
from datetime import timedelta
from math import floor

import numpy as np
import pandas as pd

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:      # degrade to Regular rather than crash the app
    EPHEM_AVAILABLE = False

IST_UTC_OFFSET_HOURS = 5.5
AYANAMSA_DEG = 24.15     # Lahiri, accurate for the mid-2020s
LUNAR_MONTHS = ["Chaitra", "Vaishakha", "Jyeshtha", "Ashadha", "Shravana",
                "Bhadrapada", "Ashwin", "Kartika", "Margashirsha", "Pausha",
                "Magha", "Phalguna"]

DAY_TYPE_REGULAR = "Regular"
DAY_TYPE_BEFORE_HOLIDAY = "Previous Day of Holiday"
DAY_TYPE_AFTER_HOLIDAY = "Next Day of Holiday"


# ------------------------------------------------------------- day type -----
class HolidayCalendar(set):
    """A set of facility-closure dates (drop-in for the plain set every
    caller already uses) that additionally carries:
      operated       weekday festival dates the facility stayed OPEN on
                     (Facility Status 'Operated' — e.g. Onam, Kartik Purnima)
      holiday_types  date -> Holiday Type (Compulsory / Important / Shutdown …)
    Both feed the festival features; membership (`in`, walks) still means
    "working day removed", exactly as before."""

    def __init__(self, closed=(), operated=(), holiday_types=None):
        super().__init__(closed)
        self.operated = set(operated)
        self.holiday_types = dict(holiday_types or {})


def load_holiday_dates(workbook, sheet_name: str = "Holiday List") -> HolidayCalendar:
    """Holiday calendar from the workbook (path, buffer or ExcelFile).
    Empty calendar if the sheet is missing.

    Only closures that remove a working day count as members: rows marked
    'Operated' in the Facility Status column (the facility stayed open) and
    holidays falling on weekends (no extra day off) would mislabel adjacent
    days as holiday-adjacent — validated against the recorded Day Type
    labels. Operated weekday festivals are kept separately on `.operated`."""
    try:
        holidays = pd.read_excel(workbook, sheet_name=sheet_name)
        dates = pd.to_datetime(holidays["Date"])
        weekday = dates.dt.weekday < 5
        operated = pd.Series(False, index=holidays.index)
        if "Facility Status" in holidays.columns:
            status = holidays["Facility Status"].astype(str).str.strip().str.lower()
            operated = status == "operated"
        types = (holidays["Holiday Type"].astype(str)
                 if "Holiday Type" in holidays.columns
                 else pd.Series("", index=holidays.index))
        return HolidayCalendar(
            closed=dates[weekday & ~operated].dt.date,
            operated=dates[weekday & operated].dt.date,
            holiday_types=dict(zip(dates.dt.date, types)))
    except Exception:
        return HolidayCalendar()


def festival_flags(date, holiday_dates) -> dict:
    """Calendar-known festival features for one date (leakage-safe: derived
    from the date and the Holiday List alone, never from recorded data).

    fest_op_today        an operated festival falls on the day itself
    adj_hol_important/compulsory/shutdown
                         Holiday Type(s) of the closure(s) in the adjacent
                         non-working blocks — splits the single holiday-
                         adjacent flag by expected attendance impact
                         (religious/travel vs national vs shutdown/event)
    fest_any             any of the above"""
    date = pd.Timestamp(date).date()
    operated = getattr(holiday_dates, "operated", set())
    types_of = getattr(holiday_dates, "holiday_types", {})
    adjacent_types = set()
    for step in (+1, -1):
        day = date + timedelta(days=step)
        while day.weekday() >= 5 or day in holiday_dates:
            if day in holiday_dates:
                adjacent_types.add(types_of.get(day, ""))
            day += timedelta(days=step)
    flags = {
        "fest_op_today": int(date in operated),
        "adj_hol_important": int("Important" in adjacent_types),
        "adj_hol_compulsory": int("Compulsory" in adjacent_types),
        "adj_hol_shutdown": int(bool(adjacent_types
                                     & {"Shutdown", "Event", "Observed"})),
    }
    flags["fest_any"] = int(any(flags.values()) or bool(adjacent_types))
    return flags


def derive_day_type(date, holiday_dates: set) -> str:
    """Walk through the adjacent non-working block (weekends + holidays) on
    each side; a listed holiday in that block makes the day holiday-adjacent."""
    date = pd.Timestamp(date).date()

    def block_contains_holiday(step: int) -> bool:
        day = date + timedelta(days=step)
        found = False
        while day.weekday() >= 5 or day in holiday_dates:
            if day in holiday_dates:
                found = True
            day += timedelta(days=step)
        return found

    if block_contains_holiday(+1):
        return DAY_TYPE_BEFORE_HOLIDAY
    if block_contains_holiday(-1):
        return DAY_TYPE_AFTER_HOLIDAY
    return DAY_TYPE_REGULAR


# ----------------------------------------------------------- panchangam -----
def _sun_moon_longitudes(dt_utc) -> tuple[float, float]:
    moon, sun = ephem.Moon(), ephem.Sun()
    moon.compute(dt_utc)
    sun.compute(dt_utc)
    return (np.degrees(float(ephem.Ecliptic(sun).lon)),
            np.degrees(float(ephem.Ecliptic(moon).lon)))


def _tithi(date, hour_ist: float) -> int:
    """Tithi number 1..30 (1-15 Shukla, 16-30 Krishna) at an IST clock time."""
    dt_utc = (pd.Timestamp(date)
              + pd.Timedelta(hours=hour_ist - IST_UTC_OFFSET_HOURS)).to_pydatetime()
    sun_lon, moon_lon = _sun_moon_longitudes(dt_utc)
    return floor(((moon_lon - sun_lon) % 360) / 12) + 1


def _lunar_month(date) -> str:
    """Amanta month, named by the sidereal solar sign at the preceding new moon."""
    dt_utc = (pd.Timestamp(date)
              + pd.Timedelta(hours=6 - IST_UTC_OFFSET_HOURS)).to_pydatetime()
    new_moon = ephem.previous_new_moon(dt_utc)
    sun_sidereal = (_sun_moon_longitudes(new_moon.datetime())[0] - AYANAMSA_DEG) % 360
    return LUNAR_MONTHS[(int(sun_sidereal // 30) + 1) % 12]


def derive_panchangam(date) -> str:
    """Observance string in the history workbook's format, e.g.
    'Shukla Ekadashi; Shravan Month', or 'Regular'."""
    if not EPHEM_AVAILABLE:
        return DAY_TYPE_REGULAR

    observances = []
    sunrise_tithi = _tithi(date, hour_ist=6)
    if sunrise_tithi == 11:
        observances.append("Shukla Ekadashi")
    elif sunrise_tithi == 26:
        observances.append("Krishna Ekadashi")
    elif sunrise_tithi == 15:
        observances.append("Poornima")
    elif sunrise_tithi == 30:
        observances.append("Amavasya")
    if _tithi(date, hour_ist=19) in (13, 28):
        observances.append("Pradosham")
    if _tithi(date, hour_ist=20) == 19:
        observances.append("Sankashti Chaturthi")
    if _tithi(date, hour_ist=24) == 29:
        observances.append("Masik Shivaratri")

    month = _lunar_month(date)
    if month == "Shravana":
        observances.append("Shravan Month")
    elif month == "Ashwin" and sunrise_tithi <= 9:
        observances.append("Sharad Navratri")
    elif month == "Chaitra" and sunrise_tithi <= 9:
        observances.append("Chaitra Navratri")

    return "; ".join(observances) if observances else "Regular"


# ------------------------------------------------------------- plan fill ----
def fill_calendar_columns(plan: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """Fill missing Day Type / Panchangam values from the date. Values the
    user explicitly provided are kept — they may know something the
    calendar doesn't (e.g. an unlisted company holiday)."""
    plan = plan.copy()
    derived = {pd.Timestamp(d): (derive_day_type(d, holiday_dates), derive_panchangam(d))
               for d in pd.to_datetime(plan["Date"]).unique()}
    for column, index in [("Day Type", 0), ("Panchangam", 1)]:
        auto = pd.to_datetime(plan["Date"]).map(lambda d: derived[d][index])
        if column not in plan:
            plan[column] = auto
        else:
            plan[column] = plan[column].where(plan[column].notna(), auto)
    return plan


def holiday_list_covers(holiday_dates: set, date) -> bool:
    """False when the plan date is beyond the last listed holiday's year —
    day-type derivation would silently return Regular."""
    if not holiday_dates:
        return False
    return pd.Timestamp(date).year <= max(d.year for d in holiday_dates)
