# Calculate and graph various sunrise parameters.
#
# Note 1:
#
# The USNO definition for sunset/sunrise is when the center of the sun is 0.8333
# degrees below the horizon this is an approximation that takes into account the
# sun's average radius and an average amount of atmospheric refraction. See
# https://github.com/astropy/astroplan/issues/409#issuecomment-554570085

import re
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import astropy.units as u
import click
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import yaml
from astroplan import Observer
from astroplan.exceptions import TargetAlwaysUpWarning, TargetNeverUpWarning
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from matplotlib.lines import Line2D
from munch import Munch
from pytz import timezone
from tqdm import tqdm
import warnings

START_DATE = datetime(2024, 10, 1)
END_DATE = datetime(2025, 11, 1)

SECONDS_IN_A_MINUTE = 60
SECONDS_IN_A_HOUR = SECONDS_IN_A_MINUTE * 60
SECONDS_IN_A_DAY = SECONDS_IN_A_HOUR * 24

USNO_HORIZON = -0.8333 * u.deg  # see Note 1

ISO_DATE_FORMAT = "%Y-%m-%d"
A4_INCHES = (11.69, 8.27)
DPI = 300

with open("config.yaml") as f:
    cfg = Munch.fromDict(yaml.load(f, Loader=yaml.FullLoader))

with open("locations.yaml", encoding="utf8") as f:
    locs = Munch.fromDict(yaml.load(f, Loader=yaml.FullLoader))

p = Path()


class DataInvalidError(Exception):
    """Data has been marked as invalid."""

    pass


class Astrolabe:
    """Use to calculate sunrise and sunset times and other events."""

    def __init__(self, observer, tz, day):
        self.observer = observer
        self.tz = tz
        self.day = day
        self.midnight = observer.midnight(Time(day), which="nearest")
        self.noon = observer.noon(Time(day), which="next")

    def to_local_seconds(self, event):
        """Convert event to local time in seconds."""
        local_event = event.to_datetime(timezone=self.tz)
        return (local_event - self.day).total_seconds()

    def calculate(self, event_name, which="next", **kwargs):
        if event_name == "noon":
            return self.to_local_seconds(self.noon)
        if event_name == "midnight":
            return self.to_local_seconds(self.midnight)

        with warnings.catch_warnings():
            # Show all warnings only once
            warnings.simplefilter("once", AstropyWarning)
            # Raise errors for specific warnings
            warnings.simplefilter("error", TargetNeverUpWarning)
            warnings.simplefilter("error", TargetAlwaysUpWarning)
            # Ignore FutureWarnings to prevent unnecessary warnings
            warnings.simplefilter("ignore", FutureWarning)

            try:
                event = getattr(self.observer, event_name)(self.midnight, which=which, **kwargs)
                return self.to_local_seconds(event)
            except TargetNeverUpWarning:
                # this event doesn't occur, return noon
                return self.to_local_seconds(self.noon)
            except TargetAlwaysUpWarning:
                # this event occurs all day, return midnight
                midnight = self.to_local_seconds(self.noon)
                if "set" in event_name or "evening" in event_name:
                    # if the event is a setting event, return the following midnight
                    midnight += SECONDS_IN_A_DAY
                return midnight


def get_sun_times(observer: Observer, start_date, end_date, tz=None):
    times = []
    year_of_days = pd.date_range(start_date, end_date, inclusive="both", tz=tz)
    for day in tqdm(year_of_days):
        astrolabe = Astrolabe(observer, tz, day)
        times.append(
            {
                "date": day,
                "midnight": astrolabe.calculate("midnight"),
                "astronomical_dawn": astrolabe.calculate("twilight_morning_astronomical"),
                "nautical_dawn": astrolabe.calculate("twilight_morning_nautical"),
                "dawn": astrolabe.calculate("twilight_morning_civil"),
                "sunrise": astrolabe.calculate("sun_rise_time", horizon=USNO_HORIZON),  # see Note 1
                "noon": astrolabe.calculate("noon"),
                "sunset": astrolabe.calculate("sun_set_time", horizon=USNO_HORIZON),  # see Note 1
                "dusk": astrolabe.calculate("twilight_evening_civil"),
                "nautical_dusk": astrolabe.calculate("twilight_evening_nautical"),
                "astronomical_dusk": astrolabe.calculate("twilight_evening_astronomical"),
            }
        )

    return pd.DataFrame(times)


def fix_noon_data_for(df, column):
    """Fix the data in the column equal to noon."""
    # Identify single values equal to noon, where the previous and next values are not equal to noon
    error_mask = (df[column] == df.noon) & (
        (df[column].shift(1) != df.noon.shift(1)) & (df[column].shift(-1) != df.noon.shift(-1))
    )

    # Set the column to the value of the midnight column where the mask is True
    offset = SECONDS_IN_A_DAY if ("set" in column or "dusk" in column) else 0
    df[column] = df[column].mask(error_mask, df.midnight + offset)

    return df


def mask_midnight_data_for(df, column):
    """Mask the data in the column equal to midnight, except for the first and last occurrences."""
    # Identify all values equal to midnight
    mask = (df[column] == df.midnight) | (df[column] == df.midnight + SECONDS_IN_A_DAY)

    # Shrink the mask to remove the first and last occurrences where mask is True
    mask = mask.shift(1) & mask.shift(-1)

    # Apply the mask to the column
    df[f"{column}_masked"] = df[column].mask(mask)

    return df


def fix_invalid_data(df):
    df = fix_noon_data_for(df, "astronomical_dawn")
    df = fix_noon_data_for(df, "nautical_dawn")
    df = fix_noon_data_for(df, "dawn")
    df = fix_noon_data_for(df, "sunrise")
    df = fix_noon_data_for(df, "sunset")
    df = fix_noon_data_for(df, "dusk")
    df = fix_noon_data_for(df, "nautical_dusk")
    df = fix_noon_data_for(df, "astronomical_dusk")

    df = mask_midnight_data_for(df, "astronomical_dawn")
    df = mask_midnight_data_for(df, "nautical_dawn")
    df = mask_midnight_data_for(df, "dawn")
    df = mask_midnight_data_for(df, "sunrise")
    df = mask_midnight_data_for(df, "sunset")
    df = mask_midnight_data_for(df, "dusk")
    df = mask_midnight_data_for(df, "nautical_dusk")
    df = mask_midnight_data_for(df, "astronomical_dusk")

    return df


def sun_times(observer: Observer, start_date, end_date, recalculate=False):
    filename = "_".join(
        [
            observer.name,
            start_date.strftime(ISO_DATE_FORMAT),
            end_date.strftime(ISO_DATE_FORMAT),
            str(observer.timezone),
        ]
    )
    filename = _clean_name(filename)
    filepath = p / "tmp" / "data" / f"{filename}.pkl"

    try:
        if recalculate:
            raise DataInvalidError("Recalculate requested.")
        return pd.read_pickle(filepath)
    except (DataInvalidError, FileNotFoundError):
        df = get_sun_times(observer, START_DATE, END_DATE, tz=observer.timezone)
        df.to_pickle(filepath)
        return df


def _clean_name(name):
    replacements = {
        r"[\']+": "",
        r"[^\w]+": "-",  # this is a catch-all and should happen last
    }
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)

    return name.lower()


def _time_formatter(seconds_in, _position):
    seconds_in = int(seconds_in)
    hours = seconds_in // SECONDS_IN_A_HOUR
    minutes = (seconds_in - (hours * SECONDS_IN_A_HOUR)) // SECONDS_IN_A_MINUTE
    return f"{hours:02d}h{minutes:02d}"


def _delta_time_formatter(seconds_in, _position):
    seconds_in = int(seconds_in)
    sign = "+" if seconds_in > 0 else "" if seconds_in == 0 else "-"
    seconds_in = abs(seconds_in)
    minutes = seconds_in // SECONDS_IN_A_MINUTE
    seconds = seconds_in - (minutes * SECONDS_IN_A_MINUTE)
    return f"{sign}{minutes:d}m{seconds:02d}s"


def _label_value(axis, x, y, formatter, colour_name, vertical_offset):
    axis.annotate(
        formatter(y, None),
        (x, y),
        xytext=(0, vertical_offset),
        textcoords="offset points",
        ha="center",
        va="center",
        color=cfg.colours[colour_name],
    )


def plot_sun_times_with_offset(ax_t, df, offset=0, media="display"):
    offset_seconds = offset * SECONDS_IN_A_DAY

    ax_t.plot(
        df.date,
        df.astronomical_dawn_masked + offset_seconds,
        lw=1,
        color=cfg.colours.astronomical_twilight,
    )
    ax_t.plot(
        df.date,
        df.astronomical_dusk_masked + offset_seconds,
        lw=1,
        color=cfg.colours.astronomical_twilight,
    )
    ax_t.fill_between(
        df.date,
        df.astronomical_dawn + offset_seconds,
        df.nautical_dawn + offset_seconds,
        **cfg[media].fills.astronomical_twilight,
    )
    ax_t.fill_between(
        df.date,
        df.nautical_dusk + offset_seconds,
        df.astronomical_dusk + offset_seconds,
        **cfg[media].fills.astronomical_twilight,
    )
    ax_t.plot(
        df.date, df.nautical_dawn_masked + offset_seconds, lw=1, color=cfg.colours.nautical_twilight
    )
    ax_t.plot(
        df.date, df.nautical_dusk_masked + offset_seconds, lw=1, color=cfg.colours.nautical_twilight
    )
    ax_t.fill_between(
        df.date,
        df.nautical_dawn + offset_seconds,
        df.dawn + offset_seconds,
        **cfg[media].fills.nautical_twilight,
    )
    ax_t.fill_between(
        df.date,
        df.dusk + offset_seconds,
        df.nautical_dusk + offset_seconds,
        **cfg[media].fills.nautical_twilight,
    )
    ax_t.plot(df.date, df.dawn_masked + offset_seconds, lw=1, color=cfg.colours.twilight)
    ax_t.plot(df.date, df.dusk_masked + offset_seconds, lw=1, color=cfg.colours.twilight)
    ax_t.fill_between(
        df.date, df.dawn + offset_seconds, df.sunrise + offset_seconds, **cfg[media].fills.twilight
    )
    ax_t.fill_between(
        df.date, df.sunset + offset_seconds, df.dusk + offset_seconds, **cfg[media].fills.twilight
    )
    ax_t.plot(df.date, df.sunrise_masked + offset_seconds, lw=2, color=cfg.colours.sunrise)
    ax_t.plot(df.date, df.sunset_masked + offset_seconds, lw=2, color=cfg.colours.sunset)
    ax_t.fill_between(
        df.date,
        df.sunrise + offset_seconds,
        df.sunset + offset_seconds,
        **cfg[media].fills.daylight,
    )
    # ax_t.plot(df.date, df.noon + offset_seconds, lw=2, color=cfg.colours.sunlight)


def plot_sun_times(observer, df, df_events, start_date, end_date, media="display"):
    gs_kw = dict(width_ratios=[1], height_ratios=[4, 1])
    fig, axd = plt.subplot_mosaic(
        [["upper"], ["lower"]],
        gridspec_kw=gs_kw,
        figsize=A4_INCHES,
        dpi=DPI,
        layout="tight",
    )
    ax_t = axd["upper"]
    ax_dt = axd["lower"]

    # night background
    ax_t.fill_between(df.date, 0, SECONDS_IN_A_DAY, **cfg[media].fills.nightlight)
    plot_sun_times_with_offset(ax_t, df, offset=-1, media=media)
    plot_sun_times_with_offset(ax_t, df, offset=0, media=media)
    plot_sun_times_with_offset(ax_t, df, offset=1, media=media)

    ax_t.plot(
        df_events.date,
        df_events.sunrise,
        linestyle="None",
        marker="o",
        color=cfg.colours.sunrise,
    )
    ax_t.plot(
        df_events.date,
        df_events.noon,
        linestyle="None",
        marker="o",
        color=cfg.colours.sunlight,
    )
    ax_t.plot(
        df_events.date,
        df_events.sunset,
        linestyle="None",
        marker="o",
        color=cfg.colours.sunset,
    )
    for _idx, row in df_events.iterrows():
        _label_value(ax_t, row.date, row.sunrise, _time_formatter, "sunrise", 10)
        _label_value(ax_t, row.date, row.noon, _time_formatter, "sunlight", -10)
        _label_value(ax_t, row.date, row.sunset, _time_formatter, "sunset", -10)

    sunrise_delta = df.sunrise.diff()
    sunrise_delta[sunrise_delta.abs() > 30 * SECONDS_IN_A_MINUTE] = pd.NA
    sunset_delta = df.sunset.diff()
    sunset_delta[sunset_delta.abs() > 30 * SECONDS_IN_A_MINUTE] = pd.NA

    ax_dt.plot(df.date, sunrise_delta, color=cfg.colours.sunrise, linestyle="dotted", lw=2)
    ax_dt.plot(df.date, sunset_delta, color=cfg.colours.sunset, linestyle="dashed")

    df_events["sunrise_delta"] = sunrise_delta.loc[df_events.index]
    df_events["sunset_delta"] = sunset_delta.loc[df_events.index]

    ax_dt.plot(
        df_events.date,
        df_events.sunrise_delta,
        linestyle="None",
        marker="o",
        color=cfg.colours.sunrise,
    )
    ax_dt.plot(
        df_events.date,
        df_events.sunset_delta,
        linestyle="None",
        marker="o",
        color=cfg.colours.sunset,
    )

    for _idx, row in df_events.iterrows():
        offset_sunrise = -16 * np.sign(row.sunrise_delta)
        offset_sunset = -16 * np.sign(row.sunset_delta)
        # if values are within a minute of each other, flip sign of one annotation
        if abs(row.sunrise_delta - row.sunset_delta) < SECONDS_IN_A_MINUTE:
            offset_sunrise *= -1

        _label_value(
            ax_dt,
            row.date,
            row.sunrise_delta,
            _delta_time_formatter,
            "sunrise",
            offset_sunrise,
        )
        _label_value(
            ax_dt,
            row.date,
            row.sunset_delta,
            _delta_time_formatter,
            "sunset",
            offset_sunset,
        )

    ax_t.set_xlim(start_date, end_date)
    # ax_t.set_ylim(0, SECONDS_IN_A_DAY)

    ax_dt.set_xlim(start_date, end_date)
    # make y-axis limits symmetrical
    low, high = ax_dt.get_ylim()
    bound = max(abs(low), abs(high))
    ax_dt.set_ylim(-bound, bound)

    ax_t.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_t.xaxis.set_minor_locator(mdates.MonthLocator())
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("1 %b '%y"))

    ax_dt.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_dt.xaxis.set_minor_locator(mdates.MonthLocator())
    ax_dt.xaxis.set_major_formatter(mdates.DateFormatter("1 %b '%y"))

    minutely = SECONDS_IN_A_MINUTE
    hourly = np.linspace(0, SECONDS_IN_A_DAY, 24 + 1)
    two_hourly = np.linspace(0, SECONDS_IN_A_DAY, 12 + 1)
    ax_t.yaxis.set_major_locator(ticker.FixedLocator(two_hourly))
    ax_t.yaxis.set_minor_locator(ticker.FixedLocator(hourly))
    ax_t.yaxis.set_major_formatter(ticker.FuncFormatter(_time_formatter))

    ax_dt.yaxis.set_major_locator(ticker.MultipleLocator(minutely))
    ax_dt.yaxis.set_major_formatter(ticker.FuncFormatter(_delta_time_formatter))

    ax_t.grid(True, "minor", "both", zorder=1000, alpha=cfg[media].grid_alpha, c="0.8")
    ax_t.grid(True, "major", "both", zorder=1001, alpha=cfg[media].grid_alpha, c="0.5")

    ax_dt.grid(True, "minor", "both", zorder=1000, alpha=cfg[media].grid_alpha, c="0.8")
    ax_dt.grid(True, "major", "both", zorder=1001, alpha=cfg[media].grid_alpha, c="0.5")

    nighttime = mpatches.Patch(**cfg[media].fills.nightlight, label="Darkness")
    astronomical_twilight = mpatches.Patch(
        **cfg[media].fills.astronomical_twilight, label="Astronomical Twilight"
    )
    nautical_twilight = mpatches.Patch(
        **cfg[media].fills.nautical_twilight, label="Nautical Twilight"
    )
    twilight = mpatches.Patch(**cfg[media].fills.twilight, label="Civil Twilight")
    daytime = mpatches.Patch(**cfg[media].fills.daylight, label="Daylight")
    sunrise = Line2D([0], [0], color=cfg.colours.sunrise, label="Sunrise")
    sunset = Line2D([0], [0], color=cfg.colours.sunset, label="Sunset")
    solstice_equinox = Line2D(
        [0],
        [0],
        color=cfg.colours.sunlight,
        marker="o",
        linestyle="none",
        label="Solstice/Equinox",
    )
    noon = Line2D([0], [0], color=cfg.colours.sunlight, label="Solar Noon")
    change_sunrise = Line2D(
        [0],
        [0],
        color=cfg.colours.sunrise,
        linestyle="dotted",
        label="Change Sunrise",
        lw=2,
    )
    change_sunset = Line2D(
        [0], [0], color=cfg.colours.sunset, linestyle="dashed", label="Change Sunset"
    )
    upper_handles = [
        nighttime,
        astronomical_twilight,
        nautical_twilight,
        twilight,
        daytime,
        solstice_equinox,
        sunrise,
        noon,
        sunset,
    ]
    # interleave top and bottom halves of the legend entries to effectively convert columns to rows
    number_of_columns = int(np.ceil(len(upper_handles) / 2))
    upper_handles = [
        val
        for pair in zip_longest(
            upper_handles[:number_of_columns],
            upper_handles[number_of_columns:],
        )
        for val in pair
        if val is not None
    ]
    lower_handles = [solstice_equinox, change_sunrise, change_sunset]
    ax_t.legend(
        handles=upper_handles,
        loc="center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=number_of_columns,
    )
    ax_dt.legend(
        handles=lower_handles,
        loc="center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(lower_handles),
    )

    ax_t.set_xlabel("Date")
    ax_t.set_ylabel("Local Time")
    ax_dt.set_xlabel("Date")
    ax_dt.set_ylabel("ΔTime")
    plt.suptitle(f"Sun Graph - {observer.name}", size=18)

    observer_name = _clean_name(observer.name)
    match media:
        case "display":
            plt.savefig(p / "tmp" / "plots" / f"sun-graph_{observer_name}.png")
        case "print":
            plt.savefig(p / "tmp" / "plots" / f"sun-graph_{observer_name}.pdf")
        case _:
            pass
    plt.close()


@click.command()
@click.argument("observer_name")
@click.option("-r", "--recalculate", is_flag=True, help="Recalculate sunrise and sunset times.")
def main(observer_name, recalculate):
    """Generate sun graphs for observer_NAME."""

    loc_data = locs[observer_name]
    observer = Observer(
        longitude=loc_data.longitude_deg * u.deg,
        latitude=loc_data.latitude_deg * u.deg,
        elevation=loc_data.elevation_m * u.m,
        name=loc_data.name,
        timezone=timezone(loc_data.timezone),
        pressure=0 * u.mbar,  # see Note 1
    )

    df = sun_times(observer, START_DATE, END_DATE, recalculate=recalculate)
    df = fix_invalid_data(df)

    events = [str(event) for event in cfg.events]
    df_events = df[["date", "sunrise", "noon", "sunset"]]
    df_events = df_events[df.date.dt.strftime(ISO_DATE_FORMAT).isin(events)]

    plot_sun_times(observer, df, df_events, START_DATE, END_DATE, "display")
    plot_sun_times(observer, df, df_events, START_DATE, END_DATE, "print")


if __name__ == "__main__":
    main()
