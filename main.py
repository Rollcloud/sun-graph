# Calculate and graph various sunrise parameters.
#
# Note 1:
#
# The USNO definition for sunset/sunrise is when the center of the sun is 0.8333 degrees below the horizon
# this is an approximation that takes into account the sun's average radius and an average amount of atmospheric refraction.
# See https://github.com/astropy/astroplan/issues/409#issuecomment-554570085

import re
from datetime import datetime
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
from astropy.time import Time
from matplotlib.lines import Line2D
from munch import Munch
from pytz import timezone
from tqdm import tqdm

START_DATE = datetime(2023, 11, 1)
END_DATE = datetime(2024, 11, 1)

SECONDS_IN_A_MINUTE = 60
SECONDS_IN_A_HOUR = SECONDS_IN_A_MINUTE * 60
SECONDS_IN_A_DAY = SECONDS_IN_A_HOUR * 24

ISO_DATE_FORMAT = "%Y-%m-%d"
A4_INCHES = (11.69, 8.27)
DPI = 300

with open("config.yaml") as f:
    cfg = Munch.fromDict(yaml.load(f, Loader=yaml.FullLoader))

with open("locations.yaml") as f:
    locs = Munch.fromDict(yaml.load(f, Loader=yaml.FullLoader))

p = Path()


class DataInvalidError(Exception):
    """Data has been marked as invalid."""

    pass


def get_sun_times(location: Observer, start_date, end_date, tz=None):
    times = []
    year_of_days = pd.date_range(start_date, end_date, inclusive="both", tz=tz)
    for day in tqdm(year_of_days):
        sunrise = location.sun_rise_time(Time(day), "next", horizon=-0.8333 * u.deg)  # see Note 1
        local_sunrise = sunrise.to_datetime(timezone=tz)
        time_of_sunrise = (local_sunrise - day).total_seconds()

        noon = location.noon(Time(day), "next")
        local_noon = noon.to_datetime(timezone=tz)
        time_of_noon = (local_noon - day).total_seconds()

        sunset = location.sun_set_time(Time(day), "next", horizon=-0.8333 * u.deg)  # see Note 1
        local_sunset = sunset.to_datetime(timezone=tz)
        time_of_sunset = (local_sunset - day).total_seconds()

        dawn = location.twilight_morning_civil(Time(day), "next")
        local_dawn = dawn.to_datetime(timezone=tz)
        time_of_dawn = (local_dawn - day).total_seconds()

        dusk = location.twilight_evening_civil(Time(day), "next")
        local_dusk = dusk.to_datetime(timezone=tz)
        time_of_dusk = (local_dusk - day).total_seconds()

        times.append(
            {
                "date": day,
                "dawn": time_of_dawn,
                "sunrise": time_of_sunrise,
                "noon": time_of_noon,
                "sunset": time_of_sunset,
                "dusk": time_of_dusk,
            }
        )

    return pd.DataFrame(times)


def sun_times(location: Observer, start_date, end_date, recalculate=False):
    filename = "_".join(
        [
            location.name,
            start_date.strftime(ISO_DATE_FORMAT),
            end_date.strftime(ISO_DATE_FORMAT),
            str(location.timezone),
        ]
    )
    filename = _clean_name(filename)
    filepath = p / "tmp" / f"{filename}.pkl"

    try:
        if recalculate:
            raise DataInvalidError("Recalculate requested.")
        return pd.read_pickle(filepath)
    except (DataInvalidError, FileNotFoundError):
        df = get_sun_times(location, START_DATE, END_DATE, tz=location.timezone)
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


def plot_sun_times(location, df, df_events, start_date, end_date, media="display"):
    gs_kw = dict(width_ratios=[1], height_ratios=[3, 1])
    fig, axd = plt.subplot_mosaic(
        [["upper"], ["lower"]], gridspec_kw=gs_kw, figsize=A4_INCHES, dpi=DPI, layout="tight"
    )
    ax_t = axd["upper"]
    ax_dt = axd["lower"]

    ax_t.fill_between(df.date, 0, df.dawn, **cfg[media].fills.nightlight)
    ax_t.plot(df.date, df.dawn, lw=1, color=cfg.colours.twilight)
    ax_t.fill_between(df.date, df.dawn, df.sunrise, **cfg[media].fills.twilight)
    ax_t.plot(df.date, df.sunrise, lw=2, color=cfg.colours.sunrise)
    ax_t.fill_between(df.date, df.sunrise, df.sunset, **cfg[media].fills.daylight)
    ax_t.plot(df.date, df.noon, lw=2, color=cfg.colours.sunlight)
    ax_t.plot(df.date, df.sunset, lw=2, color=cfg.colours.sunset)
    ax_t.fill_between(df.date, df.sunset, df.dusk, **cfg[media].fills.twilight)
    ax_t.plot(df.date, df.dusk, lw=1, color=cfg.colours.twilight)
    ax_t.fill_between(df.date, df.dusk, SECONDS_IN_A_DAY, **cfg[media].fills.nightlight)

    ax_t.plot(
        df_events.date, df_events.sunrise, linestyle="None", marker="o", color=cfg.colours.sunrise
    )
    ax_t.plot(
        df_events.date, df_events.noon, linestyle="None", marker="o", color=cfg.colours.sunlight
    )
    ax_t.plot(
        df_events.date, df_events.sunset, linestyle="None", marker="o", color=cfg.colours.sunset
    )
    for _idx, row in df_events.iterrows():
        ax_t.annotate(
            _time_formatter(row.sunrise, None),
            (row.date, row.sunrise),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="center",
            color=cfg.colours.sunrise,
        )
        ax_t.annotate(
            _time_formatter(row.noon, None),
            (row.date, row.noon),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="center",
            color=cfg.colours.sunlight,
        )
        ax_t.annotate(
            _time_formatter(row.sunset, None),
            (row.date, row.sunset),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="center",
            color=cfg.colours.sunset,
        )

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
        annotation_direction_sunrise = np.sign(row.sunrise_delta)
        annotation_direction_sunset = np.sign(row.sunset_delta)
        # if values are within a minute of each other, flip sign of one annotation
        if abs(row.sunrise_delta - row.sunset_delta) < SECONDS_IN_A_MINUTE:
            annotation_direction_sunrise *= -1
        ax_dt.annotate(
            _delta_time_formatter(row.sunrise_delta, None),
            (row.date, row.sunrise_delta),
            xytext=(0, -16 * annotation_direction_sunrise),
            textcoords="offset points",
            ha="center",
            va="center",
            color=cfg.colours.sunrise,
        )
        ax_dt.annotate(
            _delta_time_formatter(row.sunset_delta, None),
            (row.date, row.sunset_delta),
            xytext=(0, -16 * annotation_direction_sunset),
            textcoords="offset points",
            ha="center",
            va="center",
            color=cfg.colours.sunset,
        )

    ax_t.set_xlim(start_date, end_date)
    ax_t.set_ylim(0, SECONDS_IN_A_DAY)

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
    twilight = mpatches.Patch(**cfg[media].fills.twilight, label="Civil Twilight")
    daytime = mpatches.Patch(**cfg[media].fills.daylight, label="Daylight")
    sunrise = Line2D([0], [0], color=cfg.colours.sunrise, label="Sunrise")
    sunset = Line2D([0], [0], color=cfg.colours.sunset, label="Sunset")
    solstice_equinox = Line2D(
        [0], [0], color=cfg.colours.sunlight, marker="o", linestyle="none", label="Solstice/Equinox"
    )
    noon = Line2D([0], [0], color=cfg.colours.sunlight, label="Solar Noon")
    change_sunrise = Line2D(
        [0], [0], color=cfg.colours.sunrise, linestyle="dotted", label="Change Sunrise", lw=2
    )
    change_sunset = Line2D(
        [0], [0], color=cfg.colours.sunset, linestyle="dashed", label="Change Sunset"
    )
    upper_handles = [nighttime, twilight, daytime, solstice_equinox, sunrise, noon, sunset]
    lower_handles = [solstice_equinox, change_sunrise, change_sunset]
    ax_t.legend(
        handles=upper_handles, loc="center", bbox_to_anchor=(0.5, 1.05), ncol=len(upper_handles)
    )
    ax_dt.legend(
        handles=lower_handles, loc="center", bbox_to_anchor=(0.5, 1.10), ncol=len(lower_handles)
    )

    ax_t.set_xlabel("Date")
    ax_t.set_ylabel("Local Time")
    ax_dt.set_xlabel("Date")
    ax_dt.set_ylabel("ΔTime")
    plt.suptitle(f"Sun Graph - {location.name}", size=18)

    location_name = _clean_name(location.name)
    match media:
        case "display":
            plt.savefig(p / "tmp" / f"sun-graph_{location_name}.png")
        case "print":
            plt.savefig(p / "tmp" / f"sun-graph_{location_name}.pdf")
        case _:
            pass
    plt.close()


@click.command()
@click.argument("location_name")
@click.option("--recalculate", is_flag=True, help="Recalculate sunrise and sunset times.")
def main(location_name, recalculate):
    """Generate sun graphs for LOCATION_NAME."""

    loc_data = locs[location_name]
    location = Observer(
        longitude=loc_data.longitude_deg * u.deg,
        latitude=loc_data.latitude_deg * u.deg,
        elevation=loc_data.elevation_m * u.m,
        name=loc_data.name,
        timezone=timezone(loc_data.timezone),
        pressure=0 * u.mbar,  # see Note 1
    )

    df = sun_times(location, START_DATE, END_DATE, recalculate=recalculate)

    events = [str(event) for event in cfg.events]
    df_events = df[["date", "sunrise", "noon", "sunset"]]
    df_events = df_events[df.date.dt.strftime(ISO_DATE_FORMAT).isin(events)]

    plot_sun_times(location, df, df_events, START_DATE, END_DATE, "display")
    plot_sun_times(location, df, df_events, START_DATE, END_DATE, "print")


if __name__ == "__main__":
    main()
