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
        sunrise = location.sun_rise_time(Time(day), "next")
        local_sunrise = sunrise.to_datetime(timezone=tz)
        time_of_sunrise = (local_sunrise - day).total_seconds()

        sunset = location.sun_set_time(Time(day), "next")
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


def plot_sun_times(
    location, df, start_date, end_date, media="display", df_highlights=None, roc=False
):
    fig, ax = plt.subplots(figsize=A4_INCHES, dpi=DPI)

    plt.fill_between(df.date, 0, df.dawn, **cfg[media].fills.nightlight)
    plt.plot(df.date, df.dawn, lw=1, color=cfg.colours.twilight)
    plt.fill_between(df.date, df.dawn, df.sunrise, **cfg[media].fills.twilight)
    plt.plot(df.date, df.sunrise, lw=2, color=cfg.colours.sunlight)
    plt.fill_between(df.date, df.sunrise, df.sunset, **cfg[media].fills.daylight)
    plt.plot(df.date, df.sunset, lw=2, color=cfg.colours.sunlight)
    plt.fill_between(df.date, df.sunset, df.dusk, **cfg[media].fills.twilight)
    plt.plot(df.date, df.dusk, lw=1, color=cfg.colours.twilight)
    plt.fill_between(df.date, df.dusk, SECONDS_IN_A_DAY, **cfg[media].fills.nightlight)

    if df_highlights is not None:
        plt.plot(
            df_highlights.date,
            df_highlights.sunrise,
            linestyle="None",
            marker="o",
            color=cfg.colours.sunlight,
        )
        plt.plot(
            df_highlights.date,
            df_highlights.sunset,
            linestyle="None",
            marker="o",
            color=cfg.colours.sunlight,
        )
        for _idx, row in df_highlights.iterrows():
            ax.annotate(
                _time_formatter(row.sunrise, None),
                (row.date, row.sunrise),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="center",
                color=cfg.colours.sunlight,
            )
            ax.annotate(
                _time_formatter(row.sunset, None),
                (row.date, row.sunset),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="center",
                color=cfg.colours.sunlight,
            )

    if roc:
        sunrise_roc = df.sunrise.diff()
        sunrise_roc[sunrise_roc.abs() > 30 * SECONDS_IN_A_MINUTE] = pd.NA
        sunrise_roc_plot = sunrise_roc.abs() * 60
        sunset_roc = df.sunset.diff()
        sunset_roc[sunset_roc.abs() > 30 * SECONDS_IN_A_MINUTE] = pd.NA
        sunset_roc_plot = SECONDS_IN_A_DAY - sunset_roc.abs() * 60

        plt.plot(
            df.date, sunrise_roc_plot, color=cfg[media].contrast.nightlight, linestyle="dashed"
        )
        plt.plot(df.date, sunset_roc_plot, color=cfg[media].contrast.nightlight, linestyle="dashed")

        if df_highlights is not None:
            df_highlights["sunrise_roc"] = sunrise_roc.loc[df_highlights.index]
            df_highlights["sunrise_roc_plot"] = sunrise_roc_plot.loc[df_highlights.index]
            df_highlights["sunset_roc"] = sunset_roc.loc[df_highlights.index]
            df_highlights["sunset_roc_plot"] = sunset_roc_plot.loc[df_highlights.index]

            plt.plot(
                df_highlights.date,
                df_highlights.sunrise_roc_plot,
                linestyle="None",
                marker="o",
                color=cfg[media].contrast.nightlight,
            )
            plt.plot(
                df_highlights.date,
                df_highlights.sunset_roc_plot,
                linestyle="None",
                marker="o",
                color=cfg[media].contrast.nightlight,
            )

            for _idx, row in df_highlights.iterrows():
                ax.annotate(
                    _delta_time_formatter(row.sunrise_roc, None),
                    (row.date, row.sunrise_roc_plot),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    color=cfg[media].contrast.nightlight,
                )
                ax.annotate(
                    _delta_time_formatter(row.sunset_roc, None),
                    (row.date, row.sunset_roc_plot),
                    xytext=(0, -15),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    color=cfg[media].contrast.nightlight,
                )

    plt.xlim(start_date, end_date)
    plt.ylim(0, SECONDS_IN_A_DAY)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("1 %b '%y"))

    hourly = np.linspace(0, SECONDS_IN_A_DAY, 24 + 1)
    two_hourly = np.linspace(0, SECONDS_IN_A_DAY, 12 + 1)
    ax.yaxis.set_major_locator(ticker.FixedLocator(two_hourly))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(hourly))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_hour_formatter))

    plt.grid(True, "minor", "both", zorder=1000, alpha=cfg[media].grid_alpha, c="0.8")
    plt.grid(True, "major", "both", zorder=1001, alpha=cfg[media].grid_alpha, c="0.5")

    nighttime = mpatches.Patch(**cfg[media].fills.nightlight, label="Darkness")
    twilight = mpatches.Patch(**cfg[media].fills.twilight, label="Civil Twilight")
    daytime = mpatches.Patch(**cfg[media].fills.daylight, label="Daylight")
    sunrise_sunset = Line2D([0], [0], color=cfg.colours.sunlight, label="Sunrise/Sunset")
    solstice_equinox = Line2D(
        [0], [0], color=cfg.colours.sunlight, marker="o", label="Solstice/Equinox"
    )
    handles = [nighttime, twilight, daytime, solstice_equinox, sunrise_sunset]
    rate_of_change = Line2D(
        [0],
        [0],
        color=cfg[media].contrast.nightlight,
        linestyle="-",
        label="Rate of Change of Sunrise/Sunset",
    )
    handles = [nighttime, twilight, daytime, solstice_equinox, sunrise_sunset, noon, rate_of_change]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.925), ncol=len(handles))

    plt.xlabel("Date")
    plt.ylabel("Local Time")
    plt.title(f"Sun Graph - {location.name}", size=18, y=1.07)

    plt.tight_layout()

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
    )

    df = sun_times(location, START_DATE, END_DATE, recalculate=recalculate)

    events = [str(event) for event in cfg.events]
    df_events = df[["date", "sunrise", "sunset"]]
    df_events = df_events[df.date.dt.strftime(ISO_DATE_FORMAT).isin(events)]

    plot_sun_times(location, df, START_DATE, END_DATE, "display", df_events, roc=True)
    plot_sun_times(location, df, START_DATE, END_DATE, "print", df_events, roc=True)


if __name__ == "__main__":
    main()
