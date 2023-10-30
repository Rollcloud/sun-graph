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

START_DATE = datetime(2023, 10, 30)
END_DATE = datetime(2024, 10, 27)

SECONDS_IN_A_MINUTE = 60
SECONDS_IN_A_HOUR = SECONDS_IN_A_MINUTE * 60
SECONDS_IN_A_DAY = SECONDS_IN_A_HOUR * 24

DAYLIGHT = "#ffe577"
SUNLIGHT = "#f15e5c"
TWILIGHT = "#6279b8"
NIGHTLIGHT = "#4a6194"

ISO_DATE_FORMAT = "%Y-%m-%d"

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


def _hour_formatter(seconds_in, _position):
    seconds_in = int(seconds_in)
    hours = seconds_in // SECONDS_IN_A_HOUR
    minutes = (seconds_in - (hours * SECONDS_IN_A_HOUR)) // SECONDS_IN_A_MINUTE
    return f"{hours:02d}h{minutes:02d}"


def plot_sun_times(location, df, start_date, end_date, df_highlights=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.fill_between(df.date, 0, df.dawn, color=NIGHTLIGHT)
    plt.fill_between(df.date, df.dawn, df.sunrise, color=TWILIGHT)
    plt.plot(df.date, df.sunrise, lw=2, color=SUNLIGHT)
    plt.fill_between(df.date, df.sunrise, df.sunset, color=DAYLIGHT)
    plt.fill_between(df.date, df.sunset, df.dusk, color=TWILIGHT)
    plt.plot(df.date, df.sunset, lw=2, color=SUNLIGHT)
    plt.fill_between(df.date, df.dusk, SECONDS_IN_A_DAY, color=NIGHTLIGHT)

    if df_highlights is not None:
        plt.plot(
            df_highlights.date, df_highlights.sunrise, linestyle="None", marker="o", color=SUNLIGHT
        )
        plt.plot(
            df_highlights.date, df_highlights.sunset, linestyle="None", marker="o", color=SUNLIGHT
        )
        for _idx, row in df_highlights.iterrows():
            ax.annotate(
                _hour_formatter(row.sunrise, None),
                (row.date, row.sunrise),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="center",
                color=SUNLIGHT,
            )
            ax.annotate(
                _hour_formatter(row.sunset, None),
                (row.date, row.sunset),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="center",
                color=SUNLIGHT,
            )

    plt.xlim(start_date, end_date)
    plt.ylim(0, SECONDS_IN_A_DAY)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))

    hourly = np.linspace(0, SECONDS_IN_A_DAY, 24 + 1)
    two_hourly = np.linspace(0, SECONDS_IN_A_DAY, 12 + 1)
    ax.yaxis.set_major_locator(ticker.FixedLocator(two_hourly))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(hourly))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_hour_formatter))

    plt.grid(True, "both", "both", zorder=1000, alpha=0.35)

    nighttime = mpatches.Patch(color=NIGHTLIGHT, label="Darkness")
    twilight = mpatches.Patch(color=TWILIGHT, label="Civil Twilight")
    daytime = mpatches.Patch(color=DAYLIGHT, label="Daylight")
    sunrise_sunset = Line2D([0], [0], color=SUNLIGHT, label="Sunrise/Sunset")
    solstice_equinox = Line2D([0], [0], color=SUNLIGHT, marker="o", label="Solstice/Equinox")
    handles = [nighttime, twilight, daytime, solstice_equinox, sunrise_sunset]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.925), ncol=len(handles))

    plt.xlabel("Date")
    plt.ylabel("Local Time")
    plt.title(f"Sun Graph - {location.name}", size=18, y=1.07)

    plt.tight_layout()

    location_name = _clean_name(location.name)
    plt.savefig(p / "tmp" / f"sun-graph_{location_name}.png")
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
    df_highlights = df[["date", "sunrise", "sunset"]][
        df.date.dt.strftime(ISO_DATE_FORMAT).isin(
            [
                "2023-12-22",
                "2024-03-20",
                "2024-06-20",
                "2024-09-22",
                "2024-12-21",
            ]
        )
    ]
    plot_sun_times(location, df, START_DATE, END_DATE, df_highlights)


if __name__ == "__main__":
    main()
