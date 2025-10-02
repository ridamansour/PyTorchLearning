from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def get_progress_data(path: Path = "./progress.parquet") -> pd.DataFrame:
    """
    This function reads the `progress.parquet` file into a pandas DataFrame.
    :param path: The path to the `progress.parquet` file
    :return: The pandas DataFrame
    """
    df = pd.read_parquet(path)
    df["section_num"] = df["section_num"].astype("Int64")
    df["Vid idx"] = df["Vid idx"].astype("Int64")
    df["title"] = df["title"].astype("string")
    df["duration"] = pd.to_timedelta(df["duration"])
    df["done"] = df["done"].astype("boolean")
    df["finished_date"] = pd.to_datetime(df["finished_date"])
    return df
def update_progress(video_index:int, done:bool, date: pd.Timestamp = pd.Timestamp.now()):
    """
    This function updates the progress of the course to the `progress.parquet` file.
    :param video_index: The index of the video
    :param date: When the video is finished (task done)
    :param done: True if the video is done
    :return: None
    """

    df = pd.get_progress_data("./progress.parquet")
    mask = (df["Vid idx"] == video_index)
    df.loc[mask, "done"] = done
    status = "done" if df.loc[mask, "done"].values[0] else "tasked"
    df.loc[mask, "finished_date"] = date if done else pd.NaT
    df.to_parquet("./progress.parquet")
    print(f"Updated progress report. \n"
          f"Section: {df.loc[mask].values[0][0]} \n"
          f"Video: {df.loc[mask].values[0][1]}. {df.loc[mask].values[0][2]} \n"
          f"Status: {status}")

def monthly_progress() -> None:
    """
    This function creates a monthly progress report showing histogram.
    :return: None
    """
    df = get_progress_data()

    # Create month column
    df["month"] = df["finished_date"].dt.to_period("M")

    # Group by month and sum durations
    monthly_time = df.groupby("month")["duration"].sum()

    # Convert timedelta to hours
    monthly_time_hours = monthly_time.dt.total_seconds() / 3600

    # Plot
    plt.figure(figsize=(8, 5))
    monthly_time_hours.plot(kind="bar", color="skyblue")
    plt.ylabel("Total Time (hours)")
    plt.xlabel("Month")
    plt.title("Time Spent Per Month")
    plt.xticks(rotation=45)
    # Format y-axis labels nicely
    formatter = FuncFormatter(lambda x, pos: f"{x:.1f}h")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()

def done_progress_pie_chart() -> None:
    """
    This function creates a pie chart of the progress of the course.
    :return: None
    """
    df = get_progress_data()

    done_sum: pd.Timedelta = df.loc[df["done"] == True, "duration"].sum()
    not_done_sum: pd.Timedelta = df.loc[df["done"] == False, "duration"].sum()

    times = [done_sum.total_seconds() / 3600, not_done_sum.total_seconds() / 3600]
    labels = [
        f"Done ({times[0]:.1f}h)",
        f"Not Done ({times[1]:.1f}h)"
    ]

    plt.figure(figsize=(6, 6))
    plt.pie(
        times,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#FF6F61"]
    )
    plt.title("Proportion of Time: Watched vs Not Watched")
    plt.show()

def _format_duration(td: timedelta) -> str:
    """
    Makes a string into a timedelta object.
    :param td: timedelta
    :return: The formatted string
    """
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{f'{days}d ' if days!=0 else ''}{f'{hours}h ' if hours!=0 else ''}{f'{minutes}m' if minutes!=0 else ''}"

def progress_in_a_section(section: int) -> None:
    """
    This function shows the progress of a section in the course using a progress bar.
    :param section: The index of the section
    :return: None
    """
    df = get_progress_data()

    mask_done = (df["section_num"] == section) & (df["done"] == True)
    mask_remaining = (df["section_num"] == section) & (df["done"] == False)

    total_videos = (mask_done | mask_remaining).sum()
    remaining_videos = mask_remaining.sum()
    done_videos = total_videos - remaining_videos

    total_duration = df.loc[mask_remaining, "duration"].sum()
    formatted = _format_duration(total_duration)

    # Section header
    if remaining_videos > 0:
        print(f"\nSection {section}: {formatted}, {remaining_videos} videos remaining")
    else:
        print(f"\nSection {section}: Done")

    # Progress bar
    if total_videos > 0:
        tqdm.write(f"Section {section}: " +
                   tqdm.format_meter(done_videos,
                                     total_videos,
                                     0,
                                     ncols=40,
                                     bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}",
                                     colour="BLUE")
                   )
def progress_report() -> None:
    """
    This function shows the progress report of the course with progress bars.
    :return:
    """
    for section in range(1, 15):
        progress_in_a_section(section)