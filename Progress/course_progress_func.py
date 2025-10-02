from datetime import timedelta
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from pyfiglet import Figlet

PROGRESS_PARQUET_PATH = Path("/Users/ridamansour/DataspellProjects/PyTorchLearning/Progress/progress.parquet")

def get_progress_data(path: Path = PROGRESS_PARQUET_PATH) -> pd.DataFrame:
    """
    This function reads the `progress.parquet` file into a pandas DataFrame.
    :param path: The path to the `progress.parquet` file
    :return: The pandas DataFrame
    """
    path = Path(path)
    df = pd.read_parquet(path)
    df["section_num"] = df["section_num"].astype("Int64")
    df["Vid idx"] = df["Vid idx"].astype("Int64")
    df["title"] = df["title"].astype("string")
    df["duration"] = pd.to_timedelta(df["duration"])
    df["done"] = df["done"].astype("boolean")
    df["finished_date"] = pd.to_datetime(df["finished_date"])
    return df

def _format_duration(td: timedelta) -> str:
    """
    Makes a string into a timedelta object.
    :param td: timedelta
    :return: The formatted string
    """
    # days = td.days*24
    hours, remainder = divmod(td.seconds, 3600)
    hours += td.days*24
    minutes, _ = divmod(remainder, 60)
    # {f'{days}d ' if days!=0 else ''}
    return f"{f'{hours}h ' if hours!=0 else ''}{f'{minutes}m' if minutes!=0 else ''}"

def progress_bar(x: int, total: int):
    """
    A function that returns a progress bar.
    :param x: the progress (n)
    :param total: the total
    :return: A progress bar
    """
    return tqdm.format_meter(
        n=x,
        total=total,
        elapsed=0,
        ncols=40,
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}",
        colour="BLUE"
    )

def progress_in_a_section(section: int, show_section_num:bool=True) -> str:
    """
    Returns a string showing the progress of a section in the course using a progress bar.
    :param section: The index of the section
    :param show_section_num: Whether to show the section number (i.e. `Section 3:`)
    :return: str
    """
    df = get_progress_data()

    mask_done = (df["section_num"] == section) & (df["done"] == True) & (df["Vid idx"] != -1)
    mask_remaining = (df["section_num"] == section) & (df["done"] == False) & (df["Vid idx"] != -1)

    total_videos = (mask_done | mask_remaining).sum()
    remaining_videos = mask_remaining.sum()
    done_videos = total_videos - remaining_videos

    total_duration = df.loc[mask_remaining, "duration"].sum()
    formatted = _format_duration(total_duration)
    header = ""
    # Section header
    section_title = df.loc[(df["section_num"] == section) & (df["Vid idx"] == -1), "title"].iloc[0]
    section_title = section_title[section_title.find(": ")+2:]
    if show_section_num:
        header = f"\n\033[1mSection :\033[0m {section}.{section_title}\n"

    if remaining_videos > 0:
        header += (f"\033[1mStatus  :\033[0m {remaining_videos} videos remaining, {formatted} to finish the section")
    else:
        header +=  f"\033[1mStatus  :\033[0m \033[92mDone\033[0m"

    # Progress bar
    if total_videos > 0:
        progress = progress_bar(done_videos, total_videos)
        progress_line = ""
        if show_section_num:
            progress_line = f"\033[1mProgress: \033[0m"
        progress_line += f"{progress}"
    else:
        progress_line = ""

    # Combine everything
    result = header + ("\n" + progress_line if progress_line else "")
    return result


def update_progress(video_index:int, done:bool, date: pd.Timestamp = pd.Timestamp.now()):
    """
    This function updates the progress of the course to the `progress.parquet` file.
    :param video_index: The index of the video
    :param date: When the video is finished (task done)
    :param done: True if the video is done
    :return: None
    """
    df = get_progress_data()
    mask = (df["Vid idx"] == video_index)
    df.loc[mask, "done"] = done
    status = "Done" if df.loc[mask, "done"].values[0] else "Not done"
    df.loc[mask, "finished_date"] = date if done else pd.NaT

    # Check if all videos in the section are done
    section_num = df.loc[mask, "section_num"].values[0]
    section_videos_mask = (df["section_num"] == section_num) & (df["Vid idx"] != -1)

    if df.loc[section_videos_mask, "done"].all():
        df.loc[(df["section_num"] == section_num) & (df["Vid idx"] == -1), "done"] = True # Check the title of the section

    df.to_parquet(PROGRESS_PARQUET_PATH)
    formatted_date = date.strftime("%d %b %Y %I:%M %p") if done else "N/A"

    print(f"\033[1m\033[92mUpdated progress report.\033[0m \n"
          f"\033[1mVideo:\033[0m {df.loc[mask].values[0][1]}. {df.loc[mask].values[0][2]} \n"
          f"\033[1mDuration:\033[0m {_format_duration(df.loc[mask].values[0][3])} \n"
          f"\033[1mStatus:\033[0m {status} \n"
          f"\033[1mDate:\033[0m {formatted_date} \n"
          # f"Section: {df.loc[mask].values[0][0]} \n"
          f"\033[1mSection progress:\033[0m {progress_in_a_section(df.loc[mask].values[0][0], True)}"
          )

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
    plt.show();

def progress_pie_chart() -> None:
    """
    This function creates a pie chart of the progress of the course.
    :return: None
    """
    df = get_progress_data()

    done_sum: pd.Timedelta = df.loc[df["done"] == True, "duration"].sum()
    not_done_sum: pd.Timedelta = df.loc[df["done"] == False, "duration"].sum()

    times = [done_sum.total_seconds() / 3600, not_done_sum.total_seconds() / 3600]
    labels = [
        f"Done\n({times[0]:.1f}h)",
        f"Not Done\n({times[1]:.1f}h)"
    ]

    plt.figure(figsize=(6, 6))
    plt.pie(
        times,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#FF6F61"],
        textprops = {"fontsize": 14}
    )
    plt.title("Course Progress: Watched vs Not Watched", fontsize=16)
    plt.show();




def progress_report() -> str:
    """
    This function shows the progress report of the course with progress bars.
    :return:
    """
    df = get_progress_data()

    done_time_sum: pd.Timedelta = df.loc[df["done"] == True, "duration"].sum()
    not_done_time_sum: pd.Timedelta = df.loc[df["done"] == False, "duration"].sum()
    done_videos_count = df.loc[(df["done"] == True) & df["Vid idx"] != -1].__len__()
    not_done_videos_count = df.loc[(df["done"] == False) & df["Vid idx"] != -1].__len__()
    total_videos = done_videos_count+not_done_videos_count
    report = ""
    f = Figlet(font="isometric3", width=120)
    report += f"\033[1m\033[94m{f.renderText('Progress Report')}\033[0m\n"
    report += f"\033[1m\033[93mCourse:\033[0m {progress_bar(done_videos_count, total_videos)}\n"
    report += f"\033[1m\033[93mTotal time done:\033[0m {_format_duration(done_time_sum)} out of {_format_duration(not_done_time_sum + done_time_sum)} watched ({_format_duration(not_done_time_sum)} remaining).\n"
    report += f"\033[1m\033[93mTotal videos done:\033[0m {done_videos_count} out of {total_videos} finished ({done_videos_count} videos remaining).\n"
    for section in range(1, 15):
        report += progress_in_a_section(section) + "\n"
    return report
def progress_report_print() -> None:
    print(progress_report())

def strip_ansi_codes(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    """
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)
def progress_report_to_README() -> None:
    report = progress_report()
    clean_report = f"```\nProgress.course_prog_func.progress_report() github hook \n{strip_ansi_codes(report)}```\n"

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(clean_report)

if __name__ == "__main__":
    monthly_progress()
    progress_report_print()
    progress_pie_chart()

