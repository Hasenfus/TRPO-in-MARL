import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def find_train_episode_rewards_dirs(root_dir):
    """
    Finds and returns paths to directories named 'train_episode_rewards'
    within the given root directory.

    Parameters:
    - root_dir: The root directory to search within.

    Returns:
    - A list of paths to directories.
    """
    reward_dirs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.endswith('aver_rewards'):
            reward_dirs.append(dirpath)

    return reward_dirs


def load_event_data(event_dir, scalar_tag):
    """
    Loads scalar data for a given tag from an event file in a specified directory.

    Parameters:
    - event_dir: The directory containing the event file.
    - scalar_tag: The tag of the scalar data to load.

    Returns:
    - A pandas DataFrame containing the scalar data.
    """
    # Initialize an EventAccumulator instance
    event_acc = EventAccumulator(event_dir, size_guidance={'scalars': 0})

    # Load all events from the directory
    event_acc.Reload()

    # Check if the scalar tag is available
    if scalar_tag in event_acc.Tags()['scalars']:
        # Get scalar data
        scalar_data = event_acc.Scalars(scalar_tag)

        # Convert to pandas DataFrame
        df = pd.DataFrame(scalar_data)
        df = df.drop(['wall_time'], axis=1)  # Drop wall time if not needed
        df.columns = ['step', 'value']  # Rename columns for clarity

        return df
    else:
        print(f"Tag '{scalar_tag}' not found in {event_dir}.")
        return pd.DataFrame()


# Replace 'mlp' with the path to your actual directory

def retreive_data_df(reward_dirs, scalar_tag):
    data_frames = {}
    for reward_dir in reward_dirs:
        df = load_event_data(reward_dir, scalar_tag)
        if not df.empty:
            data_frames[reward_dir] = df

    return data_frames

    # At this point, you have a dictionary `data_frames` with paths as keys and data as values
    # You can convert these to numpy arrays or save them as needed
def retreive_data_np(reward_dirs, scalar_tag):
    data_frames = {}
    for reward_dir in reward_dirs:
        df = load_event_data(reward_dir, scalar_tag)
        if not df.empty:
            data_frames[reward_dir] = df
    np_arrays = []
    for path, df in data_frames.items():
        np_arrays.append(df['value'].to_numpy())

    return np_arrays

def calculate_mean_and_confidence_interval(data):
    """
    Calculate the mean and 95% confidence interval for each timestep.

    :param data: A list of lists, where each inner list represents a run and contains values for each timestep.
    :return: A tuple of two numpy arrays - one for the mean and one for the 95% confidence interval.
    """
    data = np.array(data)
    mean = np.mean(data, axis=0)
    stderr = stats.sem(data, axis=0, nan_policy='omit')
    confidence_interval = stderr * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)

    return mean, confidence_interval

def smooth_data(data, smoothing_factor=0.0):
    """
    Apply simple moving average smoothing to the data.
    :param data: Array of data points.
    :param smoothing_factor: Between 0 (no smoothing) and 1 (maximum smoothing).
    :return: Smoothed data.
    """
    if smoothing_factor <= 0:
        return data  # No smoothing
    window_size = int(len(data) * smoothing_factor) or 1
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def average_and_confidence(rewards):
    """
    Calculate the average and 95% confidence interval for rewards at each timestep.

    :param output of get_rewards_for_last_n_runs
    :return: A tuple of four numpy arrays - mean rewards, confidence interval for rewards, mean timesteps, confidence interval for timesteps.
    """
    mean_rewards, conf_rewards = calculate_mean_and_confidence_interval(rewards)
    # mean_timesteps, conf_timesteps = calculate_mean_and_confidence_interval(timesteps)

    return mean_rewards, conf_rewards #, mean_timesteps, conf_timesteps


def plot_with_confidence_interval(mean_values, confidence_interval, timesteps, title="Plot with Confidence Interval", xlabel="Timestep", ylabel="Value",
                                  ylim=None, smoothing_factor=0.0, save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', legend=True, show_title=True, fig_size=(4, 3)):
    """
    Plot mean values with confidence interval, with optional smoothing.

    :param mean_values: Array of mean values.
    :param confidence_interval: Array of confidence interval values.
    :param timesteps: Array of timesteps.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param smoothing_factor: Smoothing factor between 0 and 1.
    """
    smoothed_means = smooth_data(mean_values, smoothing_factor)
    smoothed_ci = smooth_data(confidence_interval, smoothing_factor)

    upper_bound = smoothed_means + smoothed_ci
    lower_bound = smoothed_means - smoothed_ci

    fig, ax = plt.subplots(figsize=fig_size)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines darker
    ax.spines['left'].set_color('black')  # You can specify a hex code for colors as well, e.g., '#000000'
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid behind the plot
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    # Send grid to the back
    ax.set_axisbelow(True)

    # Update the rcParams for this figure specifically
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)

    # Plot data
    ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label="Mean", color="blue")
    ax.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, color="blue", alpha=0.2,
                    label="95% Confidence Interval")

    # Setting titles and labels
    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    # Display legend
    if legend:
        ax.legend()

    # Save the figure if required
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)

    # Show the plot
    plt.show()
def plot_multiple_with_confidence_intervals(mean_values_list, confidence_intervals_list, timesteps, labels, title="Comparison Plot", xlabel="Timestep", ylabel="Value",
                                            save=False, legend = True,  save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', ylim=None, smoothing_factor=0.0, show_title=True, fig_size=(4, 3)):
    """
    Plot multiple sets of mean values with their confidence intervals, with optional smoothing.

    :param mean_values_list: List of arrays of mean values for each algorithm.
    :param confidence_intervals_list: List of arrays of confidence intervals for each algorithm.
    :param timesteps: Array of timesteps.
    :param labels: List of labels for each algorithm.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param smoothing_factor: Smoothing factor between 0 and 1.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Update the rcParams for this figure specifically
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines darker
    ax.spines['left'].set_color('black')  # You can specify a hex code for colors as well, e.g., '#000000'
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid behind the plot
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    # Send grid to the back
    ax.set_axisbelow(True)

    # Update the rcParams for this figure specifically
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)

    # Plotting mean values with confidence intervals
    for mean_values, confidence_interval, label in zip(mean_values_list, confidence_intervals_list, labels):
        smoothed_means = smooth_data(mean_values, smoothing_factor)
        smoothed_ci = smooth_data(confidence_interval, smoothing_factor)

        upper_bound = smoothed_means + smoothed_ci
        lower_bound = smoothed_means - smoothed_ci

        ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label=f"Mean - {label}")
        ax.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, alpha=0.2,
                        label=f"95% CI - {label}")

    # Setting titles and labels
    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjusting plot limits if specified
    if ylim:
        ax.set_ylim(ylim)

    # Adding legend if specified
    if legend:
        ax.legend()

    # Save the figure if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)

    # Show the plot
    plt.show()


def plot_multiple(mean_values_list, timesteps, labels, title="Comparison Plot", xlabel="Timestep", ylabel="Value",
                                            save=False, legend = True,  save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', ylim=None, smoothing_factor=0.0, show_title=True, fig_size=(4, 3)):
    """
    Plot multiple sets of mean values with their confidence intervals, with optional smoothing.

    :param mean_values_list: List of arrays of mean values for each algorithm.
    :param confidence_intervals_list: List of arrays of confidence intervals for each algorithm.
    :param timesteps: Array of timesteps.
    :param labels: List of labels for each algorithm.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param smoothing_factor: Smoothing factor between 0 and 1.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Update the rcParams for this figure specifically
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines darker
    ax.spines['left'].set_color('black')  # You can specify a hex code for colors as well, e.g., '#000000'
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid behind the plot
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    # Send grid to the back
    ax.set_axisbelow(True)

    # Update the rcParams for this figure specifically
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)

    # Plotting mean values with confidence intervals
    for mean_values, label in zip(mean_values_list, labels):
        smoothed_means = smooth_data(mean_values, smoothing_factor)
        ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label=f"Mean - {label}")

    # Setting titles and labels
    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjusting plot limits if specified
    if ylim:
        ax.set_ylim(ylim)

    # Adding legend if specified
    if legend:
        ax.legend()

    # Save the figure if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)

    # Show the plot
    plt.show()


import os
import shutil


def replicate_structure_and_copy(src_root, dst_root):
    """
    Replicates the directory structure from src_root to dst_root and copies
    all contents of 'train_episode_rewards' directories.

    Parameters:
    - src_root: The source root directory to search within.
    - dst_root: The destination root directory to copy to.
    """
    for dirpath, _, filenames in os.walk(src_root):
        if 'train_episode_rewards' in dirpath:
            relative_path = os.path.relpath(dirpath, src_root)
            dst_path = os.path.join(dst_root, relative_path)

            # Ensure the destination directory exists
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                print(f"Created directory: {dst_path}")

            # Copy each file
            for filename in filenames:
                src_file_path = os.path.join(dirpath, filename)
                dst_file_path = os.path.join(dst_path, filename)

                # Copy file and keep metadata
                shutil.copy2(src_file_path, dst_file_path)
                print(f"Copied file: {src_file_path} to {dst_file_path}")


def stacked_bar_graph(*tuples):
    averages = []
    labels = []

    # Compute averages for each array in the tuples
    for i, arrays in enumerate(tuples):
        tuple_averages = []
        for arr in arrays:
            avg = np.mean(arr)
            tuple_averages.append(avg)
        averages.append(tuple_averages)
        labels.append(f"Tuple {i+1}")

    # Compute the total sum of averages for each tuple
    totals = [sum(avg_list) for avg_list in averages]

    # Compute the percentages for each array within the tuples
    percentages = []
    for avg_list in averages:
        tuple_percentages = [avg / sum(avg_list) * 100 for avg in avg_list]
        percentages.append(tuple_percentages)

    # Set up the plot
    fig, ax = plt.subplots()
    bar_width = 0.8 / len(tuples)
    x = np.arange(len(tuples))

    # Create stacked bars for each tuple
    bottom = np.zeros(len(tuples))
    for i in range(len(percentages[0])):
        values = [tuple_percentages[i] for tuple_percentages in percentages]
        ax.bar(x, values, bar_width, bottom=bottom, label=f"Array {i+1}")
        bottom += values

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage")
    ax.set_title("Stacked Bar Graph of Array Averages")
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()