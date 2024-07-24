import os
import pandas as pd
import matplotlib.pyplot as plt

def clip_and_save_data(csv_file, output_dir, start_time=1.0, end_time=1.95):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Filter out rows with 'shoulder' and 'forearm' in the 'Joint' column
    data = data[~data['Joint'].str.contains('shoulder|forearm|foot|base', case=False)]

    # Clip the data to the specified time range
    clipped_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()

    # Reset the time to start from 0
    clipped_data['Time'] = clipped_data['Time'] - start_time

    # Generate the output file path
    base_name = os.path.basename(csv_file)
    output_file = os.path.join(output_dir, base_name)

    # Save the clipped data to the new file
    os.makedirs(output_dir, exist_ok=True)
    clipped_data.to_csv(output_file, index=False)

def plot_joint_data(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Extract unique joint names
    joints = data['Joint'].unique()

    # Initialize a figure for plotting
    plt.figure(figsize=(14, 8))
    plt.suptitle(os.path.basename(csv_file), fontsize=16)

    # Plot joint positions
    plt.subplot(2, 1, 1)
    for joint in joints:
        joint_data = data[data['Joint'] == joint]
        plt.plot(joint_data['Time'], joint_data['Position'], label=joint)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Joint Positions Over Time')
    plt.legend(loc='best')

    # Plot joint velocities
    plt.subplot(2, 1, 2)
    for joint in joints:
        joint_data = data[data['Joint'] == joint]
        plt.plot(joint_data['Time'], joint_data['Velocity'], label=joint)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('Joint Velocities Over Time')
    plt.legend(loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def process_and_plot_all_csvs(unprocessed_dir, processed_dir):
    # List all CSV files in the specified unprocessed directory
    csv_files = [f for f in os.listdir(unprocessed_dir) if f.endswith('.csv')]

    # Iterate through each CSV file, clip its data, save it, and plot it
    for csv_file in csv_files:
        csv_path = os.path.join(unprocessed_dir, csv_file)
        clip_and_save_data(csv_path, processed_dir)
        processed_csv_path = os.path.join(processed_dir, csv_file)
        plot_joint_data(processed_csv_path)

# Example usage
unprocessed_directory = 'adam_reference_trajectories/unprocessed'  # Replace with your unprocessed directory path
processed_directory = 'adam_reference_trajectories/processed'      # Replace with your processed directory path
process_and_plot_all_csvs(unprocessed_directory, processed_directory)
