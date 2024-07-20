import pandas as pd
import numpy as np
import glob
import ast
import logging
from tqdm import tqdm
import argparse

# Function to safely evaluate lists
def safe_eval(col):
    try:
        return ast.literal_eval(col)
    except ValueError:
        return col  # Return as is if it's not reftraj string representation of reftraj list

def process_trajectory_data(input_folder='data', output_file='processed_trajectory_data.csv', num_robots=5):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize an empty DataFrame
    all_data = pd.DataFrame()

    # Generate robot indices based on num_robots
    robot_indices = list(range(num_robots))

    # Use glob to find all CSV files in the specified folder
    file_pattern = f"{input_folder}/trajectory_data_*.csv"
    file_list = glob.glob(file_pattern)
    
    # Loop through the files sorted to maintain the order
    for filename in tqdm(sorted(file_list), desc="Processing files"):
        logging.info(f"Reading file {filename}")
        temp_df = pd.read_csv(filename)

        # Filter rows by robot_index
        temp_df = temp_df[temp_df['robot_index'].isin(robot_indices)]

        # Apply transformations
        temp_df['joint_positions'] = temp_df['joint_positions'].apply(safe_eval)
        temp_df['joint_velocities'] = temp_df['joint_velocities'].apply(safe_eval)
        
        all_data = pd.concat([all_data, temp_df], ignore_index=True)

    # Additional data manipulation and processing
    # Now create the derived columns
    all_data['x_t'] = all_data.apply(lambda row: row['joint_positions'] + row['joint_velocities'], axis=1)
    all_data['u_t'] = all_data.apply(lambda row: [row['velocity_x'], row['velocity_y']], axis=1)
    all_data['z_t'] = all_data.apply(lambda row: [row['traj_x'], row['traj_y']], axis=1)
    all_data['v_t'] = all_data.apply(lambda row: [row['reduced_command_x'], row['reduced_command_y']], axis=1)

    # Calculate the vector differences without norm for w_xy_t and shift it for w_xy_{t+1}
    all_data['w_xy_t'] = all_data.apply(lambda row: [row['position_x'] - row['traj_x'], row['position_y'] - row['traj_y']], axis=1)
    all_data['group'] = all_data['episode_number'].astype(str) + '_' + all_data['robot_index'].astype(str)

    # Original calculation for w_t for reference
    all_data['w_t'] = np.sqrt((all_data['position_x'] - all_data['traj_x'])**2 + (all_data['position_y'] - all_data['traj_y'])**2)

    # Shift operations for next timestep values
    all_data['x_{t+1}'] = all_data.groupby('group')['x_t'].shift(-1)
    all_data['z_{t+1}'] = all_data.groupby('group')['z_t'].shift(-1)
    all_data['w_{t+1}'] = all_data.groupby('group')['w_t'].shift(-1)
    all_data['w_xy_{t+1}'] = all_data.groupby('group')['w_xy_t'].shift(-1)

    # Function to drop the first and last 10 data points from each episode
    def drop_edges(group):
        return group.iloc[10:-10]
    all_data = all_data.groupby('group', group_keys=False).apply(drop_edges)

    # Drop rows where x_{t+1}, z_{t+1}, w_{t+1}, and w_xy_{t+1} do not exist
    all_data.dropna(subset=['x_{t+1}', 'z_{t+1}', 'w_{t+1}', 'w_xy_{t+1}'], inplace=True)
    
    # Select and order final columns for saving
    final_df = all_data[['group', 'x_t', 'u_t', 'z_t', 'v_t', 'w_t', 'w_xy_t', 'x_{t+1}', 'z_{t+1}', 'w_{t+1}', 'w_xy_{t+1}']]
    final_df.to_csv(output_file, index=False)
    logging.info(f"Data processed and saved to {output_file}")
    return final_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process trajectory data files.')
    parser.add_argument('--input_folder', type=str, default='data', help='Folder containing CSV files (default: data)')
    parser.add_argument('--output_file', type=str, default='processed_trajectory_data.csv', help='Output CSV file path (default: processed_trajectory_data.csv)')
    parser.add_argument('--num_robots', type=int, default=5, help='Number of robots to include in processing (default: 5)')
    
    args = parser.parse_args()
    
    # Call the function with command line arguments
    process_trajectory_data(args.input_folder, args.output_file, args.num_robots)
