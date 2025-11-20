import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

# Paths to folders
raw_data_folder = #add your path ... /raw-data'
results_folder = #add your path ... /results/'

# Parameters
participants = ['sub01', 'sub02', 'sub03']
tasks = ['sound', 'silent']
sessions = ['first_converted', 'second_converted']

all_results = []

for pi in participants:
    for ti in tasks:
        for si in sessions:
            # Assemble file name
            filename = os.path.join(raw_data_folder, f'{pi}_{ti}_{si}_eda.csv')
            print('Reading in ' + filename)

            try:
                # Read in the data in respective conditions
                subdata = pd.read_csv(filename, header=None, names=['EDA'], skiprows=1)
                eda_data = subdata['EDA'].values[30 * 1000:]  # Cutting the first 30 seconds

                # Downsample from 1000 Hz to 10 Hz
                downsample_factor = 100
                eda_data_downsampled = eda_data[::downsample_factor]

                # Process EDA signal using NeuroKit2
                signals_full, info = nk.eda_process(eda_data_downsampled, sampling_rate=10)
                print(f"Number of detected SCR Peaks: ", len(info['SCR_Peaks']))

                # Select segment to plot (last 30 seconds)
                eda_signals = signals_full


                # Plot the last 30 seconds of data
                nk.eda_plot(eda_signals, info)

                # Customize plot
                fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(20, 12))
                plt.subplots_adjust(hspace=0.5)

                # Set global title
                fig.suptitle(f"Electrodermal Activity (EDA) - {pi}, {ti}, {si}", fontweight="bold", fontsize=16)

                # Plot raw and cleaned signals
                ax0.set_title("Raw and Cleaned Signal", fontsize=12)
                ax0.set_ylim(0, 17)
                ax0.plot(eda_signals.index, eda_signals["EDA_Raw"], color="#B0BEC5", label="Raw")
                ax0.plot(eda_signals.index, eda_signals["EDA_Clean"], color="#9C27B0", label="Cleaned", linewidth=1.5)
                ax0.legend(loc="upper right")
                ax0.set_ylabel("EDA Amplitude (µS)", fontsize=10)
                ax0.set_xlabel("Time (mS)", fontsize=10)  # Added x-axis label

                # Plot phasic component
                ax1.set_title("Skin Conductance Response (SCR)", fontsize=12)
                ax1.set_ylim(-0.8, 1)
                ax1.plot(eda_signals.index, eda_signals["EDA_Phasic"], color="#145DA0", label="Phasic Component",
                         linewidth=1.5)
                ax1.legend(loc="upper right")
                ax1.set_ylabel("Phasic EDA Amplitude (µS)", fontsize=10)
                ax1.set_xlabel("Time (ms)", fontsize=10)  # Added x-axis label

                # Add markers for SCR events
                onsets = eda_signals.index[eda_signals["SCR_Onsets"] == 1]
                peaks = eda_signals.index[eda_signals["SCR_Peaks"] == 1]
                ax1.scatter(onsets, eda_signals.loc[onsets, "EDA_Phasic"], color="#628810", label="Onsets", zorder=2)
                ax1.scatter(peaks, eda_signals.loc[peaks, "EDA_Phasic"], color="#3C6007", label="Peaks", zorder=2)
                ax1.legend(loc="upper right")

                # Plot tonic component
                ax2.set_title("Skin Conductance Level (SCL)", fontsize=12)
                ax2.set_ylim(0, 17)
                ax2.plot(eda_signals.index, eda_signals["EDA_Tonic"], color="#145DA0", label="Tonic Component",
                         linewidth=1.5)
                ax2.legend(loc="upper right")
                ax2.set_xlabel("Time (s)", fontsize=10)
                ax2.set_ylabel("Tonic EDA Amplitude (µS)", fontsize=10)
                ax2.set_xlabel("Time (ms)", fontsize=10)  # Added x-axis label

                figure_filename = os.path.join(results_folder, f'{pi}_{ti}_{si}_eda_processed1.png')
                plt.savefig(figure_filename, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                print(f"Saved processed EDA figure to {figure_filename}")

                # Compute interval-related EDA metrics for full signal
                eda_results = nk.eda_intervalrelated(eda_signals, sampling_rate=10)

                # Add metadata
                eda_results['Participant'] = pi
                eda_results['Condition'] = f'{ti}_{si}'
                all_results.append(eda_results)

            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {str(e)}")

# Concatenate all the results into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)

# Save the concatenated DataFrame to a CSV file
output_filename = os.path.join(results_folder, 'eda_results_processed1.csv')
final_results.to_csv(output_filename, index=False)

print(f"Saved processed EDA results to {output_filename}")
