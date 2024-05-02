import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.signal import find_peaks
import scipy.stats as stats

def compute_ci(data, confidence=0.95):
    """ Computes confidence interval around the mean of the data """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - margin, mean + margin

file_path = 'all_data.xlsx'
excel_data = pd.ExcelFile(file_path)
axis = "Az"
peak_totals = []

for freq in [100, 200, 300]:
    plt.figure(figsize=(10, 5))
    max_y = 0

    for bolts, style, color in zip(["tightened", "loosened"], ["b-", "r-"], ["blue", "red"]):
        column = f"Amplitude - {axis} - Bolts {bolts} - {freq}Hz"
        data = pd.read_excel(excel_data, sheet_name='Sheet1', usecols=[column])
        data_clean = data.dropna()[600:]

        amplitude_data = data_clean.values.flatten()

        # Split the signal into segments
        segment_num = 4
        segment_size = len(amplitude_data) // segment_num  # Adjust the number of segments as needed
        segments = [amplitude_data[i:i + segment_size] for i in range(0, len(amplitude_data) - segment_size + 1, segment_size)]

        # Compute FFT for each segment
        start_cutoff = 1

        frequencies = np.fft.fftfreq(n=len(amplitude_data) // segment_num, d=1/1325)
        fft_results = [np.abs(np.fft.fft(seg)[frequencies >= 0][start_cutoff:]) for seg in segments]
        frequencies = frequencies[frequencies >= 0][start_cutoff:]

        # Compute mean and confidence intervals
        mean_fft = np.mean(fft_results, axis=0)
        lower_ci, upper_ci = np.zeros(len(mean_fft)), np.zeros(len(mean_fft))

        for i in range(len(mean_fft)):
            _, lower_ci[i], upper_ci[i] = compute_ci([fft[i] for fft in fft_results])

        plt.plot(frequencies, mean_fft, style, label=f"Bolts {bolts}")
        peaks, _ = find_peaks(mean_fft)
        sorted_peaks = sorted(peaks, key=lambda x: mean_fft[x], reverse=True)
        top_5_peaks = sorted_peaks[:5]

        plt.plot(frequencies[top_5_peaks], mean_fft[top_5_peaks], "o", color=color)
        peak_total = sum(mean_fft[top_5_peaks])
        print(peak_total)
        peak_totals.append((bolts, freq, peak_total))

        # CREATE A BAR GRAPH OF THIS peak_total VALUE

        plt.fill_between(frequencies, lower_ci, upper_ci, alpha=0.2, color=color)

        max_y = max(max_y, np.max(mean_fft))

    title = f"Fourier Transform with {freq}Hz excitation"
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 650)
    plt.ylim(0, max_y * 1.1)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.savefig(f'figures/{title}.png')

def plot_peak_totals(peak_totals):
    """Plots the given peak totals as paired bar charts with appropriate colors and a legend."""
    # Prepare the bar chart data
    pairs = [(peak_totals[i][2], peak_totals[i+1][2]) for i in range(0, len(peak_totals), 2)]
    x_labels = [f"{peak_totals[i][1]} Hz\n({peak_totals[i][0]} vs {peak_totals[i+1][0]})" for i in range(0, len(peak_totals), 2)]

    plt.figure(figsize=(10, 6))

    bar_width = 0.3
    index = np.arange(len(pairs))

    # Create bars
    for i, pair in enumerate(pairs):
        plt.bar(index[i] - bar_width/2, pair[0], bar_width, color="blue", label="Tightened" if peak_totals[i*2][0] == "tightened" else "red")
        plt.bar(index[i] + bar_width/2, pair[1], bar_width, color="red", label="Loosened" if peak_totals[i*2+1][0] == "loosened" else "blue")

    # Ensure only unique labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    title = "Comparison of Peak Magnitude for Tightened vs Loosened Bolts"
    plt.xticks(index, x_labels, rotation=45, ha="right")
    plt.ylabel("Total Peak Magnitude")
    plt.title(title)
    plt.legend(unique_handles, unique_labels)
    plt.tight_layout()

    plt.show(block=False)
    plt.savefig(f'figures/{title}.png')


plot_peak_totals(peak_totals)


input("Press Enter to finish")
sys.exit()
