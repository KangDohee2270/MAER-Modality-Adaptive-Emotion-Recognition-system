import numpy as np
import scipy.signal as si

# from sklearn.preprocessing import MinMaxScaler

# ----------------------Initial setting----------------------
sampling_rate = 1024  # 측정주파수로 변경
lowcut = 0.4
highcut = 50


# ----------------------Data setting----------------------!
def data_choice(test_file):
    # 주파수 1000Hz, 저장된 데이터에서 10초를 추출(10000으로 변경 예정)
    x_data_zero = np.zeros((1, sampling_rate * 10))
    for k in range(sampling_rate * 10):
        x_data_zero[0][k] = test_file[0][(test_file[0].__len__() - sampling_rate * 10) + k]
    x_data = x_data_zero
    # x_data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x_data)
    return x_data


# ----------------------Data preprocessing----------------------

def ppg_cleaning(ppg_raw_signal):
    frequency, filtered = signal_filter(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate)
    sos = si.butter(5, frequency, btype=filtered, output="sos", fs=sampling_rate)
    ppg_signal = si.sosfiltfilt(sos, ppg_raw_signal)
    return ppg_signal


def gsr_cleaning(gsr_raw_signal, sampling_rate=sampling_rate, lowcut=None):
    freq, filter = signal_filter(lowcut=lowcut, highcut=3, sampling_rate=sampling_rate)
    sos = si.butter(5, freq, btype=filter, output="sos", fs=sampling_rate)
    gsr_signal = si.sosfiltfilt(sos, gsr_raw_signal)
    return gsr_signal


def signal_filter(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate, normalize=False):
    if lowcut is not None and highcut is not None:
        if lowcut > highcut:
            filtered = "bandstop"
        else:
            filtered = "bandpass"
        frequency = [lowcut, highcut]
    elif lowcut is not None:
        frequency = [lowcut]
        filtered = "highpass"
    elif highcut is not None:
        frequency = [highcut]
        filtered = "lowpass"
    if normalize is True:
        frequency = np.array(frequency) / (sampling_rate / 2)
    return frequency, filtered


# ----------------------Peak data extraction----------------------
def ppg_find_point(signal, sampling_rate=sampling_rate):
    signal = np.ravel(signal)
    signal[signal < 0] = 0
    signal_power = signal ** 2
    peak_size = int(np.rint(0.111 * sampling_rate))
    peak = signal_gently(signal_power, size=peak_size)
    beat_size = int(np.rint(0.667 * sampling_rate))
    beat = signal_gently(signal_power, size=beat_size)
    thr1 = beat + 0.02 * np.mean(signal_power)

    waves = peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    end_waves = end_waves[end_waves > beg_waves[0]]

    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(0.111 * sampling_rate))
    min_delay = int(np.rint(0.3 * sampling_rate))
    peaks = [0]

    for i in range(num_waves):
        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg
        if len_wave < min_len:
            continue
        data = signal[beg:end]
        locmax, props = si.find_peaks(data, prominence=(None, None))
        if locmax.size > 0:
            peak = beg + locmax[np.argmax(props["prominences"])]
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)
    peaks.pop(0)
    peaks = np.asarray(peaks).astype(int)
    return peaks


def signal_gently(signal, size=10):
    size = int(size)
    window = si.get_window("boxcar", size)
    w = window / window.sum()
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
    re_gently = np.convolve(w, x, mode="same")
    first_gently = re_gently[size:-size]

    size = int(size)
    window = si.get_window("parzen", size)
    w = window / window.sum()
    x = np.concatenate((first_gently[0] * np.ones(size), first_gently, first_gently[-1] * np.ones(size)))
    fi_gently = np.convolve(w, x, mode="same")
    final_gently = fi_gently[size:-size]
    return final_gently
