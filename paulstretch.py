# Paul's Extreme Sound Stretch (Paulstretch) - released under Public Domain
#
# by Nasca Octavian PAUL, Targu Mures, Romania
# http://www.paulnasca.com/
# http://hypermammut.sourceforge.net/paulstretch/
#
# Modifications by @canyondust
# https://github.com/canyondust/paulstretch_python/commit/2d59c9502f39e0779e6a817e0fd8eb1ab8bbbb97
#
# Refactored by @nafeu to remove CLI (January 2025)
#
# Example usage:
# `pip install cffi numpy pysoundfile`
#
# from stretch_sample import paulstretch
# paulstretch("input.wav", "output.wav", stretch=8.0, window_size=0.25)

import sys
import numpy as np
import soundfile as sf
from datetime import datetime


def optimize_windowsize(n):
    """Optimizes the window size for FFT processing."""
    orig_n = n
    while True:
        n = orig_n
        while (n % 2) == 0:
            n /= 2
        while (n % 3) == 0:
            n /= 3
        while (n % 5) == 0:
            n /= 5

        if n < 2:
            break
        orig_n += 1
    return orig_n


def load_wav(filename, start_frame=0, end_frame=None):
    """Loads a WAV file into a NumPy array."""
    try:
        wavedata, samplerate = sf.read(filename, start=start_frame, stop=end_frame)

        ar = [[], []]

        # hack mono to stereo conversion
        if (str(type(wavedata[0])) == "<class 'numpy.float64'>"):
            for a in wavedata:
                ar[0].append(a)
                ar[1].append(a)
        else:
            for a in wavedata:
                ar[0].append(a[0])
                ar[1].append(a[1])

        wavedata = np.array(ar)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None, None
    return (samplerate, wavedata)


def paulstretch(
    input_path,
    output_path,
    stretch=8.0,  # Default stretch amount
    window_size=0.25,  # Default window size (seconds)
    start_frame=0,
    end_frame=None,
    show_logs=True
):
    """
    Applies Paul's Extreme Sound Stretch (Paulstretch) algorithm to an input WAV file.

    Parameters:
    - input_path: Path to the input WAV file
    - output_path: Path to the output stretched WAV file
    - stretch: Stretch factor (1.0 = no stretch)
    - window_size: Window size in seconds
    - start_frame: Start frame for input (default: 0)
    - end_frame: End frame for input (default: None, reads the entire file)
    """

    # Load WAV file
    samplerate, smp = load_wav(input_path, start_frame, end_frame)
    if samplerate is None:
        print("Failed to load input file.")
        return

    nchannels = smp.shape[0] if smp.ndim > 1 else 1

    if show_logs:
        print(f"Processing {input_path} with stretch={stretch}, window_size={window_size}s")

    # Open output file
    outfile = sf.SoundFile(output_path, 'w', samplerate, nchannels)

    # Ensure window size is optimized and even
    windowsize = int(window_size * samplerate)
    windowsize = max(16, optimize_windowsize(windowsize))
    windowsize = int(windowsize / 2) * 2
    half_windowsize = int(windowsize / 2)

    # Correct the end of the smp
    nsamples = smp.shape[1] if smp.ndim > 1 else len(smp)
    end_size = max(int(samplerate * 0.05), 16)

    if smp.ndim > 1:
        smp[:, nsamples - end_size: nsamples] *= np.linspace(1, 0, end_size)
    else:
        smp[nsamples - end_size: nsamples] *= np.linspace(1, 0, end_size)

    # Compute the displacement inside the input file
    start_pos = 0.0
    displace_pos = (windowsize * 0.5) / stretch

    # Create window function
    window = np.power(1.0 - np.power(np.linspace(-1.0, 1.0, windowsize), 2.0), 1.25)

    old_windowed_buf = np.zeros((nchannels, windowsize))

    start_time = datetime.now()

    while True:
        # Get the windowed buffer
        istart_pos = int(np.floor(start_pos))
        buf = smp[:, istart_pos:istart_pos + windowsize] if smp.ndim > 1 else smp[istart_pos:istart_pos + windowsize]

        if buf.shape[-1] < windowsize:
            pad_shape = (nchannels, windowsize - buf.shape[1]) if smp.ndim > 1 else (windowsize - buf.shape[0],)
            buf = np.append(buf, np.zeros(pad_shape), axis=-1)

        buf = buf * window

        # Get the amplitudes of the frequency components and discard the phases
        freqs = np.abs(np.fft.rfft(buf))

        # Randomize the phases
        ph = np.random.uniform(0, 2 * np.pi, freqs.shape) * 1j
        freqs = freqs * np.exp(ph)

        # Perform inverse FFT
        buf = np.fft.irfft(freqs)

        # Window again the output buffer
        buf *= window

        # Overlap-add the output
        output = buf[:, 0:half_windowsize] + old_windowed_buf[:, half_windowsize:windowsize] if smp.ndim > 1 else \
            buf[0:half_windowsize] + old_windowed_buf[half_windowsize:windowsize]
        old_windowed_buf = buf

        # Clamp values to -1.0 to 1.0
        np.clip(output, -1.0, 1.0, out=output)

        # Write output to file
        if smp.ndim > 1:
            d = np.int16(output.T.ravel() * 32767.0)
            d = np.array_split(d, len(d) // 2)
        else:
            d = np.int16(output * 32767.0)

        outfile.write(d)

        start_pos += displace_pos
        if start_pos >= nsamples:
            if show_logs:
                print("100% Complete")
            break

        if show_logs:
            sys.stdout.write(f"{int(100.0 * start_pos / nsamples)}% \r")
            sys.stdout.flush()

    outfile.close()
    if show_logs:
        print(f"Stretched in: {datetime.now() - start_time}")
        print(f"Exported to: {output_path}")


def main():
    args = sys.argv[1:]

    if not args or len(args) < 2:
        print("Error: input_path and output_path are required.")
        return

    input_path = args[0]
    output_path = args[1]
    stretch = float(args[2]) if len(args) > 2 else 8.0
    window_size = float(args[3]) if len(args) > 3 else 0.25

    paulstretch(input_path, output_path, stretch=stretch, window_size=window_size)

if __name__ == "__main__":
    main()
