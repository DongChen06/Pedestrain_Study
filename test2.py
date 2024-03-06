import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
import matplotlib.ticker as ticker


def format_time(x, pos=None):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes}:{seconds:02d}'


# Load your video
video_file = "scenevideo.mp4"

# Extract the audio from the video
video = VideoFileClip(video_file)
audio = video.audio.to_soundarray(fps=22050)

# Convert the stereo audio to mono
audio_mono = librosa.to_mono(audio.T)

# Calculate the decibel level for each frame
frame_length = 1024
hop_length = frame_length // 4
rms = librosa.feature.rms(y=audio_mono, frame_length=frame_length, hop_length=hop_length)
db = librosa.amplitude_to_db(rms, ref=np.max)

# Create a time array to match the db array
frames = range(len(db[0]))
t = librosa.frames_to_time(frames, sr=22050, hop_length=hop_length)

# # Plot the decibel level over time
# plt.figure(figsize=(10, 4))
# plt.plot(t, db[0])
# plt.xlabel('Time (s)')
# plt.ylabel('Decibel (dB)')
# plt.title('Decibel Level Over Time')
# plt.tight_layout()
# plt.show()

db_shifted = db - np.min(db)
threshold_dB = 60  # Example threshold
window_length = 10  # The length of the smoothing window
db_smoothed = np.convolve(db_shifted[0], np.ones(window_length)/window_length, mode='same')

plt.figure(figsize=(10, 4))
plt.plot(t, db_smoothed)
plt.xlabel('Time')
plt.ylabel('Decibel (dB)')
plt.title('Positive Decibel Level Over Time')

num_seconds = t[-1]
num_ticks = int(np.ceil(num_seconds / 30))  # Number of ticks needed
tick_positions = np.linspace(0, num_seconds, num_ticks)  # Positions at which to place the ticks
tick_labels = [f'{int(p//60)}:{int(p%60):02d}' for p in tick_positions]  # Convert to mm:ss format

plt.xticks(tick_positions, tick_labels, rotation=50)

plt.axhline(y=threshold_dB, color='grey', linestyle='--', label=f'Threshold ({threshold_dB} dB)')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
plt.tight_layout()
plt.savefig('decibel_plot.png')
plt.show()
