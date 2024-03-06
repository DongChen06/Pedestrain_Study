import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

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

# Find the minimum dB value and shift all values by this amount to make them positive
min_db = np.min(db)
db_shifted = db - min_db

# Create a time array to match the db array
frames = range(len(db_shifted[0]))
t = librosa.frames_to_time(frames, sr=22050, hop_length=hop_length)

# Create a 2D array for the heatmap
heatmap_data = np.tile(db_shifted, (int(8192 / frame_length), 1))

# Plot the heatmap
plt.figure(figsize=(10, 4))
librosa.display.specshow(heatmap_data, sr=22050, hop_length=hop_length, x_axis='time', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (s)')
plt.ylabel('Decibel (dB)')

num_seconds = t[-1]
num_ticks = int(np.ceil(num_seconds / 30))  # Number of ticks needed
tick_positions = np.linspace(0, num_seconds, num_ticks)  # Positions at which to place the ticks
tick_labels = [f'{int(p//60)}:{int(p%60):02d}' for p in tick_positions]  # Convert to mm:ss format

plt.xticks(tick_positions, tick_labels, rotation=50)

plt.title('Decibel Level Heatmap')
plt.tight_layout()
plt.savefig('decibel_heatmap.png')
plt.show()
