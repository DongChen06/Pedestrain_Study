import librosa
import librosa.display
import numpy as np
from moviepy.editor import VideoFileClip
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Load video1 and extract the audio
video1_file = "Data/scenevideo.mp4"
video1 = VideoFileClip(video1_file)
audio1 = video1.audio.to_soundarray(fps=22050)
# Convert the stereo audio to mono
audio1_mono = librosa.to_mono(audio1.T)

# Calculate the decibel level for each frame
frame_length = 1024
hop_length = frame_length // 4
rms = librosa.feature.rms(y=audio1_mono, frame_length=frame_length, hop_length=hop_length)
db = librosa.amplitude_to_db(rms, ref=np.max)

# Load video2 to get the frame rate and number of frames
video2_file = "Data/scenevideo.mp4"
video2 = VideoFileClip(video2_file)

# Prepare to write the output video
output_video_file = "scenevideo_with_waveform.mp4"  # Replace with the desired output video file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_file, fourcc, video2.fps, video2.size)

# Define the sampling rate for the audio
audio_sampling_rate = 22050  # The sampling rate used in the audio analysis

# Function to create a waveform image for the dB levels
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def create_waveform(scaled_db, i, frame_size, scale=0.2, waveform_color='red'):
    start_index = max(0, i - 100)
    end_index = i
    db_segment = scaled_db[start_index:end_index]

    # Create time axis for the segment
    X = np.arange(start_index, end_index)

    # Create the plot
    fig = Figure(figsize=(frame_size[1] / 100 * scale, frame_size[0] / 100 * scale), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor((0.9, 0.9, 0.9))  # Light grey background
    fig.patch.set_facecolor((0.9, 0.9, 0.9))  # Match figure background to axis background

    # Set y-axis limits based on the range of dB values you expect
    ax.set_ylim(40, 80)

    ax.yaxis.set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(False)  # Assuming you don't need the x-axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().set_ticks([])  # Hide x-axis ticks

    # Plot the dB levels
    ax.plot(X, db_segment, color=waveform_color)

    # Convert the Matplotlib figure to an OpenCV image
    canvas.draw()
    waveform_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    waveform_img = waveform_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return waveform_img


# Overlay waveform on the video frame
def overlay_waveform(frame, waveform_img, alpha=0.5):
    # Specify the position where the waveform will be placed
    x_offset = 1500
    y_offset = 600
    x_end = x_offset + waveform_img.shape[1]
    y_end = y_offset + waveform_img.shape[0]

    # Ensure the waveform image is in BGR format for OpenCV
    waveform_img_bgr = cv2.cvtColor(waveform_img, cv2.COLOR_RGB2BGR)

    # Resize waveform image to fit the designated area if necessary
    waveform_img_resized = cv2.resize(waveform_img_bgr, (x_end - x_offset, y_end - y_offset))

    # Blend the waveform image onto the video frame with the specified alpha
    frame[y_offset:y_end, x_offset:x_end] = cv2.addWeighted(frame[y_offset:y_end, x_offset:x_end], 1 - alpha,
                                                            waveform_img_resized, alpha, 0)

    return frame


# Scale the decibel levels to match the duration of video2
scaled_db = np.interp(np.arange(0, video2.duration, 1 / video2.fps),
                      librosa.frames_to_time(np.arange(len(db[0])), sr=22050, hop_length=hop_length),
                      db[0])

db_min = np.min(scaled_db)
if db_min < 0:
    scaled_db += abs(db_min)

# Process each frame of video2
for i, frame in enumerate(video2.iter_frames(fps=video2.fps, dtype="uint8")):
    # Convert frame to BGR color for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    waveform_img = create_waveform(scaled_db, i, frame_bgr.shape[:2])
    frame_with_waveform = overlay_waveform(frame_bgr, waveform_img)

    # Write the modified frame to the output video file
    out_video.write(frame_with_waveform)

# Release the video writer object
out_video.release()
