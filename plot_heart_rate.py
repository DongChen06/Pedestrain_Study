import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d


def interpolate_heart_rate(heart_rate_values, video_fps, total_frames):
    # Original timestamps based on 1 sample per second
    original_timestamps = np.arange(len(heart_rate_values))

    # New timestamps based on the video's FPS and total frame count
    new_timestamps = np.linspace(0, len(heart_rate_values) - 1, num=total_frames)

    # Interpolation
    interpolator = interp1d(original_timestamps, heart_rate_values, kind='linear')
    new_heart_rate_values = interpolator(new_timestamps)

    return new_heart_rate_values

def calculate_delay(start_time_video_str, start_time_heart_rate_str, fmt='%I:%M:%S%p'):
    start_time_video = datetime.strptime(start_time_video_str, fmt)
    start_time_heart_rate = datetime.strptime(start_time_heart_rate_str, fmt)
    delay = (start_time_heart_rate - start_time_video).seconds
    return delay


def create_waveform(heart_rate_values, i, frame_size, scale=0.2, waveform_color='red', delay=0):
    start_index = max(0, i - 100 + delay)
    end_index = i + delay
    hr_segment = heart_rate_values[start_index:end_index]

    X = np.arange(start_index, start_index + len(hr_segment))  # Recalculate X based on hr_segment's length

    fig = Figure(figsize=(frame_size[1] / 100 * scale, frame_size[0] / 100 * scale), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor((0.9, 0.9, 0.9))
    fig.patch.set_facecolor((0.9, 0.9, 0.9))
    ax.set_ylim(40, 180)
    ax.yaxis.set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.plot(X, hr_segment, color=waveform_color)

    canvas.draw()
    waveform_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    waveform_img = waveform_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return waveform_img


def overlay_waveform(frame, waveform_img, alpha=0.5, position=(1500, 300)):
    x_offset, y_offset = position
    x_end = x_offset + waveform_img.shape[1]
    y_end = y_offset + waveform_img.shape[0]
    waveform_img_bgr = cv2.cvtColor(waveform_img, cv2.COLOR_RGB2BGR)
    waveform_img_resized = cv2.resize(waveform_img_bgr, (x_end - x_offset, y_end - y_offset))
    frame[y_offset:y_end, x_offset:x_end] = cv2.addWeighted(frame[y_offset:y_end, x_offset:x_end], 1 - alpha,
                                                            waveform_img_resized, alpha, 0)
    return frame


# Calculate delay
delay = calculate_delay('09:38:55AM', '09:42:17AM')

# Load the heart rate data from a CSV file
heart_rate_df = pd.read_csv('Data/HeartRate.cvs')
heart_rate_values = heart_rate_df['Heart Rate'].values

# Load your video
video_path = 'scenevideo_with_waveform.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

# Interpolate heart rate data to match video FPS
interpolated_heart_rate_values = interpolate_heart_rate(heart_rate_values, video_fps, total_frames)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('heart_rate_noise.mp4', fourcc, video_fps, (frame_width, frame_height))

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    waveform_img = create_waveform(heart_rate_values, i, frame.shape[:2], delay=delay)
    frame_with_waveform = overlay_waveform(frame, waveform_img)
    out_video.write(frame_with_waveform)

    i += 1

cap.release()
out_video.release()
