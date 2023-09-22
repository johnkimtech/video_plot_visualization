import cv2
import numpy as np
import pandas as pd
import time 

def draw_results(
    input_video_path,
    event_csv_path,
    output_video_path,
    configs,
    n_frames_limit=np.inf,
    index_col="frame",
):
    # Load CSV data
    df = pd.read_csv(event_csv_path, index_col=index_col)

    # Process the video
    cap = cv2.VideoCapture(input_video_path)

    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # video config
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        int(org_frame_rate // configs["slow_down"]),
        (v_width, v_height),
    )

    first_frame_idx = min(df.index)
    frame_count = 0
    col_names = configs["column_names"] or list(df.columns)
    while True:
        if frame_count > n_frames_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(1)) - 1  # Get current frame index
        if frame_idx in df.index:
            start_idx = max(first_frame_idx, frame_idx - configs["x_range"])
            da = df[start_idx:frame_idx]
            gait_events_list = da.columns

            for i, gait_event in enumerate(gait_events_list):
                # Create a background rectangle
                x1 = int(configs["margin_left"])
                x2 = int(x1 + configs["subplot_width"] * v_width)
                y1 = int(configs["margin_top"] + i * (configs["subplot_height"] * v_height + configs["subplot_spacing"] * v_height))
                y2 = int(y1 + configs["subplot_height"] * v_height)

                cv2.putText(frame, "1.0", (x1-30,y1+10),  cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=configs["event_colors"][i], thickness=configs["axes_linewidth"]//2)
                cv2.putText(frame, "0.5", (x1-30,(y1+y2)//2),  cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=configs["event_colors"][i], thickness=configs["axes_linewidth"]//2)
                cv2.putText(frame, "0.0", (x1-30,y2+10),  cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=configs["event_colors"][i], thickness=configs["axes_linewidth"]//2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), configs["event_colors"][i], configs["axes_linewidth"])  # Fill with color
                cv2.line(frame, (x1, (y1+y2)//2), (x2, (y1+y2)//2), color=configs["event_colors"][i], thickness=configs["axes_linewidth"]//2)
                margin_x = 140
                margin_y = 25
                cv2.putText(frame, col_names[i], (((x1+x2)-margin_x)//2,y2+margin_y),  cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=configs["event_colors"][i], thickness=configs["axes_linewidth"])

                # Overlay the event probabilities as a graph on the rectangle
                # max_prob = da[gait_event].max()
                prev_x = int(x1)
                prev_y = int(y2)
                for x, prob in enumerate(da[gait_event]):
                    x_coord = int(x1 + x / len(da) * (x2 - x1))
                    y_coord = int(y2 - prob * (y2 - y1))
                    # cv2.line(frame, (x_coord, int(y2)), (x_coord, y_coord), configs["event_colors"][i], 2) 
                    # cv2.line(frame, (x_coord, y_coord), (x_coord, y_coord), configs["event_colors"][i], configs["plot_thickness"]) 
                    cv2.line(frame, (prev_x, prev_y), (x_coord, y_coord), configs["event_colors"][i], configs["plot_thickness"]) 
                    prev_x, prev_y = x_coord, y_coord

        # write frame to output file
        out.write(frame)
        frame_count += 1

    # Release video capture and writer
    cap.release()
    out.release()
    print(frame_count)

if __name__ == "__main__":
    tic = time.perf_counter()

    configs = {
    "axes_linewidth": 3,
    "slow_down": 5,  # Customize output video frame rate (e.g., slow down by a factor of 2)
    "subplot_width": 0.3,  # Width of each subplot
    "subplot_height": 0.15,  # Height of each subplot
    "margin_left": 50,  # Left margin for subplots
    "margin_top": 10,  # Top margin for subplots
    "plot_thickness": 2,
    "event_colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 210, 255)],  # BGR colors for each event
    "x_range": 100,  # Number of frames to include in the X-axis
    "subplot_positions": [10, 30, 50],  # Y-positions of subplots (as fractions of frame height)
    "subplot_spacing": .1,
    "column_names": None,  # List of column names for event labels (if None, will use column names from CSV)
}

    draw_results(
        input_video_path="14_VYI_MX.MP4",
        event_csv_path="14_VYI_MX.csv",
        output_video_path="B.mp4",
        configs=configs,
        # n_frames_limit=200,  # comment out to process the whole video
    )
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.2f} seconds")
