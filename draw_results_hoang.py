import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
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
    """ """
    # Load CSV data
    df = pd.read_csv(event_csv_path, index_col=index_col)

    # Process the video
    cap = cv2.VideoCapture(input_video_path)

    N_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # plot configs
    plt.style.use(configs["theme"])
    plt.rcParams.update({"font.size": configs["font_size"]})
    plt.rcParams.update({"font.weight": configs["font_weight"]})
    plt.rcParams.update({"axes.linewidth": configs["axes_linewidth"]})
    fig = Figure(
        figsize=(v_width / 100, v_height / 100 / configs["aspect_ratio"]), dpi=50
    )
    canvas = FigureCanvas(fig)
    axs = [
        fig.add_axes([0.1, pos, configs["subplot_width"], configs["subplot_height"]])
        for pos in configs["subplot_positions"]
    ]

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
                # draw graph
                axs[i].clear()
                axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
                axs[i].plot(
                    da.index, da[gait_event], configs["event_colors"][i], linewidth=4.0
                )
                axs[i].set_xlim(start_idx, max(configs["x_range"], frame_idx))
                axs[i].set_ylim(0, 1.0)
                axs[i].set_title(col_names[i])

            # put graph into buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            buf = np.asarray(buf)
            buf = buf[:, :, :-1]  # Remove alpha channel
            buf = buf[:, :, ::-1]  # rgb to bgr (for opencv)

            buf_gray = buf.mean(axis=-1, keepdims=True).repeat(3, axis=-1)
            mask = 1 - (buf_gray / 255) ** configs["alpha"]

            frame[
                configs["margin_top"] : configs["margin_top"] + buf.shape[0],
                configs["margin_left"] : configs["margin_left"] + buf.shape[1],
                :,
            ] = (
                frame[
                    configs["margin_top"] : configs["margin_top"] + buf.shape[0],
                    configs["margin_left"] : configs["margin_left"] + buf.shape[1],
                    :,
                ]
                * mask
                + (1 - mask) * buf
            )

        # write frame to output file
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord("q"):
            break
        frame_count += 1

    # Release video capture and writer
    cap.release()
    out.release()


if __name__ == "__main__":
    tic = time.perf_counter()
    configs = {
        "event_colors": ["white", "red", "cyan", "yellow"],  # color for each subgraph
        # "event_colors": ['red', 'green', 'yellow', 'white'],  # color for each subgraph
        "column_names": [
            "[LEFT] Heel Strike",
            "[LEFT] Toe Off",
            "[RIGHT] Heel Strike",
            "[RIGHT] Toe Off",
        ],  # column names for better visualization, put None to use default column names from the input csv file
        "alpha": 0.7,
        "font_size": 25,
        "font_weight": "bold",
        "axes_linewidth": 3,
        "theme": "dark_background",
        "x_range": 150,  # range of the horizontal axis, default to 150 frames
        "subplot_width": 0.75,  # width of each subplot, relative to the total width of the graph
        "subplot_height": 0.15,  # similar to subplot_width
        "subplot_positions": [
            0.05,
            0.3,
            0.55,
            0.8,
        ],  # y-positions of each subgraph in the bigger graph
        "aspect_ratio": 0.5,  # controls the aspect ratio of the graph to be more rectangular or square
        "margin_left": 10,  # margin of the graph to the left corner of the input video
        "margin_top": 0,  # margin of the graph from the top corner of the input video
        "slow_down": 1,  # how slow the output compares to the original video, e.g. value of 2 will slow it down by 2x times, thus making it 2x times longer in duration
    }

    draw_results(
        input_video_path="14_VYI_MX.MP4",
        event_csv_path="14_VYI_MX.csv",
        output_video_path="A.mp4",
        configs=configs,
        # n_frames_limit=200,  # comment out to process the whole video
    )
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.2f} seconds")
