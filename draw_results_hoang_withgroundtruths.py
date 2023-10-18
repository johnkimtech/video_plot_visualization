import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import time
import os


def draw_results(
    input_folder_path,
    event_csv_path,
    output_video_path,
    configs,
    n_frames_limit=np.inf,
    index_col="frame",
):
    # Load CSV data
    df = pd.read_csv(event_csv_path, index_col=index_col)

    # Get list of image files in the input folder
    img_files = sorted(
        [
            f
            for f in os.listdir(input_folder_path)
            if os.path.isfile(os.path.join(input_folder_path, f))
            and f.startswith("img_")
        ]
    )

    # Process the images
    img_idx = 0
    N_frames = len(img_files)
    org_frame_rate = 30  # Assuming a default frame rate of 30 fps for images
    v_height, v_width, _ = cv2.imread(
        os.path.join(input_folder_path, img_files[0])
    ).shape
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
    gait_events_list = configs["gait_events_list"] or ["L_HS", "L_TO", "R_HS", "R_TO"]
    col_names = configs["column_names"] or gait_events_list
    while img_idx < N_frames:
        if frame_count > n_frames_limit:
            break

        img_path = os.path.join(input_folder_path, img_files[img_idx])
        frame = cv2.imread(img_path)

        frame_idx = int(
            os.path.splitext(img_files[img_idx].split("_")[1])[0]
        )  # Get current frame index
        if frame_idx in df.index:
            start_idx = max(first_frame_idx, frame_idx - configs["x_range"])
            da = df[start_idx:frame_idx]

            for i, gait_event in enumerate(gait_events_list):
                # draw graph
                axs[i].clear()
                axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
                # plot prediction
                pred_col, target_col = f"pred_{gait_event}", f"target_{gait_event}"
                axs[i].plot(
                    da.index,
                    da[pred_col],
                    color=configs["pred_colors"][i],
                    linewidth=4.0,
                    # linestyle="-",
                    label="prediction",
                )
                # plot target
                axs[i].plot(
                    da.index,
                    da[target_col],
                    color=configs["target_colors"][i],
                    linewidth=4.0,
                    # linestyle=":",
                    label="target",
                )

                axs[i].set_xlim(start_idx, max(configs["x_range"], frame_idx))
                axs[i].set_ylim(0, 1.0)
                axs[i].legend()
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
        frame_count += 1
        img_idx += 1

    # Release video writer
    out.release()


if __name__ == "__main__":
    tic = time.perf_counter()
    configs = {
        "gait_events_list": ["L_HS", "L_TO", "R_HS", "R_TO"],
        "pred_colors": [
            "white",
            "white",
            "white",
            "white",
        ],  # dont use black, it will be transparent
        "target_colors": [
            "red",
            "red",
            "red",
            "red",
        ],  # dont use black, it will be transparent
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
        "slow_down": 2,  # how slow the output compares to the original video, e.g. value of 2 will slow it down by 2x times, thus making it 2x times longer in duration
    }

    draw_results(
        input_folder_path="data/0022128380/0022128380_1",
        event_csv_path="data/temporal_parameters_estimation/output_22128380_1.csv",
        output_video_path="A.mp4",
        configs=configs,
        # n_frames_limit=200,  # comment out to process the whole video
    )
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.2f} seconds")
