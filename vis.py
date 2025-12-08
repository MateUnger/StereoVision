import os
import numpy as np
import pandas as pd
import json
import glob
from Util.util import *

# from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt

import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display


# TODO: color foot keyppoints based on gc period (c1 when on ground, c2 when in air)
# stride cycle starts with first IC (heel)


# load static parameters from file
with open("./Util/gait_analysis_properties.json", "r") as json_file:
    gait_analysis_properties = json.load(json_file)
    stereo_properties = gait_analysis_properties["stereo"]
    qualysis_properties = gait_analysis_properties["qualisys"]


# load file, display basic info
file_path = askopenfilename()
if file_path:
    loaded_data = np.load(file_path, allow_pickle=True)
    print(f"file:  {get_basename(file_path)}")
    print(f"keys:  {list(loaded_data.keys())}")

try:
    # stereo
    data = loaded_data["keypoints_3d"]
    properties = stereo_properties
except:
    # qualisys
    # data = loaded_data["pose_data"]
    data = loaded_data["keypoints"]
    properties = qualysis_properties
    data = filter_data(
        data,
        properties["fps"],
        "lowpass",
        [properties["filter_cutoff"]],
        properties["filter_order"],
        properties["max_gap"],
    )


print(f"data shape: {data.shape}")
kpt_labels = loaded_data["kpt_labels"].tolist()

gait_events = get_gait_events(data, kpt_labels, properties)

IC_frames = [event["frame"] for event in gait_events["IC"]]
FC_frames = [event["frame"] for event in gait_events["FC"]]
print(f"ICs: {len(IC_frames)}")
print(f"FCs: {len(FC_frames)}")


class FrameViewer:
    def __init__(self, vGait, IC_frames, FC_frames, kpt_labels):
        self.vGait = vGait
        self.IC_frames = set(IC_frames)
        self.FC_frames = set(FC_frames)
        self.n_frames = vGait.shape[2]
        self.kpt_labels = kpt_labels
        self.frame_idx = 0

        self.fig = plt.figure(figsize=(11, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.plot_frame(self.frame_idx)
        plt.show()

    def plot_frame(self, idx):
        self.ax.clear()

        xs = self.vGait[:, 0, idx]
        ys = self.vGait[:, 1, idx]
        zs = self.vGait[:, 2, idx]
        self.ax.scatter(xs, ys, zs, c="b", s=40, label="Keypoints")

        self.ax.scatter(
            3.09417443, 0.19429838, -(-1.11145841), marker="D", c="r", label="stereo camera"
        )
        self.ax.scatter(0, 0, 0, c="r", label="origin")
        # self.ax.scatter(0.75, 0, 0)
        # self.ax.scatter(0, 0.45, 0)
        # self.ax.scatter(0.75, 0.45, 0)
        self.ax.plot([0, 0], [0, 0.45], "blue", zs=[0, 0])
        self.ax.plot([0, 0.75], [0, 0], "blue", zs=[0, 0])
        self.ax.plot([0.75, 0.75], [0, 0.45], "blue", zs=[0, 0])
        self.ax.plot([0.75, 0], [0.45, 0.45], "blue", zs=[0, 0])
        # Build a lookup for IC and FC events by frame
        ic_events = {event["frame"]: event for event in gait_events["IC"]}
        fc_events = {event["frame"]: event for event in gait_events["FC"]}

        # IC event
        if idx in ic_events:
            side = ic_events[idx]["side"]
            perspective = ic_events[idx].get("perspective", True)
            if side == "right":
                marker_idx = self.kpt_labels.index("right_heel")
            else:
                marker_idx = self.kpt_labels.index("left_heel")
            color = "black" if perspective is np.False_ else "r"
            print(perspective)
            label = f"IC ({side} heel{' - bad perspective' if perspective is False else ''})"
            self.ax.scatter(
                xs[marker_idx],
                ys[marker_idx],
                zs[marker_idx],
                c=color,
                s=50,
                marker="o",
                label=label,
            )

        # FC event
        if idx in fc_events:
            side = fc_events[idx]["side"]
            perspective = fc_events[idx].get("perspective", True)
            if side == "right":
                marker_idx = self.kpt_labels.index("right_big_toe")
            else:
                marker_idx = self.kpt_labels.index("left_big_toe")
            color = "black" if perspective is np.False_ else "g"
            label = f"FC ({side} toe{' - bad perspective' if perspective is False else ''})"
            self.ax.scatter(
                xs[marker_idx],
                ys[marker_idx],
                zs[marker_idx],
                c=color,
                s=100,
                marker="^",
                label=label,
            )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title(f"Frame {idx}")
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-0.75, 1.25)
        self.ax.set_zlim(0, 2)
        self.ax.set_box_aspect([2, 1, 1])

        self.ax.legend()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "d":
            self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
            self.plot_frame(self.frame_idx)
        elif event.key == "a":
            self.frame_idx = max(self.frame_idx - 1, 0)
            self.plot_frame(self.frame_idx)
        elif event.key == "ctrl+d":
            self.frame_idx = min(self.frame_idx + 100, self.n_frames - 1)
            self.plot_frame(self.frame_idx)
        elif event.key == "ctrl+a":
            self.frame_idx = max(self.frame_idx - 100, 0)
            self.plot_frame(self.frame_idx)


if __name__ == "__main__":
    FrameViewer(data, IC_frames, FC_frames, kpt_labels)
