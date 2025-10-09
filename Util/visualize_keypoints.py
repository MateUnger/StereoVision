import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np


class FrameViewer:
    def __init__(self, input_data):
        self.input_data = input_data
        self.frame_idx = 0
        self.n_frames = input_data.shape[0]

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.plot_frame(self.frame_idx)
        plt.show()

    def plot_frame(self, idx):
        self.ax.clear()
        xs = self.input_data[:, 0, idx]
        ys = self.input_data[:, 1, idx]
        zs = self.input_data[:, 2, idx]
        self.ax.scatter(xs, ys, zs, c="b", s=40, label="Keypoints")

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title(f"Frame {idx}")
        self.ax.set_xlim(-2500, 2500)
        self.ax.set_ylim(-2500, 2500)
        self.ax.set_zlim(0, 2500)
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
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select a .npy file",
        filetypes=[("NPy files", "*.npy")],
        initialdir=os.getcwd(),
    )

    # file_path = ".\preprocessed_data\A2A6841F-E163-433C-BEF4-55BD9C15437A_mate_walking.npz"
    if file_path:
        kpt_data = np.load(file_path, allow_pickle=True)
        print(f"file:  {os.path.basename(file_path)}")

        FrameViewer(kpt_data)
