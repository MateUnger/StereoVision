import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np

f = "./filtered_data.npz"
f_og = "./processed_data.npz"
all_data = []
paths = [f, f_og]
for p in paths:
    loaded_data = np.load(p, allow_pickle=True)
    data = loaded_data["pose_data"]
    all_data.append(data)
    print(f"shape of {p}: {data.shape}")

# loaded_data = np.load(f, allow_pickle=True)
# data = loaded_data["pose_data"]
# print(f"og shape: {data.shape}")
# for i, k in enumerate(loaded_data["kpt_labels"]):
#     print(i, k)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

for i, data in enumerate(all_data):
    print(i)
    if i == 0:
        label = "filtered"
        color = "b"
        alpha = 1
        s = 3
        marker = "v"
    else:
        label = "original"
        color = "r"
        alpha = 1
        s = 3
        marker = "o"

    kpt_idx = 16
    x = data[kpt_idx, 0, :]
    y = data[kpt_idx, 1, :]
    z = data[kpt_idx, 2, :]

    # ax.plot3D(0, 0, 0, "mo", markersize=3, label="origin")
    ax.scatter(x, y, z, s=s, c=color, alpha=alpha, label=label)
    # plt.show()
# set figure properties
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim(-1500, 1500)
ax.set_ylim(-1500, 1500)
ax.set_zlim(0, 2500)

# ax.view_init(elev=-145, azim=60)
ax.legend()

# ax.scatter(x, y, z)
plt.show()
