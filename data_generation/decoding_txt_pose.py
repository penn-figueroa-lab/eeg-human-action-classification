import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

log = open("log/log_1.txt", "r")
lines = log.readlines()
shoulder = []
elbow = []
wrist = []
for line in lines:
    shoulder.append(json.loads(line)['data']['shoulder_r'][:3])
    elbow.append(json.loads(line)['data']['elbow_r'][:3])
    wrist.append(json.loads(line)['data']['wrist_r'][:3])
shoulder = np.array(shoulder)
elbow = np.array(elbow)
wrist = np.array(wrist)

center = shoulder[:1, :]
shoulder = shoulder - center
elbow = elbow - center
wrist = wrist - center

N = shoulder.shape[0]


def update(num, shoulder, elbow, wrist, line1, line2):
    line1.set_data([[shoulder[num][0], elbow[num][0]],
                    [shoulder[num][1], elbow[num][1]]])
    line1.set_3d_properties([shoulder[num][2], elbow[num][2]])
    line2.set_data([[wrist[num][0], elbow[num][0]],
                    [wrist[num][1], elbow[num][1]]])
    line2.set_3d_properties([wrist[num][2], elbow[num][2]])


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
line1, = ax.plot([shoulder[0][0], elbow[0][0]],
                 [shoulder[0][1], elbow[0][1]],
                 [shoulder[0][2], elbow[0][2]])
line2, = ax.plot([wrist[0][0], elbow[0][0]],
                 [wrist[0][1], elbow[0][1]],
                 [wrist[0][2], elbow[0][2]])

upper = np.max([shoulder, elbow, wrist])
lower = np.min([shoulder, elbow, wrist])

ax.set_xlim3d([lower, upper])
ax.set_xlabel('X')

ax.set_ylim3d([lower, upper])
ax.set_ylabel('Y')

ax.set_zlim3d([lower, upper])
ax.set_zlabel('Z')
ani = animation.FuncAnimation(fig, update, N, fargs=(shoulder, elbow, wrist, line1, line2), interval=1, blit=False)
plt.show()
