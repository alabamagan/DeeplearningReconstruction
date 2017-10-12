from dataloader import BatchLoader
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

b = BatchLoader("../SIRT_Parallel_Slices/train/")
b.SetTrainMode(True)
samples = b(15)

plt.ion()
for i in xrange(15):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax1.imshow(samples['ori'][i])
    ax2.imshow(samples['ori'][i] - samples['064'][i])
    ax3.imshow(samples['064'][i] - samples['032'][i])
    plt.draw()
    plt.pause(1.5)
