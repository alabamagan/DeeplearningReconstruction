from PatchSorting import BatchLoader
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

b = BatchLoader("../SIRT_Parallel_Slices/train/")
b._SetKernelSize([32, 32])
samples = b(101)
print len(samples)
plt.ion()
for i in xrange(len(samples)):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax1.imshow(samples[i]['064'], vmin=-1000, vmax=100, cmap="Greys_r")
    ax2.imshow(samples[i]['128'], vmin=-1000, vmax=100, cmap="Greys_r")
    ax3.imshow(samples[i]['diff'], vmin=-15, vmax=15, cmap="jet")
    plt.draw()
    plt.pause(0.01)
