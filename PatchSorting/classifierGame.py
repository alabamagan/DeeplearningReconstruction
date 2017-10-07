from matplotlib import backend_bases as bb
from dataloader import BatchLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# Override save hotkey
plt.rcParams['keymap.save'] = ''


class ClassifierGame(object):
    global config

    def __init__(self, user):

        # Configs
        self.outputdir = "./" + user
        self.resultfile = "result.txt"
        self.samplepath = "../SIRT_Parallel_Slices/train"
        self.user = user
        self.currentDisplay = 0
        self.numofsamples = 50
        self.compare = ['064', '128', 'diff']

        # Initialize
        self.result = []
        self.Exec()

    def Exec(self):
        """
        Execute game
        :return:
        """

        # Set some variables
        self.classified = self.outputdir + "/" + self.resultfile
        self.currentDisplay = 0

        # Load samples
        self._b = BatchLoader(self.samplepath)
        self._samples = self._b(self.numofsamples)

        # Create plot
        self._fig = plt.figure(figsize=[16, 9])
        self._fig.set_tight_layout(True)
        self._ax = [self._fig.add_subplot(2, 3, 4 + i) for i in xrange(3)]
        [a.axis('off') for a in self._ax]
        self._textbox = self._fig.add_subplot(2, 1, 1)

        self._fig.canvas.mpl_connect('key_press_event', self.key_press_event)

        self.DrawNext()
        plt.show()
        pass

    def DrawNext(self):
        ax1, ax2, ax3 = self._ax

        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.imshow(self._samples[self.currentDisplay][self.compare[0]], cmap="Greys_r", vmin=-1000, vmax=500)
        ax2.imshow(self._samples[self.currentDisplay][self.compare[1]], cmap="Greys_r", vmin=-1000, vmax=500)
        ax3.imshow(self._samples[self.currentDisplay][self.compare[2]], cmap="Greys_r", vmin=-10, vmax=10)
        [a.axis('off') for a in self._ax]

        self._textbox.clear()
        self._textbox.axis('off')
        self._textbox.text(0, 0, "Explanation: \n" + "A set of 3 images are displayed each time, they are all \n" +
                                 "of the same type. The right-most image is the difference of the left two. \n "
                                 "Considering only the left-two images, there are a total of 3 classes, defined \n" 
                                 " as follow. \n"
                                 "1. (Press Q) Background, 90% are uniformly 0  \n" +
                                 "2. (Press W) Tissue/Objects occupies over 10% of pixels  \n" +
                                 "3. (Press E) Air, values at around -1000, may have streak artifacts. \n\n" +
                                 "Press B to go back to last image. \n\n " +
                                 "Press one of the key to select the most suitable class for each set of images \n" +
                                 "(%i / %i)"
                                  %(self.currentDisplay + 1, self.numofsamples))
        if (self.currentDisplay == self.numofsamples - 1):
            self._textbox.clear()
            self._textbox.text(0, 0, "All Done! Thank you for participating!")
            self._textbox.axis('off')
        self.currentDisplay += 1
        plt.draw()
        pass

    def DrawLast(self):
        self.currentDisplay -= 2
        self.DrawNext()
        pass

    def key_press_event(self, event):
        """

        :param matplotlib.backend_bases.LocationEvent event:
        :return:
        """
        assert isinstance(event, bb.KeyEvent)

        key_pressed = event.key

        if key_pressed == 'q':
            self.result.append('BG')
            pass
        elif key_pressed == 'w':
            self.result.append('TS')
            pass
        elif key_pressed == 'e':
            self.result.append('AR')
            pass
        elif key_pressed == 'b':
            self.result.pop()
            self.DrawLast()
            return
        else:
            return

        if (self.currentDisplay < self.numofsamples):
            self.DrawNext()
        else:
            plt.close()
            self.SaveResults()
        pass

    def SaveResults(self):
        if (not os.path.isdir(self.outputdir)):
            os.system("mkdir -p " + self.outputdir)

        if (not os.path.isfile(self.resultfile)):
            self._resultfile = file(self.outputdir + "/" + self.resultfile, 'w')

        for s in self.result:
            self._resultfile.write(s)
            self._resultfile.write("\n")

        for i in xrange(self.numofsamples):
            images = [self._samples[i][c] for c in self.compare]
            outim = np.concatenate([im.reshape(1, im.shape[0], im.shape[1]) for im in images], 0)
            name = self.outputdir + "/SE%s_%05d"%(str.join(self.compare), i)
            np.save(name, outim)
            
def main():
    name = raw_input("Please enter your name: ")
    if (os.path.isfile(name)):
        print "Warning! Detected another user with the same name! Cannot proceed!"
        return

    C = ClassifierGame(name)


if __name__ == '__main__':
    main()