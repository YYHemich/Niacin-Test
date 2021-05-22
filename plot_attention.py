from matplotlib import pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-s', '--src', required=True, help='Directory of the attention information')
args = vars(ap.parse_args())

att = np.load(args['src'])
new_ticks = np.linspace(0, 20, 21)
plt.figure(figsize=(12, 2.5), tight_layout=True)
plt.xlabel('time step')
plt.ylabel('attention weight')
plt.yticks(size=18)
plt.bar([i for i in range(21)], np.mean(att, axis=0))
plt.subplots_adjust(left=0, right=1., top=1, bottom=0.)
plt.xticks(new_ticks, size=18)
plt.show()
