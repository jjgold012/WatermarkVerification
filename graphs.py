import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader

datafile = open('./WatermarkVerification1.csv')
file_reader = DictReader(datafile)

sat_vals = np.array([float(line['sat-epsilon']) for line in file_reader])
numbers = np.array(range(1, len(sat_vals)+1))
plt.step(numbers, sat_vals)
plt.xlabel('Number of Watermark Images')
plt.ylabel('epsilon')
plt.savefig('WatermarkVerification1')
# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
# plt.show()