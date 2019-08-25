import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader, DictWriter

model_name = 'mnist.w.wm'

# epsilons_vals = np.array([])
# out_file = open('./data/results/problem2/{}.WatermarkVerification2.csv'.format(model_name), 'w')
# file_writer = DictWriter(out_file, ['unsat-epsilon','sat-epsilon','original-prediction','sat-prediction'])
# file_writer.writeheader()
# for i in range(50):
#     epsilons = np.load('./data/results/problem2/{}.WatermarkVerification2_{}-{}.vals.npy'.format(model_name, 2*i, 2*i + 1))
#     epsilons_vals = epsilons if epsilons_vals.size==0 else np.append(epsilons_vals, epsilons, axis=0)
#     datafile = open('./data/results/problem2/{}.WatermarkVerification2_{}-{}.csv'.format(model_name, 2*i, 2*i + 1))
#     file_reader = DictReader(datafile)
#     for line in file_reader:
#         file_writer.writerow(dict(line))
# out_file.close()
# np.save('./data/results/problem2/{}.WatermarkVerification2.vals'.format(model_name), epsilons_vals)


datafile = open('./data/results/problem1/{}.WatermarkVerification1.csv'.format(model_name))
file_reader = DictReader(datafile)

sat_vals = np.array([float(line['sat-epsilon']) for line in file_reader])
sat_vals = np.sort(sat_vals)
numbers = np.array(range(1, len(sat_vals)+1))
plt.step(numbers, sat_vals)
plt.xlabel('Number of Watermark Images')
plt.ylabel('epsilon')
plt.savefig('./data/results/problem1/{}.WatermarkVerification1.png'.format(model_name))
# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
# plt.show()


