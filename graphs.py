import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader, DictWriter

# from tensorflow import keras
# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# wm_images = np.load('./data/wm.set.npy')
# wm_labels = np.loadtxt('./data/wm.labels.txt', dtype='int32')
# plt.imshow(wm_images[9], cmap='gray')
# plt.show()

model_name = 'mnist.w.wm'
# vals_epsilon = {}
# vals_acc = {}
# x = [0,1,2,3,4,5,6,7,25,50,75,100]
# x = [1,2,3,4,5,6,7,25,50,75,100]
# x = [0,1,2,3,4,5,6,7]
# x = [1,2,3,4,5,6,7]
# x_str = ','.join(map(str, x))

# out_file = open('./data/results/problem3/{}.wm.summary.csv'.format(model_name), 'w')
# out_file.write('num-of-wm,average-sat-epsilon,average-test-accuracy\n')

# for i in x:
#     datafile = open('./data/results/problem3/{}.{}.wm.accuracy.csv'.format(model_name, i))
#     file_reader = DictReader(datafile)
#     vals_acc[i] = np.array([float(line['test-accuracy']) for line in file_reader])
#     datafile.close()
#     if i == 0:
#         vals_epsilon[i] = 0
#     else:
#         datafile = open('./data/results/problem3/{}.{}.wm.csv'.format(model_name, i))
#         file_reader = DictReader(datafile)
#         vals_epsilon[i] = np.array([float(line['sat-epsilon']) for line in file_reader])
#         datafile.close()
#     out_file.write('{},{},{}\n'.format(i, np.average(vals_epsilon[i]), np.average(vals_acc[i])))
# out_file.close()


# avrg_acc = np.array([np.average(vals_acc[i]) for i in x])
# max_acc = np.array([np.max(vals_acc[i]) for i in x])
# min_acc = np.array([np.min(vals_acc[i]) for i in x])
# avrg_eps = np.array([np.average(vals_epsilon[i]) for i in x])
# max_eps = np.array([np.max(vals_epsilon[i]) for i in x])
# min_eps = np.array([np.min(vals_epsilon[i]) for i in x])
# plt.bar(x, avrg_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/problem3/{}_{}_average_accuracy.png'.format(model_name.replace('.','_'), x_str))

# plt.clf()
# plt.bar(x, max_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/problem3/{}_{}_maximum_accuracy.png'.format(model_name.replace('.','_'), x_str))

# plt.clf()
# plt.bar(x, min_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/problem3/{}_{}_minimum_accuracy.png'.format(model_name.replace('.','_'), x_str))



# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))

datafile = open('./data/results/problem2/{}.csv'.format(model_name))
file_reader = DictReader(datafile)

sat_vals = np.array([float(line['sat-epsilon']) for line in file_reader])
sat_vals = np.sort(sat_vals)
numbers = np.array(range(1, len(sat_vals)+1))
plt.step(numbers, sat_vals)
plt.xlabel('Number of Watermark Images')
plt.ylabel('epsilon')
plt.savefig('./data/results/problem2/{}.png'.format(model_name.replace('.','_')))
# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
# plt.show()


