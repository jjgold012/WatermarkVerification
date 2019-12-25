import numpy as np
from csv import DictReader, DictWriter

folder = './data/results/problem4/'
model_name = 'mnist.w.wm'
batchSize = 10
num = 100


out_file = open('{}{}.2.wm.csv'.format(folder, model_name), 'w')
vals = None
file_writer = None
for i in range(num):
    start = i*batchSize
    finish = start+(batchSize-1)
    datafile = open('{}{}.2.wm_{}-{}.csv'.format(folder, model_name, start, finish))
    np_file = np.load('{}{}.2.wm_{}-{}.vals.npy'.format(folder, model_name, start, finish))
    file_reader = DictReader(datafile)
    if i==0:
        headers = file_reader.fieldnames
        file_writer = DictWriter(out_file, headers)
        file_writer.writeheader()
        vals = np_file
    else:
        vals = np.append(vals, np_file, axis=0)
    for line in file_reader:
        file_writer.writerow(line)
    datafile.close()
np.save('{}{}.2.wm.vals'.format(folder, model_name), vals)
out_file.close()