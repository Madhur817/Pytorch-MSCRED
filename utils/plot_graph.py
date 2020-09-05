import numpy as np
import argparse
import pandas as pd
# import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description = 'Plotting Time Series')
parser.add_argument('--raw_data_path', type = str, default = './data/synthetic_data_with_anomaly-s-1.csv',help='path to load raw data')

args = parser.parse_args()
print(args)

raw_data_path = args.raw_data_path

# cols = [1,2,3,4,6,16]

def plot(range_lo, range_hi,fig_name):
    data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)
    # data = data[cols,:]
    # print(data.shape)
    # print(data[0].shape)
    y = [i for i in  range(data.shape[1])]
    print(len(y))
    up = range_lo    # 11810
    down = range_hi
    fig = plt.figure(figsize=(25,10))
    for i in range(data.shape[0]):
        plt.plot(y[up:down],data[i,up:down])
    # fig.plot(y[up:down],x2[up:down])
    # fig.plot(y[up:down],x3[up:down])
    # fig.axvline(x=11810,color='red')
    # fig.axvspan(11800,11809, color='red', alpha=0.4)
    fig.savefig("./outputs/"+fig_name)
    fig.show()

if __name__ == '__main__':
    start = 14400
    dur = 200
    plot(start,start+dur,"check")

