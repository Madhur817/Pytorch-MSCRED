import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

parser = argparse.ArgumentParser(description = 'MSCRED evaluation')
parser.add_argument('--thred_broken', type = int, default = 0.005,
                   help = 'broken pixel thred')
parser.add_argument('--alpha', type = int, default = 1.5,
                   help = 'scale coefficient of max valid anomaly')
parser.add_argument('--valid_start_point',  type = int, default = 8000,
                        help = 'test start point')
parser.add_argument('--valid_end_point',  type = int, default = 10000,
                        help = 'test end point')
parser.add_argument('--test_start_point',  type = int, default = 10000,
                        help = 'test start point')
parser.add_argument('--test_end_point',  type = int, default = 20000,
                        help = 'test end point')
parser.add_argument('--gap_time', type = int, default = 10,
                   help = 'gap time between each segment')
parser.add_argument('--matrix_data_path', type = str, default = './data/matrix_data/',
                   help='matrix data path')
parser.add_argument('--low',type=int,default=10000)
parser.add_argument('--hi',type=int,default=20000)
# parser.add_argument('--raw_data_path', type = str, default = './data/synthetic_data_with_anomaly-s-1.csv',help='path to load raw data')


args = parser.parse_args()
print(args)

thred_b = args.thred_broken
alpha = args.alpha
gap_time = args.gap_time
valid_start = args.valid_start_point//gap_time
valid_end = args.valid_end_point//gap_time
test_start = args.test_start_point//gap_time
test_end = args.test_end_point//gap_time

up = args.low
down = args.hi

valid_anomaly_score = np.zeros((valid_end - valid_start , 1))
test_anomaly_score = np.zeros((test_end - test_start, 1))

matrix_data_path = args.matrix_data_path
test_data_path = matrix_data_path + "test_data/"
reconstructed_data_path = matrix_data_path + "reconstructed_data/"

criterion = torch.nn.MSELoss()

output_path = "./outputs/" 
if not os.path.exists(output_path):
    os.makedirs(output_path)


attention_wts = []
reconstructed_matrix = []
for i in range(valid_start, test_end):
    path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
    gt_matrix_temp = np.load(path_temp_1)

    path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(i) + '.npy')
    path_temp_3 = os.path.join(reconstructed_data_path, "attention_wts_" + str(i) + '.npy')
    #path_temp_2 = os.path.join(reconstructed_data_path, "pcc_matrix_full_test_" + str(i) + '_pred_output.npy')
    reconstructed_matrix_temp = np.load(path_temp_2)
    attention_matrix = np.load(path_temp_3)

    # print(reconstructed_matrix_temp.shape)
    #first (short) duration scale for evaluation  
    select_gt_matrix = np.array(gt_matrix_temp)[-1][0] #get last step matrix

    select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]

    #compute number of broken element in residual matrix
    select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
    num_broken = len(select_matrix_error[select_matrix_error > thred_b])

    #print num_broken
    if i < valid_end:
        valid_anomaly_score[i - valid_start] = num_broken
    else:
        test_anomaly_score[i - test_start] = num_broken
        attention_wts.append(attention_matrix)
        reconstructed_matrix.append(reconstructed_matrix_temp)

attention_wts = np.array(attention_wts)
reconstructed_matrix = np.array(reconstructed_matrix)

valid_anomaly_max = np.max(valid_anomaly_score.ravel())
print("Threshold",valid_anomaly_max*alpha)
test_anomaly_score = test_anomaly_score.ravel()


t = 10000+ np.arange(1000) * 10

for i in range(4):
    plt.figure(figsize=(10,6))
    ser = attention_wts[:,i,:]
    for j in range(ser.shape[1]):
        plt.plot(t,ser[:,j],label="(-"+str(4-j)+")")
    plt.xlim(left=up,right=down)
    plt.legend(loc="upper right")
    plt.savefig(output_path+"attention_layer_"+str(i+1))

fig, axes = plt.subplots(figsize=(10,6))

test_num = test_end - test_start
axes.plot(t,test_anomaly_score, color = 'black', linewidth = 2)
threshold = np.full((test_num), valid_anomaly_max * alpha)
axes.plot(t,threshold, color = 'black', linestyle = '--',linewidth = 2)

raw_data_path = './data/sim_data.csv'
data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)

y = [i for i in  range(data.shape[1])]

ax2 = axes.twinx()
for i in range(data.shape[0]):
    ax2.plot(y,data[i,:])
plt.xlim(left=up,right=down)
plt.xlabel('Test Time')
axes.set_ylabel('Anomaly Score')
ax2.set_ylabel('Series')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.25)
fig.subplots_adjust(left=0.25)
# plt.title("MSCRED", size = 25)

plt.savefig(output_path+'anomaly_score.jpg')
plt.show()
