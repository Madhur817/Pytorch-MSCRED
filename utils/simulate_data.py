import numpy as np
from matplotlib import pyplot as plt


def simulate(length,t0,w):
    x = np.arange(length)
    t0 = np.array(t0,dtype=int)
    w = np.array(w,dtype=int)
    y = (np.expand_dims(x, 0) - np.expand_dims(t0, 1))/np.expand_dims(w, 1)

    y = np.sin(y)
    return y


if __name__ == '__main__':
    length = 20000
    t0 = [0,10,30,60,110,220]
    w = [40,40,40,40,75,97]

    ser = simulate(length,t0,w)
    ser[0,15100:15110] = 0
    ser[1,15110:15120] = 0
    ser[2,15130:15140] = 0
    ser[3,15160:15170] = 0

    ser[0,16100:16110] = ser[0,16100:16110]+0.8
    ser[1,16110:16120] = ser[1,16110:16120]+0.8
    ser[2,16130:16140] = ser[2,16130:16140]+0.8
    ser[3,16160:16170] = ser[3,16160:16170]+0.8
    t = np.arange(length)
    up = 15000
    down = up+400
    plt.figure(figsize=(10,6))
    for i in range(ser.shape[0]):
        plt.plot(t[up:down],ser[i,up:down])
    plt.savefig("outputs/simulated_data1")
    plt.figure(figsize=(10,6))
    up = 16000
    down = up+400
    for i in range(ser.shape[0]):
        plt.plot(t[up:down],ser[i,up:down])
    plt.savefig("outputs/simulated_data2")
    plt.show()
    np.savetxt("data/sim_data.csv", ser, delimiter=",")
