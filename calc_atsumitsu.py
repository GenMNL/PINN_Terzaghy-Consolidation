import matplotlib.pyplot as plt
import numpy as np
import tqdm

# パラメータ
z_max = 5.0
z_min = 0.0
t_max = 100000
dz = 0.1
dt = 50.0
cv = 0.00002
load = 10000
z_num = int((z_max-z_min)/dz)+1
alpha = cv*dt/dz/dz

def calc_u(t,z):
    dz = (z_max-z_min)/z.shape[0]
    alpha = cv * dt / dz / dz
    # 初期条件
    u0 = load * np.ones(z.shape[0])
    u0[0] = 0
    u0[-1] = 0
    # 行列Aの作成
    mat_a = np.zeros([z.shape[0],z.shape[0]])
    for i in range(mat_a.shape[0]):
        for j in range(mat_a.shape[1]):
            if (i == j):
                mat_a[i, j] = 1 - 2 * alpha
            elif (j == i + 1 or i == j + 1):
                mat_a[i, j] = alpha
    t_num = int(100000/dt)   # 一応、10万secまで対応しておく
    u_list = []
    u_temp = u0
    u_list.append(u0)
    for tt in range(t_num):
        u_temp = np.dot(mat_a, u_temp)
        u_temp[0] = 0   # 境界条件
        u_temp[-1] = 0  # 境界条件
        u_list.append(u_temp)

    return u_list[int(t/dt)]

if __name__=="__main__":
    x = np.linspace(0.0, z_max, 100)
    t = np.linspace(0.0, t_max, 100)
    T, X = np.meshgrid(t, x)
    u_true = np.zeros(X.shape)
    print("数値解の計算")
    i = 0
    for tt in tqdm.tqdm(t):
        u_true[:, i] = calc_u(tt, x)
        i = i + 1
    plt.pcolormesh(T, X, u_true)
    pp = plt.colorbar()
    pp.set_label("Pressure [Pa]")
    plt.title("PDE")
    plt.xlabel("Time [sec]")
    plt.ylabel("Distance [m]")
    plt.show()
