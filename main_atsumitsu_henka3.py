"""
pinn_test : PINN Matlab tutorial conversion
created by Saida (Univercity of Tsukuba)
reference code
Matlab tutorial url : https://jp.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html?searchHighlight=PINN&s_tid=srchtitle_PINN_1#TrainPhysicsInformedNeuralNetworkWithLBFGSAndAutoDiffExample-7
github : https://github.com/jayroxis/PINNs

日本語の説明
このコードは上記二つのコードを参考に書いてあります。
DNN部分と最適化部分はPytorchで実装されています。
最適化はLBFGSBとAdamに対応していますが、とりあえずこのままだとAdamで動きます。
LBFGSBに関しては、なんか間違ってる気がするので、Adamが推奨です。
Burgers方程式に関しては、以下コードをそのまま持ってきています。
github : https://github.com/jkfids/pinn-burgers/blob/main/burgers1d/analytical.py
一応、パラメータの数はmatlabと一致することを確認してあり、DNNの構造に関しては確認してあります。
DNN構造を変えたい場合はLayerに与えるリストを変更することで、構造を変えることができます。
かなりMatlabコードに寄せてあります。
"""
import sys
import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torchsummary import summary
import calc_atsumitsu
import pyDOE2 as doe
from tensorboardX import SummaryWriter
import tqdm
warnings.filterwarnings('ignore')

global SIM    #相似則
SIM=10000
LOAD=10000/SIM #10,000 Pa
TIME=100000/SIM #100,000 sec

# CUDA がサポートされているかを確認します。
# サポートされていなければ、cpuで動きます。
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("gpu")
else:
    device = torch.device('cpu')
    print("cpu")

writer = SummaryWriter()

# DNNのクラスです。
# DNNを作ります。
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

# PINNのクラスです。
# PINNの設定やトレーニング等があります。
class PINN():
    def __init__(self, layers, x, t, x0, t0, u0):
        self.cv = 0.00002*SIM
        # data
        self.x = torch.tensor(x, requires_grad=True).float().to(device)
        self.t = torch.tensor(t, requires_grad=True).float().to(device)
        self.x0 = torch.tensor(x0, requires_grad=True).float().to(device)
        self.t0 = torch.tensor(t0, requires_grad=True).float().to(device)
        self.u0 = torch.tensor(u0).float().to(device)
        self.x = self.x.view(self.x.shape[0], 1)
        self.t = self.t.view(self.t.shape[0], 1)
        self.x0 = self.x0.view(self.x0.shape[1], 1)
        self.t0 = self.t0.view(self.t0.shape[1], 1)
        self.u0 = self.u0.view(self.u0.shape[1], 1)

        # self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        # self.lambda_1 = torch.nn.Parameter(self.lambda_1)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        # self.dnn.register_parameter('lambda_1', self.lambda_1)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(),lr=0.001)
        self.iter = 0
        summary(self.dnn,input_size=(torch.cat([self.x, self.t], dim=1).shape))

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - self.cv*u_xx
        return f

    def loss_func(self):
        u_pred = self.net_u(self.x0, self.t0)
        f_pred = self.net_f(self.x, self.t)
        loss_u = torch.mean((self.u0 - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        loss = loss_u + loss_f
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'It: %d, Loss: %e, lossU: %e, lossF: %e' %
                (
                    self.iter,
                    loss.item(),
                    loss_u.item(),
                    loss_f.item()
                )
            )
        return loss
    Loss = ['loss']                   ###損失関数の表示、

        
    # 学習のオプション
   # batch_size = 50
    #numEpochs = 100

    def train(self, nIter, optimizer="adam", batch_size=50, nEpoch=100):
        self.dnn.train()
        #dataset1 = torch.utils.data.TensorDataset(self.x0,self.t0,self.u0)
        dataset2 = torch.utils.data.TensorDataset(self.x,self.t)
        #data_loader1 = torch.utils.data.DataLoader(dataset1,batch_size=batch_size,shuffle=True)
        data_loader2 = torch.utils.data.DataLoader(dataset2,batch_size=batch_size,shuffle=True)
        if(optimizer=="adam"):
            for epoch in range(nEpoch):
                for i,(x,t) in enumerate(data_loader2):
                    u_pred = self.net_u(self.x0, self.t0)
                    f_pred = self.net_f(x, t)
                    loss_u = torch.mean((self.u0 - u_pred) ** 2)
                    loss_f = torch.mean(f_pred ** 2)
                    loss = loss_u + loss_f
                    # Backward and optimize
                    self.optimizer_Adam.zero_grad()
                    loss.backward()
                    self.optimizer_Adam.step()
                writer.add_scalar("Loss", loss.item(), epoch)
                writer.add_scalar("LossU", loss_u.item(), epoch)
                writer.add_scalar("LossF", loss_f.item(), epoch)
                print(
                        'Epoch: %d, Loss: %.3e, lossU: %.3e, lossF: %.3e' %
                        (
                            epoch,
                            loss.item(),
                            loss_u.item(),
                            loss_f.item()
                        )
                    )
            # for iter in range(nIter):
            #     u_pred = self.net_u(self.x0, self.t0)
            #     f_pred = self.net_f(self.x, self.t)
            #     loss_u = torch.mean((self.u0 - u_pred) ** 2)
            #     loss_f = torch.mean(f_pred ** 2)
            #     loss = loss_u + loss_f

            #     # Backward and optimize
            #     self.optimizer_Adam.zero_grad()
            #     loss.backward()
            #     self.optimizer_Adam.step()

            #     writer.add_scalar("Loss", loss.item(), iter)
            #     writer.add_scalar("LossU", loss_u.item(), iter)
            #     writer.add_scalar("LossF", loss_f.item(), iter)
            #     if iter % 100 == 0:
            #         print(
            #             'It: %d, Loss: %.3e, lossU: %.3e, lossF: %.3e' %
            #             (
            #                 iter,
            #                 loss.item(),
            #                 loss_u.item(),
            #                 loss_f.item()
            #             )
            #         )
        if(optimizer=="lbfgs"):
            self.optimizer.step(self.loss_func)

    def predict(self, x,t):
        x = torch.tensor(x, requires_grad=True).float().to(device)
        t = torch.tensor(t, requires_grad=True).float().to(device)
        x = x.view(x.shape[0], 1)
        t = t.view(t.shape[0], 1)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

    @staticmethod
    # 標準はsobol列ですが、LHSもサポートしておきます。
    def sampling(sampling_num, method="sobol"):
        if(method=="sobol"):
            soboleng = torch.quasirandom.SobolEngine(dimension=2)
            points = soboleng.draw(sampling_num)
            return points
        elif(method=="lhs"):
            lhs = doe.lhs(2, samples=sampling_num, criterion='center', random_state=0)
            return lhs
        else:
            print("Error: Choose sampling method", file=sys.stderr)

if __name__=="__main__":
    # create training data
    # データの設定をします。
    # 初期条件や境界条件はここで反映させます。
    # なお、ここを変更する際はburgers方程式の計算コード(calc_burgers.py)も変更した方が良いと思います。
    num_boundary_condition_points = [25, 25]
    x0BC1 = np.zeros([1, num_boundary_condition_points[0]])
    x0BC2 = 5.0*np.ones([1, num_boundary_condition_points[1]])
    #t0BC1 = np.linspace(0, 10000, num_boundary_condition_points[0]).reshape([1, num_boundary_condition_points[0]])
    #t0BC2 = np.linspace(0, 10000, num_boundary_condition_points[1]).reshape([1, num_boundary_condition_points[1]])
    
    t0BC1 = np.linspace(1/100, TIME, num_boundary_condition_points[0]).reshape([1, num_boundary_condition_points[0]])
    t0BC2 = np.linspace(1/100, TIME, num_boundary_condition_points[1]).reshape([1, num_boundary_condition_points[1]])
    u0BC1 = np.zeros([1, num_boundary_condition_points[0]])
    u0BC2 = np.zeros([1, num_boundary_condition_points[1]])

    numInitialConditionPoints = 50
    #load = 10
    x0IC = np.linspace(0.0, 5.0, numInitialConditionPoints).reshape([1, numInitialConditionPoints])
    t0IC = np.zeros([1, numInitialConditionPoints])
    u0IC = LOAD * np.ones([1, numInitialConditionPoints])
    u0IC[0] = 0
    u0IC[-1] = 0

    x0 = np.concatenate([x0IC, x0BC1, x0BC2], axis=-1)
    t0 = np.concatenate([t0IC, t0BC1, t0BC2], axis=-1)
    u0 = np.concatenate([u0IC, u0BC1, u0BC2], axis=-1)
    # print(x0.shape)
    # x = np.arange(0.0, 5.0, 0.5)
    # t = np.arange(0.0, 10000, 100)
    # T, X = np.meshgrid(t, x)
    #
    # u_true = np.zeros(X.shape)
    # i = 0
    # print("数値解の計算")
    # for tt in tqdm.tqdm(t):
    #     u_true[:, i] = calc_atsumitsu.calc_u(tt, x)
    #     i = i + 1
    # x0 = X.flatten().reshape([1, X.flatten().shape[0]])
    # t0 = T.flatten().reshape([1, T.flatten().shape[0]])
    # u0 = u_true.flatten().reshape([1, u_true.flatten().shape[0]])
    # print(x0.shape)
    # print(u0.shape)
    
    numInternalCollocationPoints1 = 1000  ###local
    TIME_L = 10000/SIM
    points = PINN.sampling(sampling_num=numInternalCollocationPoints1, method="sobol")   # or "lhs"
    dataX1 = np.array(5.0 * points[:, 0])
    dataT1 = np.array(TIME_L * points[:, 1])
    
    numInternalCollocationPoints2 = 9000  ###global
    points = PINN.sampling(sampling_num=numInternalCollocationPoints2, method="sobol")   # or "lhs"
    dataX2 = np.array(5.0 * points[:, 0])
    dataT2 = np.array(TIME * points[:, 1])
    
    # dataX = dataX1.append(dataX2)
    # dataT = dataT1.append(dataT2)
    dataX = np.concatenate([dataX1, dataX2])
    dataT = np.concatenate([dataT1, dataT2])
    
    #plt.hist(dataT)
    #plt.show()
    plt.scatter(dataT*SIM,dataX,s=1)      ##コロケーションポイントの分布図
    #plt.rcParams["font.family"] = "Times New Roman"
    #font_num = 22
    #parameters = {'axes.labelsize':22 , 'axes.titlesize':22 , 'figure.titlesize':22 , 'xtick.labelsize':22 , 'labelsize':22}
    #plt.rcParams.update(parameters)
    plt.xlabel("Time [sec]")
    plt.ylabel("Distance [m]")
    plt.title("CollocationPoint")
    plt.show()
 
    dataX.reshape(dataX.shape[0],1)
    dataT.reshape(dataT.shape[0],1)
    #numInternalCollocationPoints = 1000000
    #points = PINN.sampling(sampling_num=numInternalCollocationPoints, method="sobol")   # or "lhs"
    #dataX = np.array(5.0 * points[:, 0])
    #dataT = np.array(10000 * -0.07*np.log(1-points[:, 1]))
    #plt.hist(dataT)
    #plt.show()
    #dataX.reshape(dataX.shape[0],1)
    #dataT.reshape(dataT.shape[0],1)

    # training
    # モデルを作ってトレーニングをします。
    # レイヤーの設定さえすれば、とりあえずモデルは作れます。
    # トレーニングに関しては、反復回数と手法を決めます。
    # なお、PINNのモデル等はPINNクラスにまとめてあります。
    layers = [2]    # 入力次元数は2
    layer_num = 9
    nn_num = 20
    for i in range(layer_num-1):
        layers.append(nn_num)
    layers.append(1)    # 出力次元数は1
    #model = PINN(layers, dataX, dataT, x0, t0, u0)
    model = PINN(layers, dataX, dataT, x0, t0, u0)
    model.train(nIter=10, optimizer="adam", batch_size=50, nEpoch=100)  # or "lbfgs"
    writer.close()
    # evaluations
    # モデルの評価を行います。
    # この部分はMatlabのチュートリアルと同じです。
     #tTest = [10, 100, 1000, 10000]
     #numPredictions = 1001
     #XTest = np.linspace(0.0, 5.0, numPredictions)
     #fig = plt.figure()
     #plt.subplots_adjust(wspace=0.3, hspace=0.4)
     #for i in range(len(tTest)):
     #    t = tTest[i]
     #    TTest = t * np.ones([numPredictions, 1])
     #    u_pred, f_pred = model.predict(XTest, TTest)
     #    u_test = calc_atsumitsu.calc_u(t,XTest).reshape([u_pred.shape[0],1])
     #    err = np.linalg.norm(u_pred-u_test, 2)/np.linalg.norm(u_test, 2)
     #    print('t : %.1f \nErr : %f' % (t, err))
     #    plt.subplot(2,2,i+1)
     #    plt.title('t : %.1f, Err : %f' % (t, err))
     #    plt.plot(XTest, u_pred)
     #    plt.plot(XTest, u_test, linestyle="dashed")
     #plt.subplot(2,2,2)
     #plt.legend(['predict', 'true'])
     #writer.add_figure('result', fig)
     #plt.savefig("result.png")
     #plt.show()
     
    Loss = ['loss'] 
    plt.figure(figsize=(10,5))        ###グラフを表示するスペースを用意
    for i in range(len(Loss)):
        Loss_num = Loss[i]
        plt.title(Loss)
        plt.plot(Loss, label='loss function')
        plt.legend()
        plt.show()

    # ここより下はヒートマップを作ります。
    x = np.arange(0.0,5.0,0.05)
    t = np.linspace(0.0,TIME*SIM,100)#np.arange(0.0,TIME,100)
    T,X = np.meshgrid(t,x)
    print(T)
    u_pred,f_pred = model.predict(X.flatten().reshape([X.shape[0]*X.shape[1],1]),T.flatten().reshape([T.shape[0]*T.shape[1],1]))
    u_pred=SIM * u_pred.reshape(X.shape)
    print(u_pred.shape)
    print(X.shape)
    plt.pcolormesh(T,X,u_pred*SIM)
    pp = plt.colorbar()
    pp.set_label("Pressure [Pa]")
    plt.title("PINNs")
    plt.xlabel("Time [sec]")
    plt.ylabel("Distance [m]")
    plt.show()


    u_true = np.zeros(X.shape)
    i=0
    print("数値解の計算")
    for tt in tqdm.tqdm(t):
        u_true[:,i] = calc_atsumitsu.calc_u(tt,x)
        i=i+1
    plt.pcolormesh(T, X, u_true)
    pp=plt.colorbar()
    pp.set_label("Pressure [Pa]")
    plt.title("PDE")
    plt.xlabel("Time [sec]")
    plt.ylabel("Distance [m]")
    plt.show()
    #
    # data = scipy.io.loadmat('burgers_shock.mat')
    #
    # t = data['t'].flatten()[:, None]
    # x = data['x'].flatten()[:, None]
    # Exact = np.real(data['usol']).T
    #
    # X, T = np.meshgrid(x, t)
    # print(x.shape)
    # print(X.flatten().shape)
    # print(dataX.shape)
    # t = np.random.randn(100).flatten()[:, None]
    # x = (-1+2*np.random.randn(100)).flatten()[:, None]
    # X, T = np.meshgrid(x, t)
    # X_star = np.hstack((dataX.flatten()[:, None], dataT.flatten()[:, None]))
    # print(X.flatten().shape)
    # u_pred, f_pred = model.predict(X.flatten(),T.flatten())
    # print(X)
    # print(u_pred)
    # u_pred = u_pred.reshape(X.shape)
    # print(u_pred.shape)
    #U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='nearest')


    ####### Row 0: u(t,x) ##################
    #
    # fig = plt.figure(figsize=(9, 5))
    # ax = fig.add_subplot(111)
    # #ax.pcolormesh(X, T, u_pred, cmap='rainbow')
    # # fig.colorbar()
    # h = ax.imshow(u_pred.T, interpolation='bicubic', cmap='rainbow',
    #               extent=[dataT.min(), dataT.max(), dataX.min(), dataX.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.10)
    # cbar = fig.colorbar(h, cax=cax)
    # cbar.ax.tick_params(labelsize=15)
    #
    # ax.plot(
    #     dataT,
    #     dataX,
    #     'kx', label='Data (%d points)' % (dataX.shape[0]),
    #     markersize=4,  # marker size doubled
    #     clip_on=False,
    #     alpha=.5
    # )
    #
    # ax.set_xlabel('$t$', size=20)
    # ax.set_ylabel('$x$', size=20)
    # ax.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.9, -0.05),
    #     ncol=5,
    #     frameon=False,
    #     prop={'size': 15}
    # )
    # ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
    # ax.tick_params(labelsize=15)
    #
    # plt.show()
