import datetime
import numpy as np
import matplotlib.pylab as plt
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh_x = L.Linear(hidden_size, 4 * hidden_size),
            hh_h = L.Linear(hidden_size, 4 * hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
        self.hidden_size = hidden_size

    def __call__(self, x, t=None, train=False):
        if self.h is None:
            self.h = Variable(np.zeros((x.shape[0], self.hidden_size), dtype="float32"))
            self.c = Variable(np.zeros((x.shape[0], self.hidden_size), dtype="float32"))
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh_x(h) + self.hh_h(self.h)
        self.c, self.h = F.lstm(self.c, h)
        y = self.hy(self.h)
        if train:
            return F.mean_squared_error(y, t)
        else:
            return y.data

    def reset(self):
        self.zerograds()
        self.h = None
        self.c = None

EPOCH_NUM = 1000
HIDDEN_SIZE = 5
BATCH_ROW_SIZE = 3 
BATCH_COL_SIZE = 3 
LOOK_BACK = 3

train_data = np.array([[np.sin(i*2*np.pi/10) for i in range(10)],
    [np.sin(i*2*np.pi/10) for i in range(10)]])
train_data=train_data.T
print("train_data",train_data)

train_x, train_t = [], []
for i in range(train_data.shape[0]-LOOK_BACK-1):
    set_train_x = []
    for j in range(train_data.shape[1]):
        set_train_x.append(train_data[i:i+LOOK_BACK,j])
    train_t.append(train_data[i+LOOK_BACK,0])
    train_x.append(set_train_x)

train_x = np.array(train_x, dtype="float32")
train_t = np.array(train_t, dtype="float32")

train_x = np.resize(train_x,train_data.shape)

#print("trainx",train_x)
#print("traint",train_t)

in_size = 2
out_size = 1
N = train_data.shape[0]
print("N",N)

model = LSTM(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size)
optimizer = optimizers.Adam()
optimizer.setup(model)

print("Train")
st = datetime.datetime.now()
for epoch in range(EPOCH_NUM):
    print("epoch",epoch)

    x, t = [], []
    for i in range(BATCH_ROW_SIZE):
        set_x = []
        index = np.random.randint(0, N-BATCH_COL_SIZE+1) 
        for j in range(train_data.shape[1]):
            #print("trainx",train_x[j,index:index+BATCH_COL_SIZE]) 
            set_x.append(train_x[index:index+BATCH_COL_SIZE,j]) 
        #t.append(train_t[index:index+BATCH_COL_SIZE,0])
        t.append(train_t[index:index+BATCH_COL_SIZE])
        x.append(set_x)
    x = np.array(x, dtype="float32")
    x=np.resize(x,(BATCH_COL_SIZE,in_size))
    t=np.resize(t,(BATCH_COL_SIZE,out_size))
    print("x",x)
    print("t",t)
    raw_input()
    t = np.array(t, dtype="float32")
    loss = 0
    total_loss = 0
    model.reset() 
    for i in range(BATCH_COL_SIZE): 
        x_ = np.array([x[j, i] for j in range(BATCH_ROW_SIZE)], dtype="float32")[:, np.newaxis] 
        t_ = np.array([t[j, i] for j in range(BATCH_ROW_SIZE)], dtype="float32")[:, np.newaxis] 
        loss += model(x=x_, t=t_, train=True)
    loss.backward()
    loss.unchain_backward()
    total_loss += loss.data
    optimizer.update()
    if (epoch+1) % 100 == 0:
        ed = datetime.datetime.now()
        print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
        st = datetime.datetime.now()

print("\nPredict")
predict = np.empty(0) 
inseq_size = 50
inseq = train_data[:inseq_size] 
for _ in range(N - inseq_size):
    model.reset() 
    for i in inseq: 
        x = np.array([[i]], dtype="float32")
        y = model(x=x, train=False)
    predict = np.append(predict, y) 

    inseq = np.delete(inseq, 0)
    inseq = np.append(inseq, y)

plt.plot(range(N+1), train_data, color="red", label="t")
plt.plot(range(inseq_size+1, N+1), predict, color="blue", label="y")
plt.legend(loc="upper left")
plt.show()
