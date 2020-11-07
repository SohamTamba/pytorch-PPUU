import torch, pickle
import numpy as np
import matplotlib.pyplot as plt

predicted = pickle.load(open("predicted_img.p", "rb")).cpu().detach().numpy()
real = pickle.load(open("real_img.p", "rb")).cpu().detach().numpy()/255

print(predicted.shape)
print(real.shape)

bin_size = 0.1
eps = 1e-10
bins = np.concatenate([\
    np.array([-eps, eps]) , np.arange(bin_size, 1, bin_size) \
    ,  np.array([1-eps, 1+eps]), np.arange(1+bin_size, 1000+bin_size, 1)\
    ])
print(bins[:20])

real_hist, _ = np.histogram(real, bins)
pred_hist, _ = np.histogram(predicted, bins)
X = [ (bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

print("Range     | Real | Predicted")
for i in range(len(X)):
    if real_hist[i] != 0 or pred_hist[i] != 0:
        print("{:.2f}:{:.2f} | {} | {}".format((bins[i]), (bins[i+1]), real_hist[i], pred_hist[i]))

#plt.plot(X, real_hist, label="Real Pixels")
#plt.plot(X, pred_hist, label="Predicted Pixels")
#plt.xlabel("Pixel Values")
#plt.ylabel("Frequency")
#plt.legend()

plt.show()
