import matplotlib.pyplot as plt
import pickle
import numpy

#train_diff_stats = pickle.load(open( "saved_models/train_diff_stats.p", "rb" ))
val_diff_stats = pickle.load(open( "saved_models/val_diff_stats.p", "rb" ))

label_dict = dict(
    max_a="Max Acc",
    max_s="Max Steer",
    median_a="Median Acc",
    median_s="Median Steer",
    mean_a="Mean Acc",
    mean_s="Mean Steer"
)

def plot_stats(diff_stats, title, outpath):

    def is_steering(key):
        return key[-1] == 's'
    def is_acceleration(key):
        return key[-1] == 'a'


    plt.title(title.format(action="Acceleration"))
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    for k in diff_stats.keys():
        if is_acceleration(k):
            Y = diff_stats[k]
            plt.plot(numpy.arange(len(Y)), Y, label=label_dict[k])
    plt.legend()
    plt.savefig(outpath.format(action="acc"))
    plt.close()

    plt.title(title.format(action="Steering"))
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    for k in diff_stats.keys():
        if is_steering(k):
            Y = diff_stats[k]
            plt.plot(numpy.arange(len(Y)), Y, label=label_dict[k])
    plt.legend()
    plt.savefig(outpath.format(action="steer"))
    plt.close()


#plot_stats(train_diff_stats, "{action} Difference on Training Set", "savd_models/train_{action}.jpg")
plot_stats(val_diff_stats, "{action} Difference on Validation Set", "saved_models/val_{action}.jpg")
