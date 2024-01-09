import numpy as np
import joblib
import matplotlib
from matplotlib import pyplot as plt

Nh = 1400000

# The number of infected dogs and infected individuals in different scenarios
Nd_list = [Nh / 70, Nh / 35, Nh / 20, Nh / 10, Nh / 5, Nh / 3]
Ih_list = [460, 900, 1600, 3100, 6300, 10600]


# Disease burden=number of cases * coefficient
# Rate=(original disease burden - current disease burden)/cost

def cost_benefit(measure: int, scene: int, param, object_value, time):
    '''

    :param measure:
    :param scene:
    :param param:  float/list
    :param object_value: Health goals (number of infections), int
    :param time: Number of years of intervention(1, 3, 5), int
    :return:
    '''

    measure = str(measure)
    measure_param = []
    scene = scene - 1
    '''
    Determine if the coverage rate is empty. If the coverage rate is empty,
    it indicates optimization failure and does not calculate costs
    '''
    if ~np.isnan(param[0]):
        i = 0
        while i < len(param):
            measure_param.append(param[i:i + 2])
            i = i + 2
        total_cost = 0
        for item, meas in enumerate(measure):
            cost = calcu_cost(meas, measure_param[item], scene, time)
            total_cost = total_cost + cost
        print("total_cost", total_cost)
        ratio = (Ih_list[scene] - object_value) / total_cost
    else:
        ratio = np.nan
    return ratio


# Calculate scene 2, using the cost of each intervention for 3 years
def calcu_cost(measure, measure_param, scene=1, time=3):
    cost = 0
    measure = str(measure)

    if measure == '1':
        cost = Nd_list[scene] * measure_param[0] * 20 * time

    elif measure == '2':
        cost = Nh * measure_param[0] * (0.6 + 2) * 36 * 2 * time

    elif measure == '3':
        cost = Nd_list[scene] * 120 + Nd_list[scene] * measure_param[0] * 50
    return cost


def calcu_rate(model, measure, measure_param):
    cost = calcu_cost(measure, measure_param)
    measure_param = np.array(measure_param).reshape(1, -1)
    out = np.squeeze(model.predict(measure_param))
    diff = (out[0] - out[2]) * 0.051
    # print(diff, cost)
    return diff / cost


plt.figure(figsize=(8, 5))
config = {
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 20,
    "axes.unicode_minus": False
}
matplotlib.rcParams.update(config)
Meas = [1, 2, 3]
titles = ["Sandfly repellent dog collars", "IRS1", "Culling of infected dogs"]
cov = np.linspace(0.01, 1, 100)
eff = [0.98, 0.85, 0.5]
recorder = []
linestyles = ['-', '-.', ':']
for i in range(len(Meas)):
    param = list(map(lambda x: [x, eff[i]], cov))
    path = "./DataFolder/ModelFile/Scene2/m" + str(Meas[i]) + "_GPmodel.pkl"
    gpr = joblib.load(path)
    recorder_item = []
    for j in param:
        recorder_item.append(calcu_rate(gpr, Meas[i], j))
        recorder.append(calcu_rate(gpr, Meas[i], j))
    plt.plot(cov, recorder_item, label=titles[i], color='black', linestyle=linestyles[i])

    # recorder.append(recorder_item)
print(recorder)
data = {
    "intervene_measure": np.concatenate((np.ones((len(cov),)), np.full((len(cov),), 2), np.full((len(cov),), 3)),
                                        axis=0),
    "cov": np.concatenate((cov.reshape(-1), cov.reshape(-1), cov.reshape(-1)), axis=0),
    "value": recorder
}
# data = pd.DataFrame(data)
# plot_curve(data)
# data.to_csv("./DataFolder/ResultFile/CostBenefit.csv", index=False)
plt.xlabel("Coverage")
plt.ylabel("Effectiveness/cost")
# plt.xlim(0, 1)
plt.legend()
plt.savefig('./DataFolder/FigureFile/benefit_cost.png', dpi=500)
