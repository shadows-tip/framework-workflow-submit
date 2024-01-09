import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from Utils import save

# warnings.filterwarnings("ignore")
record = []


# objective function
def func(cov, *args):
    gpr, param, meas, object_value, year = args
    if isinstance(param, list):
        # Combination intervention measures
        inputs = np.array(cov.tolist() + param).reshape(1, -1)
    else:
        # Single intervention measures
        inputs = np.array([cov, param], dtype=object).reshape(1, -1)
    output = gpr.predict(inputs, return_std=False)[0, year]
    diff = output - object_value
    global record
    record.append(-1 if diff < 0 else 1)

    return np.abs(diff)


def optim(*args):
    # gpr, eff, object_value, year
    best_x, best_res = np.nan, 1
    for j in range(50):
        x0 = np.random.uniform(0, 1)
        result = minimize(func, x0, method='SLSQP', bounds=[[0, 1]], args=args)
        if result.fun < best_res:
            best_res = result.fun
            best_x = result.x[0]
    global record
    if record.count(-1) > record.count(1) and np.isnan(best_x):
        best_x = 0
    print("本次优化的最优值为:{:f}".format(best_x))
    record.clear()
    return best_x


def optim_running(meas, scene, param, objects: list, year: int):
    gpr_path = './DataFolder/ModelFile/Scene' + str(scene) + '/m' + str(meas) + '_GPmodel.pkl'
    gpr_model = joblib.load(gpr_path)
    out = []
    time_list = {"1": 1, "2": 3, "3": 5}
    for i, object_value in enumerate(objects):
        print("meas{:d}, scene{:d}, object:{:.2f}, year:{:d}, param:".format(meas, scene, object_value,
                                                                             time_list[str(year)]), param)
        optim_res = optim(gpr_model, param, meas, object_value, year)
        temp = [param] if isinstance(param, float) else param
        if len(str(meas)) == 1:
            out.append([meas, scene, object_value, optim_res, param, time_list[str(year)]])
        else:
            out.append([meas, scene, object_value, optim_res, '/'.join(list(map(str, temp))), time_list[str(year)]])
    return out


Scenes = [i for i in range(1, 7)]

# Time points for health goals, corresponding to the 1st, 3rd, and 5th years
times = [1, 2, 3]

# Seven intervention measures corresponding to seven sets of parameters
Meas = [1, 2, 3, 12, 13, 23, 123]
Params = [[0.8, 0.85, 0.9, 0.95],
          [0.8, 0.85, 0.9, 0.95],
          [0.1, 0.3, 0.5, 0.7],
          [[0.95, 0.6, 0.85], [0.95, 0.7, 0.85], [0.95, 0.8, 0.85], [0.95, 0.9, 0.85]],
          [[0.95, 0.3, 0.5], [0.95, 0.5, 0.5], [0.95, 0.7, 0.5], [0.95, 0.9, 0.5]],
          [[0.85, 0.3, 0.5], [0.85, 0.5, 0.5], [0.85, 0.7, 0.5], [0.85, 0.9, 0.5]],
          [[0.95, 0.6, 0.85, 0.5, 0.5], [0.95, 0.7, 0.85, 0.5, 0.5], [0.95, 0.8, 0.85, 0.5, 0.5],
           [0.95, 0.9, 0.85, 0.5, 0.5]]]

# Health goals (number of new infections)
Objects = [1, 5, 10, 15, 20]
Out = []

# Execute four cycles, intervention measures - time points - scene - parameters
for item, M in enumerate(Meas):
    Param = Params[item]
    for t, time in enumerate(times):
        for s, Scene in enumerate(Scenes):
            for p in Param:
                res = optim_running(M, Scene, p, Objects, time)
                Out.extend(res)

save(Out, './DataFolder/ResultFile/Optim.csv',
     columns=['interv_measure', 'scene', 'object_value', 'min_coverage', 'efficacy', 'year'])
