import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from .setup import *


class Model:
    def __init__(self, T: int = 120, Nd=40000):
        '''

        :param Nd: The number of dogs in this area. Used to control the scene.
        :param T: Epidemic time of the disease, unit: year
        '''

        Sh0, Eh0, Ih0, Rh0, Sd0, Ed0, Id0, Rd0, Ss0, Es0, Is0, Nh, Nd, Ns = get_initial_data(Nd)

        self.Nh = Nh
        self.Ns = Ns
        self.Nd = Nd

        self.T = T
        # INI为初始状态下的数组
        self.INI = [Sh0, Eh0, Ih0, Rh0, Sd0, Ed0, Id0, Rd0, Ss0, Es0, Is0]
        self.param = None
        self.RES = None

        self.intervene_meas = None
        self.intervene_time = None

    def set_param(self, intervene_meas: int = None, intervene_time=None, intervene_param=None):
        '''

        :param intervene_meas: int, For example: 1 represents intervention measure 1
        :param intervene_time: unit: year
        :param intervene_param: The parameter values corresponding to the intervention measures are consistent with the
                                intervention_meas order corresponds to
        '''

        if intervene_time is not None and intervene_meas is not None:
            self.intervene_time = intervene_time - 1
            self.intervene_meas = str(intervene_meas)
            intervene_param = self.param_assign(intervene_param)
        initial_data = [self.intervene_time, self.Nh, self.Nd, self.Ns]
        self.param = (get_fixed_param(), intervene_param, initial_data)

    def param_assign(self, intervene_param):
        beta, gamma, epsilon = 0, 0, 0
        if intervene_param is not None:
            if len(intervene_param) == 2 * len(self.intervene_meas):
                intervene_param = list(map(lambda x, y: x * y, intervene_param[::2], intervene_param[1::2]))
            for i in range(len(self.intervene_meas)):
                if self.intervene_meas[i] == '1':
                    beta = intervene_param[i]
                elif self.intervene_meas[i] == '2':
                    gamma = intervene_param[i]
                elif self.intervene_meas[i] == '3':
                    epsilon = intervene_param[i]
        return [beta, gamma, epsilon]

    @staticmethod
    def ode_func(t, x, *param):

        fixed_param, var_param, initial_data = param
        miuh, alphah, bh, deltah, faih, rouh, miud, alphad, bd, deltad, faid, roud, mius, cy, tao, ch, cd = fixed_param
        intervene_time, Nh, Nd, Ns = initial_data

        n = int(t / 365)
        if (n * 365 + 142) < t < (n * 365 + 283):
            hyq = 1
        else:
            hyq = 0

        if intervene_time is not None:
            if t / 365 <= intervene_time:
                beta, gamma, epsilon = 0, 0, 0  # Indicating no intervention measures will be taken
            else:
                epsilon = var_param[2]
                if (n * 365 + 142) < t < (n * 365 + 283):
                    beta, gamma = var_param[0], var_param[1]
                else:
                    beta, gamma = 0, 0
        else:
            beta, gamma, epsilon = 0, 0, 0

        # differential equation
        Sh, Eh, Ih, Rh, Sd, Ed, Id, Rd, Ss, Es, Is = x
        dSh = miuh * (Eh + Ih + Rh) + alphah * Ih - hyq * bh * ch * Sh * Is / Nh
        dEh = hyq * bh * ch * Sh * Is / Nh - (miuh + deltah + faih) * Eh
        dIh = faih * Eh - (miuh + alphah + rouh) * Ih
        dRh = rouh * Ih + deltah * Eh - miuh * Rh
        dSd = miud * (Ed + Id + Rd) + alphad * Id - hyq * bd * (1 - beta) * cd * Sd * Is / Nd
        dEd = hyq * bd * (1 - beta) * cd * Sd * Is / Nd - (miud + deltad + faid) * Ed
        dId = faid * Ed - (alphad + miud + roud) * Id * (1 + epsilon)
        dRd = roud * Id + deltad * Ed - miud * Rd * (1 + epsilon)
        dSs = (1 + gamma) * mius * (Es + Is) - hyq * (1 - beta) * cd * cy * Id * Ss / Ns
        dEs = hyq * (1 - beta) * cd * cy * Id * Ss / Ns - (1 + gamma) * mius * Es - Es / tao
        dIs = Es / tao - (1 + gamma) * mius * Is

        return [dSh, dEh, dIh, dRh, dSd, dEd, dId, dRd, dSs, dEs, dIs]

    def ode_solver(self):
        t = self.T * 365
        t_eval = [i for i in range(t)]
        self.RES = solve_ivp(fun=self.ode_func, t_span=[0, t], y0=self.INI, t_eval=t_eval, args=self.param)

    def redirect_index(self, out_label):
        # 'New_Ih' indicates a new infection case
        index_dict = pd.Series(
            {'Sh': 0, 'Eh': 1, 'Ih': 2, 'Rh': 3, 'Sd': 4, 'Ed': 5, 'Id': 6, 'Rd': 7, 'Ss': 8, 'Es': 9, 'Is': 10,
             'New_Ih': 1})
        out = self.RES.y[index_dict[out_label].to_list(), :]
        return out

    def turn_count_mode(self, res):
        if res.ndim == 1:
            res = res.reshape(1, -1)
        res = list(map(lambda x: np.sum(x, axis=1).tolist(), np.hsplit(res, self.T)))
        return np.array(res).T

    def get_result(self, label=None, period='day', given_times=None, plot=False):
        # equation solving
        self.ode_solver()

        # According to out_label output solution result
        out = self.redirect_index(label)
        if label == ['New_Ih']:
            out = out * 0.0056  # faih = 0.0056
        if period == 'year':
            out = self.turn_count_mode(out)
            if given_times is not None:
                out = np.squeeze(out[:, given_times])
        if plot:
            self.plot(out, label)
        return out

    def plot(self, out, out_label):
        plt.figure(figsize=(8, 4), dpi=200)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        for i in range(len(out_label)):
            plt.plot(out[i], label=out_label[i])
            for j, num in enumerate(out[i]):
                print("-{:s}, time:{:d}(day/year), num:{:d}".format(out_label[i], j + 1, int(num)))
        plt.title('SimulationModel', fontsize=10)
        plt.legend(fontsize=8)
        plt.xlabel('Time(day/year)', fontsize=9)
        plt.ylabel('Number', fontsize=9)
        plt.show()
