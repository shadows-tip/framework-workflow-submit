import numpy as np
from ModelPackage import Model
from joblib import Parallel, delayed
from Utils import lhs_sampling, get_data_column, save


def timeline_generator(intervene_measure, intervene_values):
    Ih = []
    for i, intervene_value in enumerate(intervene_values):
        model = Model(T=120, Nd=40000)
        model.set_param(intervene_meas=intervene_measure, intervene_time=100, intervene_param=intervene_value)
        res = model.get_result(label=['Ih', 'Id'])
        Ih.extend(res[:, 97 * 365:105 * 365])
    Ih = np.array(Ih).T
    out = np.concatenate((np.full((len(Ih), 1), intervene_measure), np.arange(0, len(Ih)).reshape(len(Ih), 1), Ih), axis=1)
    return out


def parallel_module(t, intervene_time, item):
    scene_name, scene, intervene_meas = item

    # Sample size for training and testing sets
    dataset = {'train_set': 8000, 'test_set': 2000}
    '''
    The time points of attention are before intervention, the first year of intervention, 
    the third year of intervention, and the fifth year of intervention
    '''
    given_times = [(intervene_time - 2 + i) for i in [0, 1, 3, 5]]
    for dataset_key, dataset_value in dataset.items():
        print("{:s}, intervene_measure{:d}, {:s} Solving start....".format(scene_name, intervene_meas, dataset_key))
        # Perform sampling
        lhs_param = lhs_sampling(n=dataset_value, param_range=[[0, 1]] * (len(str(intervene_meas)) * 2))
        res = np.zeros((len(lhs_param), len(given_times)))
        # Sample by sample solving
        for i in range(len(lhs_param)):
            if i % 100 == 0 and i != 0:
                print("Combination[{:d}]-[{:d}] Solve completed".format(i - 100, i))
            model = Model(t, scene)
            model.set_param(intervene_meas, intervene_time, lhs_param[i])
            res[i] = model.get_result(label=['New_Ih'], period='year', given_times=given_times)
        out = np.concatenate((lhs_param, res), axis=1)
        save(out, './DataFolder/DatasetFile/' + scene_name + '/meas' + str(
            intervene_meas) + '_' + scene_name + '_' + dataset_key + '.csv',
             columns=get_data_column(intervene_meas) + ['Ih(before)', 'Ih(1th)', 'Ih(3th)', 'Ih(5th)'])


if __name__ == '__main__':
    # intervene_list = [[0.7, 0.85, 0.98], [0.5, 0.7, 0.9], [0.1, 0.3, 0.5]]
    intervene_list = [[[0.7, 0.5], [0.85, 0.7], [0.98, 0.9]],
                      [[0.7, 0.1], [0.85, 0.3], [0.98, 0.5]],
                      [[0.5, 0.1], [0.7, 0.3], [0.9, 0.5]],
                      [[0.7, 0.5, 0.1], [0.85, 0.7, 0.3], [0.98, 0.9, 0.5]]]
    intervene_measures = [12, 13, 23, 123]
    OUT = []
    for m, measure in enumerate(intervene_measures):
        timeline_Out = timeline_generator(measure, intervene_list[m])
        OUT.extend(timeline_Out)
    save(OUT, './DataFolder/ResultFile/Combination_timeline_data.csv',
         ['interv_measure', 'day', 'min_P', 'min_D', 'med_P', 'med_D', 'max_P', 'max_D'])
    print("timeline_data is saved!!")

    Nh = 1400000
    Scenes = {'Scene1': Nh / 70, 'Scene2': Nh / 35, 'Scene3': Nh / 20, 'Scene4': Nh / 10, 'Scene5': Nh / 5,
              'Scene6': Nh / 3}
    T = 120  # Epidemic time of the disease, unit: year
    Intervene_time = 100
    Measures = [12, 13, 23, 123]  # intervention measures
    Parallel_tuple = [(Scene_key, Scene, Measure) for Scene_key, Scene in Scenes.items() for Measure in Measures]
    Task = [delayed(parallel_module)(T, Intervene_time, parallel_item) for parallel_item in Parallel_tuple]
    Worker = Parallel(n_jobs=30, backend='multiprocessing')
    Worker(Task)
