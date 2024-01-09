import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from plotnine import *
import warnings
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# GP result plot
def plot_scatter(y_pre, y_true):
    for p in range(y_pre.shape[1]):
        plt.rcParams.update({'font.size': 11})
        fig, ax = plt.subplots()
        axis_line = np.linspace(*ax.get_xlim(), 2)
        ax.plot(axis_line, axis_line, transform=ax.transAxes, linestyle='--', linewidth=2, color='black',
                label="1:1 Line")
        plt.xlabel("True value", fontsize=12)
        plt.ylabel("Predicted value", fontsize=12)
        plt.scatter(y_true[:, p], y_pre[:, p], c='orange', s=100, alpha=0.3, marker='o', edgecolors='red')
        plt.show()


def plot_scatter_boxplot(true_data, pre_data, cv_score, time_score, meas):
    titles = ["One year follow-up", "Three year follow-up", "Five year follow-up"]
    position = [[0.26, .22, .055, .27], [0.587, .22, .055, .27], [0.914, .22, .055, .27]]
    config = {
        "font.family": "serif",
        "font.serif": ["Arial"],
        "font.size": 14,
        "axes.unicode_minus": False  # 处理负号，即-号
    }
    matplotlib.rcParams.update(config)
    fig = plt.figure(figsize=(12, 4.5))
    for i in range(3):
        X, Y = true_data[:, i], pre_data[:, i]
        ax1 = fig.add_subplot(1, 3, i + 1)

        axis_line = np.linspace(*ax1.get_xlim(), 2)
        ax1.plot(axis_line, axis_line, transform=ax1.transAxes, linestyle='--', linewidth=1, color='black')
        ax1.scatter(X, Y, c="w", s=200, edgecolors='k')
        plt.xlim(xmax=26)
        plt.ylim(ymax=26)
        # plt.xticks([0, 5, 10, 15, 20, 25])
        # plt.xticks(np.linspace(int(ax1.get_xlim()[0]) if ax1.get_xlim()[0] > 0 else 0, ax1.get_xlim()[1], 5))
        # plt.yticks(np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 5))
        if i == 0:
            ax1.text(ax1.get_xlim()[0] + 1.5, 23, 'R2=' + str(round(time_score[i], 4)), fontsize=16)
        else:
            ax1.text(ax1.get_xlim()[0] + 1.5, 23, 'R2=' + str(round(time_score[i], 4)), fontsize=16)

        plt.xlabel("True value", fontdict={'family': "Arial", 'size': 16})
        plt.ylabel("Predicted value", fontdict={'family': "Arial", 'size': 16})
        plt.title(titles[i])

        sub_axes = plt.axes(position[i])
        sub_axes.boxplot(cv_score[0:5], widths=0.6, showcaps=False,
                         medianprops={'color': 'k'}, )  # whiskerprops={'linestyle': ""}
        sub_axes.scatter(x=[1], y=[time_score[i]], marker='D', c='k', edgecolors='k', s=50)
        # sub_axes.set(ylabel="r2")  # title="boxplot"
        print(cv_score[0:5])
        print(np.arange(round(np.min(cv_score[0:5]), 4), 0.9999, 0.0001))
        # plt.yticks(np.arange(round(np.min(cv_score[0:5]), 4), round(np.max(cv_score[0:5]), 4), 0.0001))
        sub_axes.xaxis.set_visible(False)
        sub_axes.yaxis.set_visible(False)
        plt.ylabel("r2")

        plt.setp(sub_axes)
    fig.suptitle('GP fit result(Measure' + meas + ' Scene2)', fontsize=20)
    fig.tight_layout()
    plt.savefig('./DataFolder/FigureFile/GP-fit-result(Meas' + meas + ' , Scene2)1.png', dpi=500)
    # plt.show()


def plot_gp_result():
    measures = [12, 13, 23, 123]
    for meas in measures:
        m = str(meas)
        true_data = pd.read_csv("DataFolder/DatasetFile/Scene2/meas" + m + "_Scene2_test_set.csv").values[:,
                    len(m) * 2:]
        pred_data = pd.read_csv("DataFolder/ResultFile/Scene2/m" + m + "_GP_predict.csv").values[:, [1, 2, 3]]
        cv_score = pd.read_csv("DataFolder/ResultFile/Scene2/m" + m + "_GP_cv_res.csv").values[:, 1]
        time_score = []
        true_data = true_data[:, [1, 2, 3]]
        for i in range(3):
            r2 = r2_score(true_data[:, i].reshape(-1), pred_data[:, i].reshape(-1))
            time_score.append(r2)

        plot_scatter_boxplot(true_data, pred_data, cv_score, time_score, m)


def plot_timeline(plot_data):
    plot_data['interv_measure'].replace({1: 'insect repellent collar', 2: 'indoor residual spray', 3: 'kill sick dogs',
                                         12: 'measure 1+2', 13: 'measure 1+3', 23: 'measure 2+3',
                                         123: 'measure 1+2+3'},
                                        inplace=True)
    human_df = plot_data[['interv_measure', 'day', 'min_P', 'med_P', 'max_P']]
    dog_df = plot_data[['interv_measure', 'day', 'min_D', 'med_D', 'max_D']]

    human_plot = (
            ggplot(human_df, aes(x='day/30', y='med_P', ymin='min_P', ymax='max_P', fill='interv_measure',
                                 linetype='interv_measure'))
            + geom_line()
            + geom_ribbon(alpha=0.5)
            + theme_bw()
            # + theme(panel_grid=element_blank())
            + theme_matplotlib()
            + labs(x="Months", y="No. of infected people", fill="Interventions", linetype='Interventions'))
    dog_plot = (
            ggplot(dog_df, aes(x='day/30', y='med_D', ymin='min_D', ymax='max_D', fill='interv_measure',
                               linetype='interv_measure'))
            + geom_line()
            + geom_ribbon(alpha=0.5)
            + theme_bw()
            # + theme(panel_grid=element_blank())
            + theme_matplotlib()
            + labs(x="Months", y="No. of infected dogs", fill="Interventions", linetype='Interventions')
            + scale_color_discrete(breaks=['measure 1+2', 'measure 1+3', 'measure 2+3', 'measure 1+2+3'])
    )

    ggsave(plot=human_plot, filename='./DataFolder/FigureFile/combination_human_timeline.png', dpi=1000)
    ggsave(plot=dog_plot, filename='./DataFolder/FigureFile/combination_dog_timeline.png', dpi=1000)
    print(human_plot)
    print(dog_plot)


# optimization result plot
def plot_optim_figure(plot_data):
    low_colors = ("#d0d1e6", "#c7e9b4", "#9ecae1", "#D3E2EF", "#f1faeb", "#AB82FF", "#B6C5B2")
    high_colors = ("#3792BF", "#6CC3B9", "#0D539C", "#8C6BB1", "#addfb7", "#5D478B", "#7D8B72")
    intervene_titles = ["Sandfly repellent dog collars", "IRS", "Culling of infected dogs"]

    combination_titles = ["Sandfly repellent dog collars(Efficacy=0.95) + IRS",
                          "Sandfly repellent dog collars(Efficacy=0.95) + Culling of infected dogs",
                          "IRS(Efficacy=0.85) + Culling of infected dogs",
                          "Sandfly repellent dog collars(Efficacy=0.95) + Culling of infected dogs(Coverage=0.5, Efficacy=0.95) + IRS"]
    titles = intervene_titles + combination_titles
    years_titles = [", One year's intervention", ", Three year's intervention", ", Five year's intervention"]

    for i, measure in enumerate(pd.unique(plot_data.interv_measure)):
        for t, year in enumerate(pd.unique(plot_data.year)):

            def process_subtitle(x, *arg):
                item, meas = arg
                if str(x) == 'nan':
                    out = str(x)
                else:
                    if len(str(meas)) == 1:
                        out = titles[i] + '(Efficacy=' + str(x) + ')'
                    else:
                        temp = x.split('/')
                        out = intervene_titles[int(str(measure)[1]) - 1] + "(Coverage=" + temp[1] + ", Efficacy=" + \
                              temp[2] + ")"
                return out

            df = plot_data[(plot_data.interv_measure == measure) & (data.year == year)]
            df.scene = df.scene.apply(lambda x: str(x) if str(x) == 'nan' else 'Scene' + str(int(x)))  # Scene
            # df.efficacy = df.efficacy.apply(lambda x: str(x) if str(x) == 'nan' else 'Efficacy' + str(x))
            df.efficacy = df.efficacy.apply(process_subtitle, args=(i, measure))
            df['object_value'] = pd.Categorical(-df['object_value'])
            df['object_value'] = df['object_value'].apply(lambda x: int(-x))
            # 单一干预措施：intervene_titles[i] + years_titles[t]；组合干预措施：combination_titles[i] + years_titles[t]
            args = (df, low_colors[i], high_colors[i], measure, year, titles[i] + years_titles[t])
            plot_tile(args)


def plot_tile(args):
    df, low_color, high_color, measure, year, title = args
    opt_param_ranges = (0, 1)
    if df['min_coverage'].isnull().all():
        df['min_coverage'] = -1
    base_plot = (
            ggplot(df, aes(x='scene', y='object_value', fill='min_coverage'))
            + geom_tile()
            + theme_bw(base_size=11.5)
            + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
            + scale_fill_gradient2(low=low_color, mid=low_color, high=high_color, na_value="white",
                                   limits=opt_param_ranges, midpoint=-1)

            + facet_wrap('~efficacy', scales='free_x', nrow=1)
            # + theme(subplots_adjust={'wspace': 0.02})  # 设置各分图的间距
            + labs(x="Transmission level", y="Target Health Goal", fill="Minimum Coverage", title=title)
            # + labs(x="传播水平", y="健康目标", fill="最小覆盖率", title=title)
            + guides(fill=guide_colourbar(title_position='left'))
            + theme(legend_text=element_text(va='bottom', ha='left'),
                    legend_title=element_text(angle=90, va='bottom', ha='right'))
            + theme(text=element_text(size=8),
                    plot_title=element_text(size=10),
                    legend_title=element_text(size=10))
            + theme(panel_background=element_rect(fill='white'))
            + theme(strip_background=element_rect(fill="white"))
    )

    ggsave(plot=base_plot, filename='./DataFolder/FigureFile2/optim_m_' + str(measure) + '_t_' + str(year) + '.png',
           dpi=1024, width=22, height=4)
    # print(base_plot)


# plot the optimization result figure
data = pd.read_csv("./DataFolder/ResultFile/Optim.csv")
plot_optim_figure(data)

# plot the GP training result figure
# plot_gp_result()

# plot a timeline of disease prevalence. Single_timeline_data.csv/Combination_timeline_data.csv
# time_line_data = pd.read_csv("./DataFolder/ResultFile/Combination_timeline_data.csv")
# plot_timeline(Combination_time_line_data)
