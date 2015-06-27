import datetime

import matplotlib.dates as mdates
import matplotlib.cm as colormaps
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from thresholdanalysis.analysis import experiment_examine, summerization_tools

FIG_DIR = '/Users/ataylor/Research/thesis/myFigures/'
mod_key = '/home/ataylor/water_sampler_experiment/src/h2o_sampling/h2o_safety/h2o_safety.py:119:0'


def fix_and_plot(score, collapse):
    # get rankings
    ranking = summerization_tools.calculate_ranking(score, collapse)
    # determine where a ranking changed by shifting the value to the left and right
    rank_change = (ranking == ranking.shift(1)).apply(lambda row: not row.all(), axis=1)
    for idx in ranking[rank_change].index:
        new_idx = idx - datetime.timedelta(0,.01)
        ranking.loc[new_idx, :] = np.nan
    fig, ax = plt.subplots()
    count = 1
    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            ranking[col].plot(linewidth=2, ax=ax, c='b',)
            count += 1
    ranking[mod_key].plot(linewidth=2, ax=ax, c='r')
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax


def fix_and_plot_num(score, collapse, n):
    # get rankings
    ranking = summerization_tools.calculate_ranking(score, collapse)
    # determine where a ranking changed by shifting the value to the left and right
    val = pd.DatetimeIndex([ranking.index[0]])
    # take care of rankings...
    rank_change = ranking[(ranking == ranking.shift(-1)).apply(lambda row: not row.all(), axis=1)].index
    rank_change = val.append(rank_change)
    rank_change.append(pd.DatetimeIndex([ranking.index[-1]]))
    # get the location of all zeros and replace them with NaN
    zero_row = (ranking == 0).any(axis=1)
    zero_series = zero_row.apply(lambda x: 0 if x else np.NaN)

    zero_mod = ranking[ranking[mod_key] == 0][mod_key]

    ranking[ranking == 0] = np.NaN

    fig, ax = plt.subplots()
    total = 0
    for s, f in zip(rank_change[0:-1], rank_change[1:]):
        split_df = ranking[s:f]
        split_df = split_df[1:]
        total += len(split_df)

        count = 1
        marks = [len(split_df) / 2]
        for col in ranking.columns:
            if col == mod_key:
                pass
            else:
                split_df[col].plot(ax=ax, color='b', marker='${:d}$'.format(count), markevery=marks, linewidth=2,zorder=2)
                count += 1
        ser = split_df[mod_key]
        ser = ser.apply(lambda x: 0 if x == np.NaN else x)
        ser.plot(linewidth=2, ax=ax, color='r', markevery=marks, marker='${:d}$'.format(count, zorder=2))
    zero_series.plot(linewidth=2, ax=ax, c='black')
    zero_mod.plot(linewidth=2, ax=ax, c='r', zorder=20)
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax


def fix_and_plot_connected(score, collapse):
    # get rankings
    ranking = summerization_tools.calculate_ranking(score, collapse)
    # get the location of all zeros and replace them with NaN
    zero_row = (ranking == 0).any(axis=1)
    zero_series = zero_row.apply(lambda x: 0 if x else np.NaN)
    zero_mod = ranking[ranking[mod_key] == 0][mod_key]
    ranking[ranking == 0] = np.NaN
    fig, ax = plt.subplots()
    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            ranking[col].plot(ax=ax, color='b',linewidth=2)
    zero_series.plot(linewidth=2, ax=ax, c='b',)
    zero_mod.plot(linewidth=2, ax=ax, c='r', zorder=20)
    ranking[mod_key].plot(linewidth=2, ax=ax, c='r')
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax

def fix_and_plot_connectedcolor(score, collapse):
    # get rankings
    cmap = colormaps.get_cmap('winter')
    ranking = summerization_tools.calculate_ranking(score, collapse)
    levels = {}
    for num, val in enumerate(ranking.columns):
        levels[val] = float(num) / len(ranking.columns)
    levels = {k: cmap(v) for k, v in levels.iteritems()}

    # get the location of all zeros and replace them with NaN
    zero_row = (ranking == 0).any(axis=1)
    zero_series = zero_row.apply(lambda x: 0 if x else np.NaN)
    zero_mod = ranking[ranking[mod_key] == 0][mod_key]
    ranking[ranking == 0] = np.NaN
    fig, ax = plt.subplots()
    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            ranking[col].plot(ax=ax, color=levels[col], linewidth=2)
    zero_series.plot(linewidth=2, ax=ax, c='black',)
    zero_mod.plot(linewidth=2, ax=ax, c='r', zorder=20)
    ranking[mod_key].plot(linewidth=2, ax=ax, c='r')
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax

def fix_and_plot_arrows(score, collapse):
    # get rankings
    ranking = summerization_tools.calculate_ranking(score, collapse)
    rank_change = ranking != ranking.shift(1)
    arrows = {}

    last_idx = None
    for idx, row in rank_change.iterrows():
        if row.any():
            for c in row[row].index:
                if last_idx is not None:
                    if c in arrows:
                        lid = mdates.date2num(last_idx)
                        diff = mdates.date2num(idx) - lid
                        arrows[c].append((lid, ranking.loc[last_idx,c], diff, ranking.loc[idx,c] - ranking.loc[last_idx, c]))
                    else:
                        lid = mdates.date2num(last_idx)
                        diff = mdates.date2num(idx)
                        arrows[c] = [(lid, ranking.loc[last_idx,c], diff, ranking.loc[idx,c])]
        last_idx = idx

    ranking[rank_change] = np.NaN
    # get the location of all zeros and replace them with NaN
    zero_row = (ranking == 0).any(axis=1)
    zero_series = zero_row.apply(lambda x: 0 if x else np.NaN)
    zero_mod = ranking[ranking[mod_key] == 0][mod_key]
    ranking[ranking == 0] = np.NaN
    fig, ax = plt.subplots()
    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            ranking[col].plot(ax=ax, color='b',linewidth=2)
    zero_series.plot(linewidth=2, ax=ax, c='black',zorder=100)
    zero_mod.plot(linewidth=2, ax=ax, c='r', zorder=100)
    ranking[mod_key].plot(linewidth=2, ax=ax, c='r')

    for i in arrows:
        for val in arrows[i]:
            arr_width = 1e-6
            ax.arrow(val[0], val[1], val[2], val[3],
                     width=arr_width,
                     head_width=7 * arr_width,
                     head_length = 10000 * arr_width,
                     length_includes_head=True,
                     fc='b',
                     ec='b',
                     zorder=30
                     )
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax

def fix_and_plot_arrows_color(score, collapse):
    # get rankings
    ranking = summerization_tools.calculate_ranking(score, collapse)
    rank_change = ranking != ranking.shift(1)

    ranking = summerization_tools.calculate_ranking(score, collapse)

    arrows = {}
    last_idx = None
    for idx, row in rank_change.iterrows():
        if row.any():
            for c in row[row].index:
                if last_idx is not None:
                    if c in arrows:
                        lid = mdates.date2num(last_idx)
                        diff = mdates.date2num(idx) - lid
                        arrows[c].append((lid, ranking.loc[last_idx,c], diff, ranking.loc[idx,c] - ranking.loc[last_idx, c]))
                    else:
                        lid = mdates.date2num(last_idx)
                        diff = mdates.date2num(idx)
                        arrows[c] = [(lid, ranking.loc[last_idx,c], diff, ranking.loc[idx,c])]
        last_idx = idx
    levels = {}

    cmap = colormaps.get_cmap('winter')
    for num, val in enumerate(ranking.columns):
        levels[val] = float(num) / len(ranking.columns)
    levels = {k: cmap(v) for k, v in levels.iteritems()}
    levels[mod_key] = 'red'

    ranking[rank_change] = np.NaN
    # get the location of all zeros and replace them with NaN
    zero_row = (ranking == 0).any(axis=1)
    zero_series = zero_row.apply(lambda x: 0 if x else np.NaN)
    zero_mod = ranking[ranking[mod_key] == 0][mod_key]
    ranking[ranking == 0] = np.NaN
    fig, ax = plt.subplots()
    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            ranking[col].plot(ax=ax, color=levels[col],linewidth=2)
    zero_series.plot(linewidth=2, ax=ax, c='black',zorder=100)
    zero_mod.plot(linewidth=2, ax=ax, c='r', zorder=100)
    ranking[mod_key].plot(linewidth=2, ax=ax, c='r')

    for i in arrows:
        color = levels[i]
        for val in arrows[i]:
            zorder = 30
            if i == mod_key:
                zorder = 35
            arr_width = 1e-6
            ax.arrow(val[0], val[1], val[2], val[3],
                     width=arr_width,
                     head_width=10 * arr_width,
                     head_length = 15000 * arr_width,
                     length_includes_head=True,
                     shape='right',
                     fc=color,
                     ec=color,
                     zorder=zorder,
                     )
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax

if __name__ == '__main__':
    info_dir = '../test_data/water_sampler/static_info/'
    experiment = '../test_data/water_sampler/dynamic/all_info/experiment1_2015-04-30-15-52-51.csv'
    info = summerization_tools.get_info(info_dir)
    thresh_df = summerization_tools.get_df(experiment, info)
    df = pd.read_csv(experiment, parse_dates=True, index_col=0)
    action, no_actions = experiment_examine.get_marks(experiment)
    no_adv_score = pd.read_csv('../test_data/water_sampler/dynamic/all_info/no_advance/experiment1_2015-04-30-15-52-51_no_advance_scores.csv', parse_dates=True, index_col=0)
    adv_score = pd.read_csv('../test_data/water_sampler/dynamic/all_info/advance/experiment1_2015-04-30-15-52-51_advance_scores.csv', parse_dates=True, index_col=0)

    act = []
    for k, x in action.iteritems():
        if isinstance(k, int):
            for y in x:
                act.append(y)
    noact = []
    for k,x in no_actions.iteritems():
        if isinstance(k,int):
            for y in x:
                noact.append(y)


    vals = [.4, .6]
    nopoints = zip(noact, vals * (len(noact) / 2))
    advpoints = zip(act, vals * (len(noact) / 2))

    fig, ax = fix_and_plot(no_adv_score, False)
    for x in noact:
        ax.axvline(x, c='g' )
    ax.scatter(zip(*nopoints)[0], zip(*nopoints)[1], marker='x', c='g', s=320)
    plt.savefig(FIG_DIR + 'water1NoAdv.png')

    fig, ax = fix_and_plot(no_adv_score, True)
    for x in noact:
        ax.axvline(x, c='g' )
    ax.scatter(zip(*nopoints)[0], zip(*nopoints)[1], marker='x', c='g', s=320)
    plt.savefig(FIG_DIR + 'water1NoAdvCollapsed.png')

    fig, ax = fix_and_plot(adv_score, False)
    for x in act:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1Adv.png')

    fig, ax = fix_and_plot(adv_score, True)
    for x in act:
        ax.axvline(x, c='g',zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1AdvCollapsed.png')

    fig, ax = fix_and_plot_num(adv_score, True, 20)
    for x in act:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1AdvNum.png'.format(20))

    fig, ax = fix_and_plot_connected(adv_score, True)
    for x in act:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1AdvConnected.png'.format(20))

    fig, ax = fix_and_plot_connectedcolor(adv_score, True)
    for x in act:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1AdvConnectedColor.png'.format(20))

    fig, ax = fix_and_plot_arrows(adv_score, True)
    for x in act:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1AdvArrow.png'.format(20))

    fig, ax = fix_and_plot_arrows_color(adv_score, True)
    for x in act:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1AdvArrowColor.png'.format(20))

    fig, ax = fix_and_plot_arrows_color(no_adv_score, True)
    for x in noact:
        ax.axvline(x, c='g', zorder=0)
    ax.scatter(zip(*nopoints)[0], zip(*nopoints)[1], marker='x', c='g', s=320, zorder=0)
    plt.savefig(FIG_DIR + 'water1NoAdvArrowColor.png'.format(20))

