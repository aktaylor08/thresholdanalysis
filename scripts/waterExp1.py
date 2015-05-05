import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sumerize_vals
import experiment_examine

FIG_DIR = '/Users/ataylor/Research/thesis/myFigures/'


def fix_and_plot(score, collapse):
    # get rankings
    ranking = sumerize_vals.calculate_ranking(score, collapse)
    # determine where a ranking changed by shifting the value to the left and right
    rank_change = (ranking == ranking.shift(1)).apply(lambda row: not row.all(), axis=1)
    for idx in ranking[rank_change].index:
        new_idx = idx - datetime.timedelta(0,.01)
        ranking.loc[new_idx, :] = np.nan
    fig, ax = plt.subplots()
    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            ranking[col].plot(linewidth=3, ax=ax, c='b')
    ranking[mod_key].plot(linewidth=3, ax=ax, c='r')
    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax


info_dir = '../test_data/water_sampler/static_info/'
experiment = '../test_data/water_sampler/dynamic/all_info/experiment1_2015-04-30-15-52-51.csv'
info = sumerize_vals.get_info(info_dir)
thresh_df = sumerize_vals.get_df(experiment, info)
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

mod_key = '/home/ataylor/water_sampler_experiment/src/h2o_sampling/h2o_safety/h2o_safety.py:119:0'

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
    ax.axvline(x, c='g' )
ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320)
plt.savefig(FIG_DIR + 'water1Adv.png')

fig, ax = fix_and_plot(adv_score, True)
for x in act:
    ax.axvline(x, c='g' )
ax.scatter(zip(*advpoints)[0], zip(*advpoints)[1], marker='x', c='g', s=320)
plt.savefig(FIG_DIR + 'water1AdvCollapsed.png')
sys.exit(-1)





for i in act:
    print i
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
for i in noact:
    print i
    idx = no_adv_score.index.asof(i)
    res = no_adv_score.loc[idx, :]
    res.sort()


ax = df.a_subject_pose__translation_z.dropna().plot(figsize=(11,8.5), fontsize=20,linewidth=4 )
lims = ax.get_ylim()
pairs = [(x,(lims[0] + lims[1]) /2) for x in act]
nopairs = [(x,(lims[0] + lims[1]) /2) for x in noact]
print zip(*pairs)
ax.scatter(zip(*pairs)[0], zip(*pairs)[1], c='r',marker='x', s=300)
ax.scatter(zip(*nopairs)[0], zip(*nopairs)[1], c='g', marker='x', s=300)
ax.set_ylabel('Height (m)', fontsize=20)
ax.set_xlabel('Time', fontsize=20)
fig = ax.get_figure()
fig.savefig('../../thesis/myFigures/waterMission1.png')
plt.show()

for i in range(3):
    print i
    print '\t', action[i]
    print '\t', no_actions[i]


# In[15]:

ok = df[df.a_subject_pose__translation_x < -2.0]
first = ok.index[0]
n = ok[ok.a_subject_pose__translation_z < 1.8]
n.index[0] - first


# In[98]:

key = '/home/ataylor/water_sampler_experiment/src/h2o_sampling/h2o_safety/h2o_safety.py:119:0'


# In[100]:

changed_only = thresh_df[thresh_df['key'] == key]


# In[104]:

ft = changed_only[changed_only['flop']]


# In[107]:

idx = ft.index[0]


# In[111]:

for i in noact:
   print i - idx
for i in act:
    print i -idx


# In[113]:

for i in act:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]


# In[117]:

for i in noact:
    idx = no_adv_score.index.asof(i)
    res = no_adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]
    print info[res.index[0]]['source']


# In[134]:

adv_score[adv_score[key] != 9999.9][key].plot()


# In[122]:

for i in noact:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]
    print info[res.index[0]]['source']


# In[143]:

ax = adv_score.min(axis=1).plot(figsize=(11 ,8.5), linewidth=3)
ax.scatter(adv_score[key].index, adv_score[key], marker='x', c='r')
ax.set_ylim(0,20)


# In[148]:

no_max = adv_score.min(axis=1) < 9999
no_oob = adv_score[no_max]


# In[153]:

a = no_oob.idxmin(axis=1).value_counts() / len(no_oob)
times = {}
last = None
st = None
for t, v in no_oob.idxmin(axis=1).iteritems():
    if last is None:
        last = v
        st = t
    elif last != v:
        telapsed = t - st
        dt = (telapsed).total_seconds()
        st = t
        last = v
        if last in times:
            times[last].append(dt)
        else:
            times[last] = [dt]
        
print times
vals = []
for i, dom_times in times.iteritems():
    print info[i]['source'], np.mean(dom_times)
    vals.append(np.mean(dom_times))
    
print 'mean time on top: ', np.mean(vals)

