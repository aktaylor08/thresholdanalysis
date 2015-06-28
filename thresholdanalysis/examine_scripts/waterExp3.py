
# coding: utf-8

# In[67]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from thresholdanalysis.analysis import summerization_tools, experiment_examine
from thresholdanalysis.examine_scripts import experiment_examine

get_ipython().magic(u'matplotlib inline')


# In[68]:

info_dir = '../test_data/water_sampler/static_info/'
experiment = '../test_data/water_sampler/dynamic/all_info/experiment3_2015-04-30-15-57-33.csv'
info = summerization_tools.get_info(info_dir)
thresh_df = summerization_tools.get_df(experiment, info)
df = pd.read_csv(experiment, parse_dates=True, index_col=0)
action, no_actions = experiment_examine.get_marks(experiment)
no_adv_score = pd.read_csv('../test_data/water_sampler/dynamic/all_info/no_advance/experiment3_2015-04-30-15-57-33_no_advance_scores.csv', parse_dates=True, index_col=0)
adv_score = pd.read_csv('../test_data/water_sampler/dynamic/all_info/advance/experiment3_2015-04-30-15-57-33_advance_scores.csv', parse_dates=True, index_col=0)


# In[70]:

act = []
for k,x in action.iteritems():
    if isinstance(k,int):
        for y in x:
            act.append(y)
noact = []
for k,x in no_actions.iteritems():
    if isinstance(k,int):
        for y in x:
            noact.append(y)
print act
print noact


# In[73]:

for i in act:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res
for i in noact:
    idx = no_adv_score.index.asof(i)
    res = no_adv_score.loc[idx, :]
    res.sort()
    print i
    print res
    print '\n\n'


# In[83]:

print info['d4a8c703-b0b9-49d4-aae0-d531c47e2d8a']
print info['0c95250a-abf7-48bc-bff8-4b7193913af1']


# In[75]:

fig, axes = plt.subplots(2, 1, sharex=True)
ax = axes[0]
df.a_subject_pose__translation_z.dropna().plot(figsize=(11,8.5), fontsize=20,linewidth=4, ax=ax )
lims = ax.get_ylim()
pairs = [(x,(lims[0] + lims[1]) /2) for x in act]
nopairs = [(x,(lims[0] + lims[1]) /2) for x in noact]
#print zip(*pairs)
#ax.scatter(zip(*pairs)[0], zip(*pairs)[1], c='r',marker='x', s=200)
ax.scatter(zip(*nopairs)[0], zip(*nopairs)[1], c='g', marker='x', s=200)
ax.set_ylabel('Height (m)', fontsize=20)
ax.set_xlabel('Time', fontsize=20)

ax = axes[1]
df.a_subject_pose__translation_x.dropna().plot(figsize=(11,8.5), fontsize=20,linewidth=4, ax=ax)
#df.a_subject_pose__translation_y.dropna().plot(figsize=(11,8.5), fontsize=20,linewidth=4, ax=ax )

lims = ax.get_ylim()
pairs = [(x,(lims[0] + lims[1]) /2) for x in act]
nopairs = [(x,(lims[0] + lims[1]) /2) for x in noact]
#print zip(*pairs)
#ax.scatter(zip(*pairs)[0], zip(*pairs)[1], c='r',marker='x', s=200)
ax.scatter(zip(*nopairs)[0], zip(*nopairs)[1], c='g', marker='x', s=200)
ax.set_ylabel('X Location (m)', fontsize=20)
ax.set_xlabel('Time', fontsize=20)



fig.savefig('../../thesis/myFigures/waterMission3.png')


# In[76]:

vals = df[df.a_subject_pose__translation_x < -2]


# In[77]:

at_loc = vals.index[0]


# In[78]:

print noact[0] - at_loc
print noact[1] - at_loc
print noact[2] - at_loc


# In[79]:

thresh_df.key.unique()


# In[80]:

key = '/home/ataylor/water_sampler_experiment/src/h2o_sampling/script_utils/FlyToObject_mod.py:41:0'


# In[81]:

changed_only = thresh_df[thresh_df['key'] == key]


# In[44]:

changed_only


# In[45]:

ft = changed_only[changed_only['flop']]


# In[48]:

print ft
if len(ft) > 0:
    idx = ft[0]


# In[49]:




# In[51]:

for i in act:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]


# In[53]:

for i in noact:
    idx = no_adv_score.index.asof(i)
    res = no_adv_score.loc[idx, :]
    res.sort()
    for idx, val in res.iteritems():
        if val < 9999.9:
            print res[idx], idx
            print info[idx]['source']
    


# In[62]:

no_adv_score[no_adv_score[key] != 9999.9][key].plot()


# In[63]:

for i in noact:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]
    print info[res.index[0]]['source']


# In[65]:

ax = no_adv_score.min(axis=1).plot(figsize=(11 ,8.5), linewidth=3)
ax.scatter(no_adv_score[key].index, no_adv_score[key], marker='x', c='r')
ax.set_ylim(0,2)


# In[82]:

no_max = no_adv_score.min(axis=1) < 9999
no_oob = no_adv_score[no_max]


# In[66]:

a = no_oob.idxmin(axis=1).value_counts() / len(no_oob)


# In[84]:

type(a)


# In[85]:

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
        
        
        
        


# In[89]:

print times
vals = []
for i, dom_times in times.iteritems():
    print info[i]['source'], np.mean(dom_times)
    vals.append(np.mean(dom_times))
    


# In[88]:

np.mean(vals)


# In[88]:




# In[59]:

no_adv_score[key]


# In[84]:

k1 = 'd4a8c703-b0b9-49d4-aae0-d531c47e2d8a'
k2 = '0c95250a-abf7-48bc-bff8-4b7193913af1'


# In[86]:

no_adv_score[no_adv_score[k1] < 9999][k1].plot()


# In[88]:

no_adv_score[no_adv_score[k2] < 9999][k2].plot()


# In[89]:

k3 = '/home/ataylor/water_sampler_experiment/src/h2o_sampling/h2o_safety/h2o_safety.py:119:0'


# In[90]:

no_adv_score[no_adv_score[k3] < 9999][k3].plot()


# In[ ]:



