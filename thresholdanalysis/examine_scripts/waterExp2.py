
# coding: utf-8

# In[183]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from thresholdanalysis.analysis import experiment_examine, summerization_tools

get_ipython().magic(u'matplotlib inline')


# In[255]:

info_dir = '../test_data/water_sampler/static_info/'
experiment = '../test_data/water_sampler/dynamic/all_info/experiment2_2015-04-30-15-54-45.csv'
info = summerization_tools.get_info(info_dir)
thresh_df = summerization_tools.get_df(experiment, info)
df = pd.read_csv(experiment, parse_dates=True, index_col=0)
action, no_actions = experiment_examine.get_marks(experiment)
no_adv_score = pd.read_csv('../test_data/water_sampler/dynamic/all_info/no_advance/experiment2_2015-04-30-15-54-45_no_advance_scores.csv', parse_dates=True, index_col=0)
adv_score = pd.read_csv('../test_data/water_sampler/dynamic/all_info/advance/experiment2_2015-04-30-15-54-45_advance_scores.csv', parse_dates=True, index_col=0)


# In[256]:

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


# In[257]:

for i in act:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res
for i in noact:
    idx = no_adv_score.index.asof(i)
    res = no_adv_score.loc[idx, :]
    res.sort()
    print res
    


# In[187]:

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
print nopairs
#ax.scatter(zip(*pairs)[0], zip(*pairs)[1], c='r',marker='x', s=200)
ax.scatter(zip(*nopairs)[0], zip(*nopairs)[1], c='g', marker='x', s=200)
ax.set_ylabel('X Location (m)', fontsize=20)
ax.set_xlabel('Time', fontsize=20)



fig.savefig('../../thesis/myFigures/waterMission2.png')


# In[188]:

vals = df[df.a_subject_pose__translation_x < -2]


# In[189]:

at_loc = vals.index[0]


# In[190]:

noact[0] - at_loc


# In[191]:

thresh_df.key.unique()


# In[192]:

key = '/home/ataylor/water_sampler_experiment/src/h2o_sampling/script_utils/FlyToObject.py:41:0'


# In[193]:

changed_only = thresh_df[thresh_df['key'] == key]


# In[194]:

changed_only


# In[258]:

thresh_df[thresh_df.flop]


# In[259]:

idx = ft.index[0]
info['0c95250a-abf7-48bc-bff8-4b7193913af1']


# In[197]:

for i in noact:
   print i - idx
for i in act:
    print i -idx


# In[211]:

for i in noact:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]


# In[212]:

for i in noact:

    idx = no_adv_score.index.asof(i)
    res = no_adv_score.loc[idx, :]
    res.sort()
    for idx, val in res.iteritems():
        if val < 9999.9:
            print res[idx], idx
            print info[idx]['source']
    


# In[200]:

no_adv_score[adv_score[key] != 9999.9][key].plot()


# In[244]:

for i in noact:
    idx = adv_score.index.asof(i)
    res = adv_score.loc[idx, :]
    res.sort()
    print res[0], res.index[0]
    print info[res.index[0]]['source']


# In[248]:

fig, ax = plt.subplots()
fig.set_size_inches(11,8.5)
ax.scatter(no_adv_score[key].index, no_adv_score[key], marker='x', c='r')
xlims = ax.get_xlim()
no_adv_score.min(axis=1).plot(figsize=(11 ,8.5), linewidth=3, ax=ax)
ax.set_ylim(0,2)


# In[254]:

fig, ax = plt.subplots()
fig.set_size_inches(11,8)
for i in no_adv_score.columns:
    data = no_adv_score[no_adv_score[i] != 9999.9]
    if len(data) > 1:
        data[i].plot()


# In[203]:

no_max = no_adv_score.min(axis=1) < 9999
no_oob = no_adv_score[no_max]


# In[204]:

a = no_oob.idxmin(axis=1).value_counts() / len(no_oob)


# In[205]:

type(a)


# In[206]:

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
        
        
        
        


# In[207]:

print times
vals = []
for i, dom_times in times.iteritems():
    print info[i]['source'], np.mean(dom_times)
    vals.append(np.mean(dom_times))
    


# In[208]:

np.mean(vals)


# In[209]:

for i in thresh_df


# In[ ]:

a = thresh_df[thresh_df.key == 'd4a8c703-b0b9-49d4-aae0-d531c47e2d8a']


# In[ ]:

a.cmp.plot()
print info['d4a8c703-b0b9-49d4-aae0-d531c47e2d8a']['file']
print info['d4a8c703-b0b9-49d4-aae0-d531c47e2d8a']['lineno']
print info['d4a8c703-b0b9-49d4-aae0-d531c47e2d8a']['source']




# In[ ]:

a.thresh.plot()


# In[ ]:

df.a_distance_to_task_pose__dist.dropna().plot()


# In[ ]:

df.a_subject_pose__translation_x.dropna().plot()


# In[ ]:

for i,d in thresh_df.groupby('key'):
    ax =  d.thresh.plot(linewidth=4)
ax.set_ylim(-1,2)


# In[182]:

for i in info:
    print i


# In[214]:

df.a_distance_to_task_pose__dist.dropna().plot()


# In[225]:

thresh_df[thresh_df['key'] == key].plot()


# In[ ]:



