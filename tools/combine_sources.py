import pandas as pd
import matplotlib.pyplot as plt

sys_info = 'system_info.csv'
time_source = 'res_2.csv'
result_source = 'compiled_res_param_only.csv'

system_df = pd.read_csv(sys_info)
system_df['Total Files'] = system_df['c++ Files'] + system_df['Headers'] + system_df['Python Files']
system_df['Total LOC'] = system_df['c++ LOC'] + system_df['Header LOC'] + system_df['Python LOC']

time_df = pd.read_csv(time_source)
time_df = time_df[['name', 'clean_time', 'additional_time']]

res = pd.read_csv(result_source)

combined = time_df.merge(res, how='inner', on='name')
combined = combined.merge(system_df, how='inner', on='name')


examined = system_df[system_df['Total LOC'] > 1]
droped = system_df[system_df['Total LOC'] == 0]

print combined.threshold_comparisions.describe()

combined.to_csv('combined_new.csv')
