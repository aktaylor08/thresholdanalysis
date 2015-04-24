import pandas as pd
import matplotlib.pyplot as plt



mapping = {
    'asctec base': ["asctec base", ],
    'water sampler': ["water sampler", ],
    'crop surveying': ["crop surveying", ],
    'care o bot': ["cob external", "cob common", "cob_robots", "Cob Command Tools", "cob_driver", "shcunk modular",
                   "careobot contro", "careobot perception", "careobot manipulation", "careobot evnironment perception",
                   "careobot navigation perception", ],
    "navigation stack": ["navigation stack", ],
    "Baxter Robot": ["Baxter Robot", ],
    "Controls": ["Realtime tools", "Control Toolbox", "Ros Control", "Ros Controllers", ],
    "BWI from Texas": ["BWI from Texas", ],
    "AR Tracking": ["AR Tracking", ],
    "AprilTags Tracking": ["AprilTags Tracking", ],
    "Airbotix ros package": ["Airbotix ros package", ],
    "Asctec Mav Pacakge": ["Asctec Mav Pacakge", ],
    "UOS Tools": ["UOS Tools", ],
    "Calvin Ros Stack": ["Calvin Ros Stack", ],
    "CrazyFlie Ros Stack": ["CrazyFlie Ros Stack", ],
    "Ros Concert": ["Ros Concert", ],
    "ROS Create Driver": ["ROS Create Driver", ],
    "ROS Darwin": ["ROS Darwin", ],
    "ROS Descartes": ["ROS Descartes", ],
    "Func Maninulators": ["Func Maninulators", ],
    "ros filter library": ["ros filter library", ],
    "graft": ["graft", ],
    "Grizzly robot": ["Grizzly robot", ],
    'Hector': ["Hector Slam", "Hector Arm", "Hector navigation", "Hector diagnostics", "Hector turtlebot", ],
    "ICART MINI": ["ICART MINI", ],
    "Jaco Robot Arm": ["Jaco Robot Arm", ],
    "Calibration": ["Calibration", ],
    'jsk Applications': ["jsk smart apps", "jsk_control", "jsk_travis", ],
    "kobuki": ["kobuki", ],
    "kobuki soft": ["kobuki_soft", ],
    "Mav Ros": ["Mav Ros", ],
    "Maxwell": ["Maxwell", ],
    "Motoman": ["Motoman", ],
    "NAO": ["NAO Ros", "NAO Robot repo", "NAO Extras", "NAO interaction", "Naopi bridge", "nao camera", "nao virtual",
            "nao viz", "nao sensors", ],
    "Nav2 platform": ["Nav2 platform", ],
    "neo robot": ["neo robot", ],
    "next stage": ['next_stage'],
    "novatel_spann": ["novatel_spann"],
    "p2 os robot": ["p2 os robot"],
    "robot rescue": ["robot rescue"],
    "People tracking ros": ["People tracking ros"],
    "Pepper robot for stuff": ["Pepper robot for stuff", ],
    'Rail Robots': ["Rail Pick and place library", "rail_segmentation", "rail ceiling", ],
    "robitician ric": ["robitician ric"],
    'segbot': ["segbot", "segbot apps", ],
    'turtlebot': ["turtlebot apps", "turtlebot interactions", "turtlebot", "turtlebot create", "turtlebot arm", ],
    "ros universial robot": ["ros universial robot ", ],
    "ocs library": ["ocs library", ],
    'sr_robots': ["sr utils", "sr manipulation", "sr demo", ],
    "app manager": ["app_manager", ],
    "arm nav": ["arm_nav", ],
    'pr2': ["pr futre", "pre2 apps", "pr2_colibraiton", "/pr2_common", "pr2_common_actions", "pr2_delivery", "pr2_doors",
           "pr2_kinematics", "pr2_navigation", "pr2_pbd", "pr2_precise_trajectory", "pr2_self_test", "pr2_surrogate",
           "rqt_pr2_dashboard", ]
}


code_cols = ['name', 'c++ Files', 'c++ LOC', 'Headers', 'Header LOC', 'Python Files', 'Python LOC', 'Total Files', 'Total LOC']
thresh_cols = ['name', 'clean_time', 'additional_time', 'threshold_comparisions', 'cpp', 'python', 'unique', 'param', 'unique_param', 'files']


def main():
    df = pd.read_csv('/Users/ataylor/Research/thresholdanalysis/tools/combined.csv', index_col=0)
    df = df[df["Total LOC"] > 1]
    code_df = df[code_cols]
    code_df = fix_up(code_df)
    code_df = add_data(code_df)
    code_df.to_csv('/Users/ataylor/Research/thresh_writeups/data/code_table.csv', sep='&',
                   line_terminator="\\\\\n", header=False,float_format='%.0f' )
    code_df.to_csv('/Users/ataylor/Research/thresh_writeups/data/code_table_real.csv')


    tdf = df[thresh_cols]
    tdf.loc[:, 'Time Factor'] = (tdf.loc[:, 'clean_time'] + tdf.loc[:, 'additional_time']) / tdf.loc[:, 'clean_time']
    tdf = fix_up(tdf)
    out_df = pare_down(tdf)
    only_thresh = out_df[out_df['threshold_comparisions'] > 0]
    print "Only contain thresholds: ", len(only_thresh)
    out_df = add_both(out_df, only_thresh)

    out_df.to_csv('/Users/ataylor/Research/thresh_writeups/data/repo_results.csv', sep='&',
                   line_terminator="\\\\\n", header=False, float_format='%.1f')
    out_df.to_csv('/Users/ataylor/Research/thresh_writeups/data/repo_results_real.csv')

    combined_df = pd.DataFrame()
    for i in mapping:
        temp = df.loc[df['name'].isin(mapping[i])]
        sums = temp.sum()
        sums['name'] = i
        sums['number_of_repos'] = len(mapping[i])
        combined_df = combined_df.append(sums, ignore_index=True)

    combined_df.loc[:, 'Time Factor'] = (combined_df.loc[:, 'clean_time'] + combined_df.loc[:, 'additional_time']) / combined_df.loc[:, 'clean_time']
    combined_df = fix_up(combined_df)
    out_df = pare_down(combined_df)
    only_thresh = out_df[out_df['threshold_comparisions'] > 0]
    print "Only contain thresholds combined: ", len(only_thresh)
    out_df = add_both(out_df, only_thresh)

    out_df.to_csv('/Users/ataylor/Research/thresh_writeups/data/system_results.csv', sep='&',
                  line_terminator="\\\\\n", header=False, float_format='%.1f')
    out_df.to_csv('/Users/ataylor/Research/thresh_writeups/data/system_results_real.csv')

    only_thresh = out_df[out_df.threshold_comparisions > 0]
    no_thresh = out_df[out_df.threshold_comparisions == 0]
    print no_thresh.number_of_repos
    print no_thresh['Total LOC'].describe()
    print out_df.number_of_repos.describe()
    print out_df.number_of_repos.value_counts()
    print 'Hey'
    print len(combined_df[combined_df.number_of_repos == 1]) / float(len(combined_df))
    print len(only_thresh)
    print len(only_thresh) / float(len(combined_df))
    print only_thresh.threshold_comparisions.describe()


def pare_down(tdf):
    out_df = pd.DataFrame(index=tdf.index)
    out_df['time'] = tdf['clean_time']
    out_df['time_factor'] = tdf['Time Factor']
    cpp = tdf['cpp']
    python = tdf['param'] - tdf['cpp']
    total = cpp + python
    out_df['threshold_comparisions'] = total
    out_df['cpp'] = cpp
    out_df['python'] = python
    out_df['unique'] = tdf['unique_param']
    out_df['files'] = tdf['files']
    return out_df


def add_only_rows(to_add_to, original):
    only = original[original['threshold_comparisions'] > 0]
    desc = only.describe()
    sums = only.sum()
    medians = only.median()
    for i in to_add_to.columns:
        to_add_to.loc['\\textbf{threshold only mean}', i] = desc.loc['mean', i]
        to_add_to.loc['\\textbf{threshold only median}', i] = medians[i]
        to_add_to.loc['\\textbf{threshold only std}', i] = desc.loc['std', i]
        to_add_to.loc['\\textbf{threshold only min}', i] = desc.loc['min', i]
        to_add_to.loc['\\textbf{threshold only max}', i] = desc.loc['max', i]
        to_add_to.loc['\\textbf{threshold only sum}', i] = sums[i]
    return to_add_to


def fix_up(df):
    df.loc[:, 'name'] = df['name'].apply(lambda x: x.lower())
    df.loc[:, 'name'] = df['name'].apply(lambda x: x.replace('_', ' '))
    df = df.set_index('name')
    df = df.sort_index()
    return df


def add_both(df, df_only):
    desc = df.describe()
    sums = df.sum()
    medians = df.median()
    for i in desc.columns:
        df.loc['\\textbf{mean}', i] = desc.loc['mean', i]
        df.loc['\\textbf{median}', i] = medians[i]
        df.loc['\\textbf{std}', i] = desc.loc['std', i]
        df.loc['\\textbf{min}', i] = desc.loc['min', i]
        df.loc['\\textbf{max}', i] = desc.loc['max', i]
        df.loc['\\textbf{sum}', i] = sums[i]
    desc = df_only.describe()
    sums = df_only.sum()
    medians = df_only.median()
    for i in desc.columns:
        df.loc['\\textbf{only threshold mean}', i] = desc.loc['mean', i]
        df.loc['\\textbf{only threshold median}', i] = medians[i]
        df.loc['\\textbf{only threshold std}', i] = desc.loc['std', i]
        df.loc['\\textbf{only threshold min}', i] = desc.loc['min', i]
        df.loc['\\textbf{only threshold max}', i] = desc.loc['max', i]
        df.loc['\\textbf{only threshold sum}', i] = sums[i]
    return df

def add_data(df):
    desc = df.describe()
    sums = df.sum()
    medians = df.median()
    for i in desc.columns:
        df.loc['\\textbf{mean}', i] = desc.loc['mean', i]
        df.loc['\\textbf{median}', i] = medians[i]
        df.loc['\\textbf{std}', i] = desc.loc['std', i]
        df.loc['\\textbf{min}', i] = desc.loc['min', i]
        df.loc['\\textbf{max}', i] = desc.loc['max', i]
        df.loc['\\textbf{sum}', i] = sums[i]
    return df

if __name__ == '__main__':
    main()


