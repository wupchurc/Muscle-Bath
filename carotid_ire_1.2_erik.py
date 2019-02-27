
# coding: utf-8

# In[1]:


import pandas as pd
import scipy as sp
import matplotlib as plt
import os 
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# Erik cool code

# In[3]:


import numpy as np
from pathlib import Path
from collections import namedtuple
import pdb

PATH = Path('C:/Users/david/Documents/muscle_baths/')


Study = namedtuple('Study', ['baseline_df', 'recovery_df', 'date', 'bath_key'])
Result = namedtuple('Result', ['voltage', 'min_delta', 'max_delta', 'mean_delta'])

# Should be a dictionary
# (date, num) = > namedtuple 

def analyze_data(path):
    studies = get_studies(path)
    results = [process_study(i) for i in studies]
    results = sum(results, [])
    # TODO: Add stats code or whatever
    return results

def get_treatment_dict(csv):
    bkey = pd.read_csv(PATH / 'bkey.csv')
    dates = bkey['Date'].values
    bkeys = [int(i) for i in bkey['key'].values]
    fnums = [int(i[5:]) for i in bkey['Force']]
    voltages = [int(i) if i !='na' else -1 for i in bkey['Voltage']]
    return {i : voltages[ind] for ind, i in enumerate(zip(dates, bkeys, fnums))}


def process_study(study):
    TIME = -30*60
    #Return means we care about
    bl = study.baseline_df.drop(columns=['Comments'])
    recov = study.recovery_df.drop(columns=['Comments'])
    min_delta = recov.iloc[TIME:].min() - bl.iloc[TIME:].min()
    max_delta = recov.iloc[TIME:].max() - bl.iloc[TIME:].max()
    mean_delta = recov.iloc[TIME:].mean() - bl.iloc[TIME:].mean()
    return [Result(bkey_dict[(study.date, study.bath_key, int(index_name[5]))],
                min_delta[index_name],
                max_delta[index_name],
                mean_delta[index_name]) for index_name in min_delta.index]

    

#####################################################
    
def get_studies(path):
    '''Return studies, treatment_keys'''
    # All the studies we care about
    # TODO: Intercept dependent data here.
        # Treatments... etc
    studies = [get_study_dfs(i) for i in path.iterdir() if i.is_dir()]
    studies = sum(studies, [])
    return studies

def get_study_dfs(path):
    '''Produce a dataframe for a study given by path.'''
    tsvs = [i for i in path.iterdir() if '.tsv' in i.name]
    dfs = [pd.read_csv(i, sep='\t').drop(columns=['Timestamp', 'Stimulus']) for i in tsvs]
    dfs = [i.set_index('Experiment Time') for i in dfs]
    treatment_nums = [get_treat_num(i) for i in tsvs]
    
    return [Study(df[df['Comments'] == 'baseline'],
               df[df['Comments'] == 'recovery'],
               treatment_nums[ind][0], treatment_nums[ind][1]) for ind, df in enumerate(dfs)]

def get_treat_num(study_path):
    '''Grab key for treatments. (Date, num)'''
    return study_path.stem[0:10], int(study_path.stem[-1])

bkey_dict = get_treatment_dict('bkey.csv')


# In[ ]:


paths = list(PATH.iterdir())
tsvs = [i for i in paths[0].iterdir() if '.tsv' in i.name]
dfs = pd.read_csv(tsvs[0], sep='\t')
#dfs.drop(columns=['Timestamp', 'Stimulus'])


# In[5]:




    
bkey_dict[('2019_02_07', 1, 2)]


# In[6]:


results = analyze_data(PATH)
results = [i for i in results if i.voltage != -1]
groups = sorted(set([i.voltage for i in results]))

ls = []
for group in groups:
    ls.append([i for i in results if i.voltage == group])
    
out = pd.DataFrame([(i.voltage, i.min_delta, i.max_delta, i.mean_delta) for i in sum(ls, [])])
out.columns = ['voltage', 'min_delta', 'max_delta', 'mean_delta']
# out.groupby('voltage').plot(x='voltage', kind='bar')
out.to_csv(PATH / 'out2.csv', index=False)


# In[7]:


results


# In[ ]:


studies[0].recovery_df.set_index('Experiment Time')[foo].plot()


# In[9]:


out


# In[ ]:


studies[4].baseline_df.plot()


# In[ ]:


studies[0].baseline_df.iloc[-60*30:].mean()


# In[ ]:


studies[0].recovery_df.iloc[-60*30:].mean()

