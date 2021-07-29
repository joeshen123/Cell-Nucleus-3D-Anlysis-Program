
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from pandas import HDFStore
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
import scipy.stats as stats
import seaborn as sns
import matplotlib
import os, glob

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory):
   os.chdir(directory)

   #df_list = []

   df_filenames = glob.glob('*.csv' )

   max_int_list = []
   max_area_list = []
   max_vol_list = []
   for n in range(len(df_filenames)):
      df_name = df_filenames[n]

      df = pd.read_csv(df_name)
      df = df.drop(df.index[-1])
      particle_list=df['particle'].unique().tolist()
      
      obs = 1
      for p in particle_list:
          df_subset = df.loc[df.particle == float(p)]
          df_subset["Normalized mid section area"] = df_subset['mid section area'] / df_subset['mid section area'].iloc[0]
          df_subset["Normalize volume"] = df_subset['volume'] / df_subset['volume'].iloc[0]
          df_subset["number"] = obs

          max_area = df_subset["Normalized mid section area"].max()
          max_vol = df_subset["Normalize volume"].max()
          max_int = df_subset["normalized intensity ratio"].max()

          max_int_list.append(max_int)
          max_vol_list.append(max_vol)
          max_area_list.append(max_area)
    
          obs += 1



   return max_int_list, max_area_list, max_vol_list



root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1]

max_int_list1, max_area_list1, max_vol_list1 = curve_df_combine(Name1)


root.directory = filedialog.askdirectory()
Name2 = root.directory
label_2 = Name2.split("/")[-1]
max_int_list2, max_area_list2, max_vol_list2 = curve_df_combine(Name2)



Control_Data = pd.DataFrame(max_int_list1, columns =['Intensity'])
TSA_Data = pd.DataFrame(max_int_list2, columns = ['Intensity'])

#Combine Iso and Hypo data into 1 dataframe
Control_Data['condition'] = 'Control'
TSA_Data['condition'] = 'Treated'
#df_fin2['condition'] = d2.split('/')[-1]
Control_drug_list = [Control_Data,TSA_Data]
Control_drug_Final = pd.concat(Control_drug_list)

#Hypo_Iso_Final=Hypo_Iso_Final.rename(columns = {'Normalized GFP intensity':'Fluorescence Life Time (ns)'})

print(Control_drug_Final)



# Perform ttest to determine significance
import scipy.stats
control = max_int_list1
treated = max_int_list2

print(scipy.stats.ttest_ind(control, treated, equal_var=False)) 

# Drawthe point plot with individual data points

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

sns.set(style="ticks")
sns.set_context("paper", font_scale=2, rc={"font.size":16,"axes.labelsize":16})


ax = sns.pointplot(x="condition", y="Intensity", data=Control_drug_Final, ci=95, color='k',capsize = 0.1, errwidth=4)
ax = sns.swarmplot(x="condition", y="Intensity",data=Control_drug_Final, hue = 'condition',size = 8,palette=['b','r'])


#labels = ['Hypotonic \n(n = %d)' % len(Hypo),'Isotonic \n(n = %d)' % len(Iso)]

#fig.set(xticklabels = labels)

ax.tick_params(labelsize= 16)
plt.ylabel('Intensity', fontsize=14, fontweight = 'bold')
ax.get_legend().remove()
plt.rcParams['axes.labelweight'] = 'bold'
plt.show()

# Save as PDF
#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
#plt.savefig(fig_save_name, transparent=True)
