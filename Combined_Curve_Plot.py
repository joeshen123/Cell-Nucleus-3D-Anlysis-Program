import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import glob,os
import warnings
import numpy as np
import h5py
import matplotlib
import scipy
from tkinter import simpledialog

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)



# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory, gap):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*.csv' )

   
   for n in range(len(df_filenames)):
      df_name = df_filenames[n]

      df = pd.read_csv(df_name)

      particle_list=df['particle'].unique().tolist()
      
      obs = 1
      for p in particle_list:
          df_subset = df.loc[df.particle == int(p)]
          
          df_subset["Time"] = df_subset['frame'] * gap
          df_subset["Normalize volume"] = df_subset['volume'] / df_subset['volume'].iloc[0]

          df_subset["number"] = obs
          df_list.append(df_subset)
          obs += 1

   df_len = len(df_list)

   df_final = pd.concat(df_list)

   return df_final, df_len

root = tk.Tk()
root.withdraw()

#Ask how many time (secs) in the interval
intervals = simpledialog.askstring("Input", "How long between intervals in min? ", parent=root)
intervals = float(intervals)

root.directory = filedialog.askdirectory()

#Save directory name for graph
Save_Name = root.directory.rsplit('/',1)[0]


print(root.directory)
Name1 = root.directory
label_1 = Name1.split("/")[-1]


df_final_one,df_final_one_len = curve_df_combine(Name1,gap=intervals)

root.directory = filedialog.askdirectory()
Name2 = root.directory
label_2 = Name2.split("/")[-1]
df_final_two,df_final_two_len = curve_df_combine(Name2,gap=intervals)

#Save treatment name for graph
Treatment_Name = root.directory.rsplit('/',1)[1]

#Plot for Normalized volume ratio change accross time
sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12,"lines.linewidth": 5,'lines.markersize': 7})
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax = sns.lineplot(x='Time', y='Normalize volume',ci=68, data = df_final_one, label='%s (n = %d)' %(label_1, df_final_one_len))


ax = sns.lineplot(x='Time', y='Normalize volume',ci=68, data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))


ax.tick_params(axis = 'y', labelsize = 'x-large') 
ax.tick_params(axis = 'x',  labelsize = 'x-large')
ax.set_xlabel('Time (min)', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('Volume change ratio', fontweight = 'bold', fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

#Save Plot
plot_save_name_vol='{File_Name}/{Condition} volume plot.tiff'.format(File_Name = Save_Name, Condition = Treatment_Name)
plt.savefig(plot_save_name_vol, dpi=200)

#Plot for Normalized intensity ratio change accross time
sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12,"lines.linewidth": 5,'lines.markersize': 7})
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax = sns.lineplot(x='Time', y='normalized intensity ratio',ci=68, data = df_final_one, label='%s (n = %d)' %(label_1, df_final_one_len))


ax = sns.lineplot(x='Time', y='normalized intensity ratio',ci=68, data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))


ax.tick_params(axis = 'y', labelsize = 'x-large') 
ax.tick_params(axis = 'x',  labelsize = 'x-large')
ax.set_xlabel('Time (min)', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('Binding Intensity Ratio', fontweight = 'bold', fontsize = 20)
#ax.set_ylim([0,24.5])
#ax.get_legend().remove()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

#Save plot
plot_save_name_int ='{File_Name}/{Condition} cPla2 binding intensity plot.tiff'.format(File_Name = Save_Name, Condition = Treatment_Name)
plt.savefig(plot_save_name_int, dpi=200)
