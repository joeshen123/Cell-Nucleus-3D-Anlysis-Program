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

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)



# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory):
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
          df_subset["Normalize volume"] = df_subset['volume'] / df_subset['volume'].iloc[0]
          df_subset["number"] = obs
          df_list.append(df_subset)
          obs += 1

   df_len = len(df_list)

   df_final = pd.concat(df_list)

   return df_final, df_len



root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1]


df_final_one,df_final_one_len = curve_df_combine(Name1)
print(df_final_one)

root.directory = filedialog.askdirectory()
Name2 = root.directory
label_2 = Name2.split("/")[-1]
df_final_two,df_final_two_len = curve_df_combine(Name2)
print(df_final_two)

sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12,"lines.linewidth": 5,'lines.markersize': 7})
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax = sns.lineplot(x='frame', y='normalized intensity ratio', data = df_final_one, label='%s (n = %d)' %(label_1, df_final_one_len))


ax = sns.lineplot(x='frame', y='normalized intensity ratio', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))


ax.tick_params(axis = 'y', labelsize = 'x-large') 
ax.tick_params(axis = 'x',  labelsize = 'x-large')
ax.set_xlabel('Frame', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('Binding Intensity', fontweight = 'bold', fontsize = 20)
#ax.set_ylim([0,24.5])
#ax.get_legend().remove()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)

#Save figure as pdf for later processing in AI illustrator
#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
#ass = AnySizeSaver(fig=fig, dpi=600, filename=fig_save_name)
#plt.show()

#plt.savefig(fig_save_name, transparent=True)

'''
File_save_names = filedialog.asksaveasfilename(parent=root,title="Please select a file name for saving:",filetypes=[('Image Files', '.hdf5')])
decay_Name='{File_Name}.hdf5'.format(File_Name = File_save_names)

with h5py.File(decay_Name, "w") as f:
      f.create_dataset('control', data = df_means_control, compression = 'gzip')
      f.create_dataset('swell', data = df_means_swell, compression = 'gzip')
      f.create_dataset('calcium', data = df_means_calcium, compression = 'gzip')
'''
