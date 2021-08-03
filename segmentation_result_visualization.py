import tkinter as tk
from tkinter import filedialog 
from tkinter import messagebox
import pandas as pd
import h5py
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib.pyplot import cm
import napari

root = tk.Tk()
root.withdraw()


filez = filedialog.askopenfilenames(parent=root, title='Choose a file')

print(filez)
filename = filez[0]
seg_name = filez[2]

f = h5py.File(filename, 'r')

lmb1_image = f['488 Channel'][:]
cpla2_intensity_image = f['561 Channel'][:]

f1 = h5py.File(seg_name, 'r')
contour_img = f1['Contour Image'][:]
seg_img = f1['Segmented Object'][:]
threshold = f1['thresholded image'][:]
seed_map = f1['Seed Map'][:]

df = pd.read_csv(filez[1])
track_df = df[['particle','frame','z','y','x']]
track_df.index.names = ['Data']
track_df.sort_values(by=['particle','frame'], inplace=True)
track_points = track_df.values

def plotting(df, marker = None):
    
    static_canvas = FigureCanvas(Figure(figsize=(3, 1)))
    
    axes = static_canvas.figure.subplots(3, sharex=True)

    #list all remaining tracks
    all_labels = df.particle.unique()

    color = cm.tab20b(np.linspace(0,1,len(all_labels)))
    
    axes[0].set_ylabel('Mid Section', fontsize = 16,fontweight = 'bold')
    axes[1].set_ylabel('Volume', fontsize = 16,fontweight = 'bold')
    axes[2].set_xlabel('Frame', fontsize = 16, fontweight = 'bold')
    axes[2].set_ylabel('Protein Binding', fontsize = 16,fontweight = 'bold')

    num = 0

    for n in all_labels:
       df_subset = df[df.particle == n]
       frame_list = df_subset['frame'].tolist()
       area_list = df_subset['mid section area'].tolist()
       volume_list = df_subset['volume'].tolist()
       binding_list = df_subset['normalized intensity ratio'].tolist()
       axes[0].plot(frame_list, area_list, color = color[num], label = str(n), marker = marker)
       axes[1].plot(frame_list, volume_list, color = color[num], label = str(n), marker = marker)
       axes[2].plot(frame_list, binding_list, color = color[num], label = str(n), marker = marker)
       num += 1
    
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
      
    return static_canvas

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(lmb1_image, name = 'Raw Image', colormap='green',scale = [1,5,1,1], blending='additive')
    viewer.add_image(cpla2_intensity_image, name = 'Intensity Image', colormap='red',scale = [1,5,1,1], blending='additive')
    viewer.add_image(threshold,name='Thresholded',scale = [1,5,1,1], blending='additive')
    peaks = np.nonzero(seed_map)
    viewer.add_points(np.array(peaks).T, name='peaks', size=5, face_color='red',scale = [1,5,1,1],blending='additive')
    viewer.add_image(contour_img,name='contours crop', scale = [1,5,1,1], blending='additive')
    viewer.add_labels(seg_img, name='Segmented Object Crop', scale = [1,5,1,1], blending='additive')
    viewer.add_tracks(track_points, name='tracks',scale = [1,5,1,1], blending='additive')
      
    #Plotting
    canvasobj = plotting (df,marker='o')
    viewer.window.add_dock_widget(canvasobj, area='right', name='Analysis Plot')  
