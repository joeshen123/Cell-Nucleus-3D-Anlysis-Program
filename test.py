import tkinter as tk
from tkinter import filedialog 
from tkinter import messagebox


root = tk.Tk()
root.withdraw()
filez = filedialog.askopenfilenames(parent=root, title='Choose a file')

print(filez)