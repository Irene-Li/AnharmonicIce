import numpy as np
import pandas as pd
from readers import EfieldReader

factor = 2.1947463e5

file1 = 'Efield/input_files/freqs_grid.dat'
file2 = 'Efield/output/static/IceIh.efield'

freq1 = np.array(pd.read_table(file1, header = None, delim_whitespace=True).iloc[:, 0], dtype = float)
freq1 *= factor
freqs = np.zeros((freq1.size, 2))
freqs[:, 0] = freq1

Reader = EfieldReader()
Reader.read(file2)
freqs[:, 1] = Reader.osciFreq

np.savetxt('freq_comparison.dat', freqs)
