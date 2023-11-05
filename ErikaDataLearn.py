import pandas as pd
from pandas import DataFrame, Series
import os
import re

init = r'C:\Users\nikhi\Dropbox\JaldiKaro\DataScience\EPI\Src\ErikaProj'
finalpath = init.replace('\\', '/')
wkdir = finalpath[:-14]
os.chdir(wkdir)
files_to_load = os.listdir(finalpath)

print(files_to_load)

file_suffix = re.compile('DY*[1-9][0-9]', flags=False)
file_suffix_matches = [file_suffix.search(f) for f in files_to_load]
file_suffixes = [fsm[0] for fsm in file_suffix_matches]
print(file_suffixes)
dfs = [pd.read_csv(finalpath + '/' + f, low_memory=False) for f in files_to_load]

df = pd.concat(dfs, keys=file_suffixes)
print(df.head())

#df.to_csv(wkdir + '/ErikaData_fromNS.csv')
