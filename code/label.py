import pandas as pd
import shutil
df=pd.read_csv('eeglabels.csv', sep=',',header=None)
print(df.values)
for i,j,k in df.values:
	if k =="ABNORMAL":
		shutil.copy2('KT88-2400/000'+str(i)+'.edf', 'abnormal') 
	else:
		shutil.copy2('KT88-2400/000'+str(i)+'.edf', 'normal')