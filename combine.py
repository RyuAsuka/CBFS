#将每个文件的提取的20%的数据文件合并成一个新的文件
import pandas as pd
import os
import numpy as np
import glob

for n in range(1,11):
	print("---------------this is "+ str(n) +"----------------")
	read_path='E:\python\splitDataSet\data'+str(n)
	save_path='E:\python\splitDataSet\result'
	save_name="dataset_random_split"+ str(n) +'.csv'
	csv_list=os.listdir(read_path)
	print(csv_list)

	df=pd.read_csv(read_path+'\\'+csv_list[0])

	df=pd.DataFrame(df)
	df.to_csv(save_name,encoding="utf_8",index=False,mode = 'a')

	x = 1
	print(x)
	for i in range(1,len(csv_list)):
		new_file=pd.read_csv(read_path+'\\'+csv_list[i])
		new_file=pd.DataFrame(new_file)
		new_file.to_csv(save_name,encoding="utf_8",index=False,header=0,mode = 'a')
		x=x+1
		print(x)
