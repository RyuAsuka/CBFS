"""
Combine the split files to a new file.
"""
import pandas as pd
import os

for n in range(1, 11):
	print("--------------- File: " + str(n) + "----------------")
	read_path = r'E:\python\splitDataSet\data' + str(n)
	save_path = r'E:\python\splitDataSet\result'
	save_name = "dataset_random_split" + str(n) + '.csv'
	csv_list = os.listdir(read_path)
	print(csv_list)

	df = pd.read_csv(read_path + '\\' + csv_list[0])

	df = pd.DataFrame(df)
	df.to_csv(save_name, encoding="utf_8", index=False, mode='a')

	for i in range(1, len(csv_list)):
		new_file = pd.read_csv(read_path + '\\' + csv_list[i])
		new_file = pd.DataFrame(new_file)
		new_file.to_csv(
			save_name,
			encoding="utf_8",
			index=False,
			header=True,
			mode='a'
		)
