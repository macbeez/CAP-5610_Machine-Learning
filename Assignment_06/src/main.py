import pandas as pd
import numpy as np
import os.path
from surprise import SVD, Reader, Dataset, KNNWithMeans, accuracy, KNNBasic
from surprise.model_selection import cross_validate, train_test_split
from surprise import Reader
from surprise import Dataset
import matplotlib.pyplot as plt

data_path = "archive/ratings_small.csv"

def plot_graph(rmse, mae, title):
	rmse = np.array(rmse)
	mae = np.array(mae)
	rmse_mean = rmse.mean()
	mae_mean = mae.mean()
	print("RMSE: ", rmse)
	print("RMSE mean: ", rmse_mean)
	print("MAE: ", mae)
	print("MAE mean: ",mae_mean)

	ylim_max = max(np.max(rmse), np.max(mae))

	fig, ax = plt.subplots(1,2, figsize = (7,7))
	x_values = ("Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5")
	y_pos = np.arange(len(x_values))
	fig.suptitle(title)

	ax[0].bar(y_pos, rmse, align = "center", alpha = 0.5)
	ax[0].set_xticks(y_pos)
	ax[0].set_ylim(0, 1.2 * ylim_max)
	# ax[0].set_xtickslabels(x_values)
	ax[0].set_title("RMSE graph")
	ax[0].set_xlabel("Fold Values")
	ax[0].set_ylabel("Error Vaules")
	ax[0].axhline(rmse_mean, color = 'k', linestyle = 'dashed', linewidth = 1)

	ax[1].bar(y_pos, mae, align = "center", alpha = 0.5)
	ax[1].set_xticks(y_pos)
	ax[1].set_ylim(0, 1.2 * ylim_max)
	ax[1].set_title("MAE graph")
	ax[1].set_xlabel("Fold Values")
	ax[1].set_ylabel("Error Vaules")
	ax[1].axhline(mae_mean, color = 'k', linestyle = 'dashed', linewidth = 1)

	plt.tight_layout()
	# plt.savefig("graph.png")
	plt.show()

def k_vs_error_graph(k_list, rmse_list, mae_list, graph_title):
	plt.figure(figsize = (12,8))
	plt.title(graph_title, loc = 'center')
	plt.plot(k_list, rmse_list, label = "RMSE values", color = "green", marker = "x")
	plt.plot(k_list, mae_list, label = "MAE values", color = "red", marker = "o")
	plt.xlabel("K-Neighbors")
	plt.ylabel("Error values")
	plt.legend()
	plt.grid(ls = "dotted")
	plt.show()

### PART A ###

reader = Reader()
ratings = pd.read_csv(data_path)
print(ratings.head())

# ### PART C ###

file_path = os.path.abspath(data_path)
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

# Average MAE and RMSE of the Probabilistic Matrix Factorization (PMF) under 5 fold cross validation
algorithm1 = SVD(biased = False) # For PMF model set biased to bool in SVD model
output1 = cross_validate(algorithm1, data, ['MAE', 'RMSE'], cv = 5)
print("\nRMSE and MAE for five-fold PMF:")
print(output1)

# Average MAE and RMSE of the User based Collaborative Filtering under 5 fold cross validation
algorithm2 = KNNWithMeans(k = 50, sim_options = {'name': 'cosine', 'user_based': True})
output2 = cross_validate(algorithm2, data, ['MAE', 'RMSE'], cv = 5)
print("\nRMSE and MAE for five-fold User Based Collaborative Filtering: ")
print(output2)

# Average MAE and RMSE of the Item based Collaborative Filtering under 5 fold cross validation
algorithm3 = KNNWithMeans(k = 50, sim_options = {'name': 'pearson_baseline', 'user_based': False})
output3 = cross_validate(algorithm3, data, ['MAE', 'RMSE'], cv = 5)
print("\nRMSE and MAE for five-fold Item Based Collaborative Filtering: ")
print(output3)

# PART D ###
pmf_mae_mean = output1['test_mae'].mean()
pmf_rmse_mean = output1['test_rmse'].mean()
print("\n\nAverage MAE of Probabilistic Matrix Factorization (PMF): ", pmf_mae_mean)
print("Average RMSE of Probabilistic Matrix Factorization (PMF): ", pmf_rmse_mean)

user_based_mae_mean = output2['test_mae'].mean()
user_based_rmse_mean = output2['test_rmse'].mean()
print("\nAverage MAE of User based Collaborative Filtering: ", user_based_mae_mean)
print("Average RMSE of User based Collaborative Filtering: ", user_based_rmse_mean)

item_based_mae_mean = output3['test_mae'].mean()
item_based_rmse_mean = output3['test_rmse'].mean()
print("\nAverage MAE of Item based Collaborative Filtering: ", item_based_mae_mean)
print("Average RMSE of Item based Collaborative Filtering: ", item_based_rmse_mean)

### PART E ###

# USER BASED COLLABORATIVE FILTERING:

msd_user = KNNWithMeans(k = 50, sim_options = {'name': 'MSD', 'user_based': True})
msd_user_output = cross_validate(msd_user, data, ['MAE', 'RMSE'], cv = 5)
msd_user_rmse = msd_user_output['test_rmse']
msd_user_mae = msd_user_output['test_mae']
plot_graph(msd_user_rmse, msd_user_mae, title = "User Based Collaborative Filtering: MSD")

pearson_user = KNNWithMeans(k = 50, sim_options = {'name': 'pearson', 'user_based': True})
pearson_user_output = cross_validate(pearson_user, data, ['MAE', 'RMSE'], cv = 5)
pearson_user_rmse = pearson_user_output['test_rmse']
pearson_user_mae = pearson_user_output['test_mae']
plot_graph(pearson_user_rmse, pearson_user_mae, title = "User Based Collaborative Filtering: Pearson")

cosine_user = KNNWithMeans(k = 50, sim_options = {'name': 'cosine', 'user_based': True})
cosine_user_output = cross_validate(cosine_user, data, ['MAE', 'RMSE'], cv = 5)
cosine_user_rmse = cosine_user_output['test_rmse']
cosine_user_mae = cosine_user_output['test_mae']
plot_graph(cosine_user_rmse, cosine_user_mae, title = "User Based Collaborative Filtering: Cosine")

# ITEM BASED COLLABORATIVE FILTERING:

msd_item = KNNWithMeans(k = 50, sim_options = {'name': 'MSD', 'user_based': False})
msd_item_output = cross_validate(msd_item, data, ['MAE', 'RMSE'], cv = 5)
msd_item_rmse = msd_item_output['test_rmse']
msd_item_mae = msd_item_output['test_mae']
plot_graph(msd_item_rmse, msd_item_mae, title = "Item Based Collaborative Filtering: MSD")

pearson_item = KNNWithMeans(k = 50, sim_options = {'name': 'pearson', 'user_based': False})
pearson_item_output = cross_validate(pearson_item, data, ['MAE', 'RMSE'], cv = 5)
pearson_item_rmse = pearson_item_output['test_rmse']
pearson_item_mae = pearson_item_output['test_mae']
plot_graph(pearson_item_rmse, pearson_item_mae, title = "Item Based Collaborative Filtering: Pearson")

cosine_item = KNNWithMeans(k = 50, sim_options = {'name': 'cosine', 'user_based': False})
cosine_item_output = cross_validate(cosine_item, data, ['MAE', 'RMSE'], cv = 5)
cosine_item_rmse = cosine_item_output['test_rmse']
cosine_item_mae = cosine_item_output['test_mae']
plot_graph(cosine_item_rmse, cosine_item_mae, title = "Item Based Collaborative Filtering: Cosine")

### PART F ###

mae_userCF_values = []
rmse_userCF_values = []
k_values = []

for i in range(1,101):
	msd_user_k = KNNWithMeans(k = i, sim_options = {'name': 'MSD', 'user_based': True})
	msd_user_k_output = cross_validate(msd_user_k, data, ['MAE', 'RMSE'], cv = 5)
	msd_user_k_rmse = np.array(msd_user_k_output['test_rmse'])
	msd_user_k_mae = np.array(msd_user_k_output['test_mae'])
	mae_userCF_values.append(msd_user_k_mae.mean())
	rmse_userCF_values.append(msd_user_k_rmse.mean())
	k_values.append(i)

k_vs_error_graph(k_values, rmse_userCF_values, mae_userCF_values, graph_title = "K vs. Error for User based CF using MSD")

mae_itemCF_values = []
rmse_itemCF_values = []
k_values = []

for i in range(1,101):
	msd_item_k = KNNWithMeans(k = i, sim_options = {'name': 'MSD', 'user_based': False})
	msd_item_k_output = cross_validate(msd_item_k, data, ['MAE', 'RMSE'], cv = 5)
	msd_item_k_rmse = np.array(msd_item_k_output['test_rmse'])
	msd_item_k_mae = np.array(msd_item_k_output['test_mae'])
	mae_itemCF_values.append(msd_item_k_mae.mean())
	rmse_itemCF_values.append(msd_item_k_rmse.mean())
	k_values.append(i)

k_vs_error_graph(k_values, rmse_itemCF_values, mae_itemCF_values, graph_title = "K vs. Error for Item based CF using MSD")

### PART G ###

min_userCF_rmse = min(rmse_userCF_values)
userCF_best_k = np.argmin(rmse_userCF_values)
backup_userCF_K = rmse_userCF_values.index(min(rmse_userCF_values))

print("The minimum RMSE value in User based Collaborative Filtering using MSD is: ", min_userCF_rmse)
print("The K value that gives the minimum RMSE in User based Collaborative Filtering using MSD is: ", userCF_best_k)

min_itemCF_rmse = min(rmse_itemCF_values)
itemCF_best_k = np.argmin(rmse_itemCF_values)
backup_itemCF_K = rmse_itemCF_values.index(min(rmse_itemCF_values))

print("The minimum RMSE value in Item based Collaborative Filtering using MSD is: ", min_itemCF_rmse)
print("The K value that gives the minimum RMSE in Item based Collaborative Filtering using MSD is: ", itemCF_best_k)

















