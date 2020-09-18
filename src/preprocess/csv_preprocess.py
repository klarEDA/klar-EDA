# get mean, median, mode, null
# https://scikit-learn.org/stable/modules/preprocessing.html
import pandas as pd
import numpy as np
from constants import VIZ_ROOT, NUNIQUE_THRESHOLD
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

class CSVPreProcess:

	def __init__(self, input, target_col = None):
		if type(input)==str:
			self.df = pd.read_csv(input, index_col = 0)
		else:
			self.df = input
		self.col_names = list(self.df.columns)
		self.num_cols = len(self.col_names)
		self.output_format = 'png'
		self.categorical_column_list = []
		self.target_column = self.col_names[-1] if target_col == None else target_col
		self.populate_categorical_column_list()
		self.numerical_column_list = list(self.get_filtered_dataframe(include_type=np.number))

	def get_filtered_dataframe(self, include_type = [], exclude_type = []):

		if include_type or exclude_type:
			return self.df.select_dtypes(include = include_type, exclude = exclude_type)
		else:
			return self.df

	def populate_categorical_column_list(self):

		df = self.get_filtered_dataframe(exclude_type=np.number)
		if not self.categorical_column_list:
			for column in df:
				if df[column].nunique() <= NUNIQUE_THRESHOLD:
					self.categorical_column_list.append(column)

	def fill_numerical_na(self):

		columns = self.df.columns[self.df.isna().any()].tolist()
		# df1 = self.df.copy()
		# df2 = self.df.copy()
		for col in self.numerical_column_list:
			if col in columns:
				x = y = self.df[col]
				x = x.fillna(self.df[col].mean())
				# df1[col] = x
				mean_corr = x.corr(self.df[self.target_column])
				y = y.fillna(self.df[col].median())
				# df2[col] = y
				median_corr = y.corr(self.df[self.target_column])
				if(abs(mean_corr) > abs(median_corr)):
					self.df[col] = x
				else:
					self.df[col] = y
		# print(self.df.head(10))

	def fill_categorical_na(self):
		self.df = self.df.fillna("Unknown")
		# print(self.df.head(5))

	def normalize_numerical(self):
		for col in self.numerical_column_list:
			if col != self.target_column:
				self.df[col]=(self.df[col]-self.df[col].min())/(self.df[col].max()-self.df[col].min())
		# print(self.df.head())

	def encode_categorical(self):
		enc = OneHotEncoder(handle_unknown='ignore')
		self.df.reset_index(drop=True, inplace=True)
		for col in self.categorical_column_list:
			if col != self.target_column:
				enc_df = pd.DataFrame(enc.fit_transform(self.df[[col]]).toarray())
				del self.df[col]
				target = self.df[self.target_column]
				del self.df[self.target_column]
				self.df = pd.concat([self.df,enc_df], axis = 1).reset_index()
		self.df = pd.concat([self.df,target], axis = 1).reset_index()
		self.df.reset_index(drop=True, inplace=True)
		# print(self.df.head())

	def format_date(self):
		pass

	def remove_outliers(self):
		z_scores = stats.zscore(self.df)
		abs_z_scores = np.abs(z_scores)
		filtered_entries = (abs_z_scores < 3).all(axis=1)
		self.df = self.df[filtered_entries]
		# print(self.df)

# Pearson Correlation
	def remove_non_contributing_features(self):
		self.df.drop_duplicates()
		cor = self.df.corr()
		print(self.df)
		del_list = []
		# Remove related columns
		for col1 in list(self.df.columns):
			for col2 in list(self.df.columns):
				if col1 != self.target_column and col2 != self.target_column and col1 != col2:
					cor12 = abs(cor[col1][col2])
					if(cor12 > 0.6):
						if(cor[col1][self.target_column] > cor[col2][self.target_column]):
							del_list.append(col2)
						else:
							del_list.append(col1)
		self.df = self.df.drop(columns = del_list, axis = 1)
		print(self.df)
		# Removing unrelated columns
		del_list = []
		for col in list(self.df.columns):
			if(abs(cor[col][self.target_column]) < 0.1 and col in list(self.df.columns)):
				del_list.append(col)
		self.df = self.df.drop(columns = del_list, axis = 1)
		print(self.df)
