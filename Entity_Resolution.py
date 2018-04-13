import numpy as np
import pandas as pd
import usaddress as ad
from vincenty import vincenty
from jellyfish import levenshtein_distance, jaro_winkler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier

list_remove = ['!',',','(',')','?','.', '\'', '/','\"','-', ' ']

def str_cleaning(i):
	i = str(i)
	i = i.lower()
	for j in list_remove:
		if j in i:
			i = i.translate({ord(x): '' for x in list_remove})
	return i

def phone_cleaning(i):
	if i != '':
		first_half, second_half = i.split(' ')
		first = first_half.split('(')[1].split(')')[0]
		second, third = second_half.split('-')
		phone_number = first + second + third
	else:
		phone_number = ''
	return phone_number

def website_cleaning(string):
	for i in ['.com', '.net', '.org']:
		string = string.split(i)[0]
	L = ['http://', 'www.']
	for i in L:
		string = string.replace(i, "")
	return string

def cleaning_locu(df):
	df['name'] = df['name'].apply(str_cleaning)
	df['website'] = df['website'].apply(website_cleaning).apply(str_cleaning)
	df['street_address_norm'] = df['street_address'].apply(str_cleaning)
	return df

def cleaning_four(df):
	df['name'] = df['name'].apply(str_cleaning)
	df['phone'] = df['phone'].apply(phone_cleaning)
	df['website'] = df['website'].apply(website_cleaning).apply(str_cleaning)
	df['street_address_norm'] = df['street_address'].apply(str_cleaning)
	return df

def get_address(df):
	list_add = ['AddressNumber', 'StreetNamePreDirectional', 'StreetName', 'StreetNamePostType']
	L = []
	for index, row in df.iterrows():
		try:
			dic = ad.tag(row['street_address'])[0]
		except ad.RepeatedLabelError as e:
			continue
		dic['id'] = row['id']
		L.append(dic)
	df_result = pd.DataFrame(L)

	df_result = df_result[list_add + ['id']]
	df_result['StreetName'] = df_result['StreetName'].apply(str_cleaning).replace("nan","")
	df_result['AddressNumber'] = df_result['AddressNumber'].replace("NaN","")
	df_result['StreetNamePreDirectional'] = df_result['StreetNamePreDirectional'].apply(str_cleaning).replace(['e', 'w', 'n', 's', 'sw', 'se', 'nw', 'ne'], 
																										  ['east', 'west', 'north', 'south', 'southwest', 'southeast', 'northwest', 'northeast'])
	df_result['StreetNamePostType'] = df_result['StreetNamePostType'].apply(str_cleaning).replace(['st', 'ave', 'plz', 'pl', 'sq', 'blvd', 'pkwy', 'ln', 'riv'], 
																							  ['street', 'avenue', 'plaza', 'place', 'square', 'boulevard', 'parkways', 'lane', 'river'])
	return df_result

def compute_distances(field_x, field_y, row):
	if (row[field_x] != '') | (row[field_y] != ''):
		leven_distance = levenshtein_distance(row[field_x], row[field_y])
		jw_distance = jaro_winkler(row[field_x], row[field_y])
	else:
		leven_distance = np.NaN
		jw_distance = np.NaN
	return leven_distance, jw_distance

def check_equality(field_x, field_y, row):
	if (row[field_x] == row[field_y]) & (row[field_x] != ''):
		value = 1
	elif (row[field_x] == '') | (row[field_y] == ''):
		value = np.NaN
	else:
		value = 0
	return value

def feature_creation(df):

	dist = []
	leven_phone = []
	jw_phone = []
	leven_name = []
	jw_name = []
	leven_street_name = []
	jw_street_name = []
	leven_address = []
	jw_address = []
	same_postal_code = []
	same_street_number = []
	same_address = []
	same_website = []
	same_phone = []
	same_name = []
	same_street_name = []
	

	for index, row in df.iterrows():
	
		if (row['latitude_x'] != '') & (row['longitude_x'] != '') & (row['latitude_y'] != '') & (row['longitude_y'] != ''): 
			distance = vincenty((row['latitude_x'],row['longitude_x']), (row['latitude_y'],row['longitude_y'])) 
		else:
			distance = np.NaN
		dist.append(distance)
	
		leven_distance, jw_distance = compute_distances('phone_x', 'phone_y', row)
		leven_phone.append(leven_distance)
		jw_phone.append(jw_distance)
		
		leven_distance, jw_distance = compute_distances('name_x', 'name_y', row)
		leven_name.append(leven_distance)
		jw_name.append(jw_distance)
	
		leven_distance, jw_distance = compute_distances('StreetName_x', 'StreetName_y', row)
		leven_street_name.append(leven_distance)
		jw_street_name.append(jw_distance)
		
		leven_distance, jw_distance = compute_distances('street_address_norm_x', 'street_address_norm_y', row)
		leven_address.append(leven_distance)
		jw_address.append(jw_distance)
	
		value = check_equality('postal_code_x', 'postal_code_y', row)
		same_postal_code.append(value)
	
		value = check_equality('AddressNumber_x', 'AddressNumber_y', row)
		same_street_number.append(value)
	
		value = check_equality('street_address_norm_x', 'street_address_norm_y', row)
		same_address.append(value)
		
		value = check_equality('website_x', 'website_y', row)
		same_website.append(value)
		
		value = check_equality('phone_x', 'phone_y', row)
		same_phone.append(value)
		
		value = check_equality('name_x', 'name_y', row)
		same_name.append(value)
		
		value = check_equality('StreetName_x', 'StreetName_y', row)
		same_street_name.append(value)
	
	df['dist'] = dist
	df['leven_phone'] = leven_phone
	df['jw_phone'] = jw_phone
	df['leven_name'] = leven_name
	df['jw_name'] = jw_name
	df['leven_street_name'] = leven_street_name
	df['jw_street_name'] = jw_street_name
	df['leven_address'] = leven_address
	df['jw_address'] = jw_address
	df['same_postal_code'] = same_postal_code
	df['same_street_number'] = same_street_number
	df['same_address'] = same_address
	df['same_website'] = same_website
	df['same_phone'] = same_phone
	df['same_name'] = same_name
	df['same_street_name'] = same_street_name
	
	return df

def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
	"""
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.
	"""
	df_locu_train = pd.read_json(locu_train_path).replace([None], [''])
	df_foursquare_train = pd.read_json(foursquare_train_path).replace([None], [''])
	df_locu_test = pd.read_json(locu_test_path).replace([None], [''])
	df_foursquare_test = pd.read_json(foursquare_test_path).replace([None], [''])
	df_matches = pd.read_csv(matches_train_path)

	print('Lecture success')

	list_feature = ['id','latitude', 'longitude', 'name', 'phone', 'postal_code', 'street_address', 'website']

	locu_train = df_locu_train[list_feature]
	four_train = df_foursquare_train[list_feature]
	locu_train = cleaning_locu(locu_train)
	four_train = cleaning_four(four_train)
	df_locu = get_address(locu_train)
	df_four = get_address(four_train)
	locu = pd.merge(df_locu, locu_train, on='id', how='outer')
	four = pd.merge(df_four, four_train, on='id', how='outer')

	df1 = locu
	df1['key'] = 0
	df2 = four
	df2['key'] = 0
	all_in = pd.merge(df1, df2, on='key')
	all_in = all_in.fillna('')
	L_matches = []
	for index, row in df_matches.iterrows():
		L_matches.append([row['locu_id'], row['foursquare_id']])

	l_match = []
	for index, row in all_in.iterrows():
		if [row['id_x'],row['id_y']] in L_matches:
			l_match.append(1)
		else:
			l_match.append(0)
	all_in['is_match'] = l_match

	print('Entering feature creation train set')

	all_in = feature_creation(all_in)
	all_in_filter = all_in[(all_in['dist'] <= 1) 
	   | (all_in['same_website'] == 1)
	   | (all_in['same_street_name'] == 1)
	   | (all_in['same_postal_code'] == 1)
	   | (all_in['same_street_number'] == 1)
	   | (all_in['same_address'] == 1)
	   | (all_in['same_phone'] == 1)
	   | (all_in['same_name'] == 1)].reset_index(drop=True)

	print('Done : number of combinations kept for training is', np.shape(all_in_filter)[0] )

	features = ['dist', 
			'leven_phone', 'jw_phone',
			'leven_name', 'jw_name', 
			'leven_street_name', 'jw_street_name', 
			'leven_address', 'jw_address',
			'same_postal_code','same_street_number', 
			'same_address', 'same_website',
			'same_phone', 'same_name', 'same_street_name']

	locu_test = df_locu_test[list_feature]
	four_test = df_foursquare_test[list_feature]
	locu_test = cleaning_locu(locu_test)
	four_test = cleaning_four(four_test)
	df_locu_test = get_address(locu_test)
	df_four_test = get_address(four_test)
	locutest = pd.merge(df_locu_test, locu_test, on='id', how='outer')
	fourtest = pd.merge(df_four_test, four_test, on='id', how='outer')
	df1 = locutest
	df1['key'] = 0
	df2 = fourtest
	df2['key'] = 0
	all_in_test = pd.merge(df1, df2, on='key')
	all_in_test = all_in_test.fillna('')
	print('Entering feature creation test set')

	all_in_test = feature_creation(all_in_test)
	all_in_test_filter = all_in_test[(all_in_test['dist'] <= 1) 
	   | (all_in_test['same_website'] == 1)
	   | (all_in_test['same_street_name'] == 1)
	   | (all_in_test['same_postal_code'] == 1)
	   | (all_in_test['same_street_number'] == 1)
	   | (all_in_test['same_address'] == 1)
	   | (all_in_test['same_phone'] == 1)
	   | (all_in_test['same_name'] == 1)].reset_index(drop=True)


	print('Done : number of probable combinations in test set is', np.shape(all_in_test_filter)[0])

	X_train = all_in_filter[features]
	Y_train = all_in_filter['is_match']
	X_test = all_in_test_filter[features]

	impt = Imputer()
	X_train_impute = impt.fit_transform(X_train)
	X_test_impute = impt.transform(X_test)
	sm = SMOTE(random_state=12, ratio=0.03)
	X_train_res, Y_train_res = sm.fit_sample(X_train_impute, Y_train)

	xgb = XGBClassifier(max_depth=10, n_estimators=300)
	xgb.fit(X_train_res, Y_train_res)
	Y_pred = xgb.predict(X_test_impute)

	df_matches_test = all_in_test_filter[Y_pred == 1][['id_x', 'id_y']].reset_index(drop=True)
	df_matches_test.rename(columns={'id_x': 'locu_id', 'id_y': 'foursquare_id'}, inplace=True)
	df_matches_test.to_csv('data/matches_test.csv', index=False)
