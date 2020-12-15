import os
import shutil
import random

"""This code will take images of skin moles and determine 
	whether they are benign and malignant tumours"""

#filenames are edited to "images" folder and "labels.csv"

#create "train" folder with subfolders: "benign" and "malignant"

seed = 1 #allows exact result if run mutliple times
random.seed(seed)

directory = "ISIC/iamges/"
train = "data/train/"
test = "data/test/"
validation = "/data/validation/"

#Make data subfolders
os.makedirs(train + "benign/")
os.makedirs(train + "malignant/")

os.makedirs(test + "benign/")
os.makedirs(test + "malignant/")

os.makedirs(validation + "benign/")
os.makedirs(validation + "malignant/")

#initialize counters for number of examples in each subfolder
test_examples = train_examples = validation_examples = 0

#labels contain whether malignant of benign
for line in open("ISIC/labels.csv").readlines()[1:]: #read from first actual data row, ignore column names row
	split_line = line.split(",") #csv file so split on commas
	img_file = split_line[0] #filename
	benign_malign = split_line[1] #label value 0 or 1

	#Here we randomly assign each image to train/test/validation
	random_num = random.random()

	if random_num < 0.8:	#We want 80% to be training set
		location = train 
		train_examples +=1

	elif random_num < 0.9:	#We want 10% to be validation set
		location = validation
		validation_examples += 1

	else: 					#We want 10% to be test set
		location = test
		test_examples += 1

#Use shutil.copy to move from ISIC/images to location/benign
	if int(float(benign_malign)) == 0:	#change 1.0/0.0 to 1/0
		shutil.copy(
			"ISIC/images/" + img_file + ".jpg",
			location + "benign/" + img_file + ".jpg"
			)

#Use shutil.copy to move from ISIC/images to location/malignant
	elif int(float(benign_malign)) == 1:	#change 1.0/0.0 to 1/0
		shutil.copy(
			"ISIC/images/" + img_file + ".jpg",
			location + "malignant/" + img_file + ".jpg"
			)

print(f"Number of training examples {train_examples}")
print(f"Number of testing examples {test_examples}")
print(f"Number of validation examples {validation_examples}")