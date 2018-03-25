import re
import sys
import csv
import math

# Print the database
# The database is a list of list
# In each sublist, the elements are age, gender, full path of the picture, year of birth
# and year of photo taken. 

def main(argv):

	db = []



	with open('age.csv', 'rb') as age:
		reader = csv.reader(age)
		for row in reader:
			i = 0
			for item in row:
				db.append({})
				db[i]['age']=item
				db[i]['valid']=1
				i = i + 1



					

	with open('gender.csv', 'rb') as gender:
		reader = csv.reader(gender)
		for row in reader:
			i = 0
			for item in row:
				db[i]['gender']=item
				i = i + 1
			


	with open('full_path.csv', 'rb') as fp:
		reader = csv.reader(fp)
		for row in reader:
			i = 0
			for item in row:
				db[i]['fileName']=item
				i = i + 1

	print(db[0])
	print(db[1])



	for sample in db:

		if(sample['gender']!="NaN" and sample['age']!='NaN'):
		
			sample['gender']=int(sample['gender'])
			sample['age']=int(sample['age'])
			if(sample['age']>100 or sample['age']<=0):
				sample['valid']=0
		else:
			sample['valid']=0

	for sample in db: 
		if(sample['valid']==0): 
			print(sample)




	thefile = open('test.txt', 'w')
	for sample in db:
  		thefile.write("%s\n" % sample)





	











if __name__ == "__main__":
    main(sys.argv)