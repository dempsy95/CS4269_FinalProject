import re
import sys
import csv

# Print the database
# The database is a list of list
# In each sublist, the elements are age, gender, full path of the picture, year of birth
# and year of photo taken. 

def main(argv):

	db = [];

	with open('age.csv', 'rb') as age:
			reader = csv.reader(age)
			for row in reader:
				for item in row:
					db.append([item])

	with open('gender.csv', 'rb') as gender:
		reader = csv.reader(gender)
		for row in reader:
			i = 0
			for item in row:
				db[i].append(item)
				i = i + 1

	with open('full_path.csv', 'rb') as fp:
		reader = csv.reader(fp)
		for row in reader:
			i = 0
			for item in row:
				db[i].append(item)
				i = i + 1

	with open('year_of_birth.csv', 'rb') as yob:
		reader = csv.reader(yob)
		for row in reader:
			i = 0
			for item in row:
				db[i].append(item)
				i = i + 1

	with open('year_of_photo_taken.csv', 'rb') as yot:
		reader = csv.reader(yot)
		for row in reader:
			i = 0
			for item in row:
				db[i].append(item)
				i = i + 1

	print(db)

if __name__ == "__main__":
    main(sys.argv)