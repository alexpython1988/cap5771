import csv

mapping = {1:"o", 2:"tw", 3:"th", 4:"fo", 5:"fi", 6:"si", 7:"se", 8:"e"}

with open("dataset1.csv", "r") as f:
	with open("dataset1_new.csv", "w", newline="") as fw:
		reader = csv.DictReader(f)
		headers = reader.fieldnames + ['label']
		writer = csv.DictWriter(fw, fieldnames=headers)
		writer.writeheader()
		for each in reader:
			each['label'] = mapping[int(each['cluster'])]
			writer.writerow(each)
