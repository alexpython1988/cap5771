import csv

continent = ['Europe', 'Asia', 'Oceania', 'North America', 'South America', 'Africa']
c2l = dict()
for i, each in enumerate(continent):
	c2l[each] = i+1
print(c2l)

with open("proj1_data.csv", "w", newline='') as fw:
	with open("project1_data.csv", "r") as f:
		reader = csv.DictReader(f)
		head = reader.fieldnames
		head.append("label")
		print(head)
		writer = csv.DictWriter(fw, head)
		writer.writeheader()
		for each in reader:
			label = c2l[each['continent']]
			each['label'] = label
			writer.writerow(each) 