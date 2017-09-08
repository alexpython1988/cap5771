import csv

def create_submission_csv(txt_file, csv_file):
    with open(txt_file, "r") as f:
        with open(csv_file, "w", newline='') as f1:
            writer = csv.writer(f1)
            for each in f:
                l = []
                data = each[:-1].split('\t')
                l.append(data[0])
                l.append(data[1])
                l.append(1)
                writer.writerow(l)

def main():
    #create_submission_csv("processed_full_cover.txt", "sub_filtered_full_cover.csv")
    create_submission_csv("processed_neg_56588.txt", "sub_filtered_neg_56588.csv")

if __name__ == '__main__':
    main()