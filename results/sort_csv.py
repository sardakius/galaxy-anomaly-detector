import csv
import csv
import operator


with open('output.csv', 'r', newline='') as infile, open('output_sorted_roc_auc.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader)
    writer.writerow(header)

    sorted_data = sorted(reader, key=operator.itemgetter(4), reverse=True) # descending order
    
    # Write the sorted rows
    writer.writerows(sorted_data)
