import os
import csv

from utils import open_write_file


DATA_DIR = 'data'


def process_csv():
    ple_data = []
    plqy_path = open_write_file(DATA_DIR, 'cu_halides.csv')
    with open(plqy_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader) 
        for row in reader:
            ple_value = row[6]
            if '-' in ple_value:    # value range
                # ! if value is range, choose lower value (conservative estimate)
                ple_value = ple_value.split('-')[0]
            ple_data.append([row[2], ple_value])

    csv_path = open_write_file(DATA_DIR, 'ple_original.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(ple_data)


if __name__ == "__main__":
    process_csv()
    pass