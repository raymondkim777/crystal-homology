import os
import csv

from utils import open_write_file


DATA_DIR = 'data'


def process_csv():
    plqy_data = []
    plqy_path = open_write_file(DATA_DIR, 'cu_plqy.csv')
    with open(plqy_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader) 
        for row in reader:
            plqy_data.append([row[2], row[9]])

    csv_path = open_write_file(DATA_DIR, 'plqy.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(plqy_data)


if __name__ == "__main__":
    process_csv()
    pass