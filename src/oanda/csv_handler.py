import csv


class CSVHandler:
    def __init__(self, path):
        self.path = path

    def save_to_csv(self, data):
        with open(self.path, mode='a') as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=',', quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(data)
