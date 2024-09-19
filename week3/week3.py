import os


folder_path = '20021010_spam/spam'
files = os.listdir (folder_path)

table = list()

# For each filename
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    f = open(file_path, 'r', encoding="latin_1")
    table.append(f.read())

print(table)