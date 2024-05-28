import csv

# Function to convert microjoules to joules
def microjoules_to_joules(microjoules):
    return microjoules / 1000000  # 1 microjoule = 1/1000000 joules

# Open the CSV file
with open('result.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Iterate through each row and convert energy values to joules
for row in rows:
    if row[1] == 'main':  # Check if it's the row with energy values
        for i in range(3, len(row)):  # Start from index 3 as energy values start from there
            row[i] = microjoules_to_joules(float(row[i]))  # Convert each energy value to joules

# Save the modified data back to the CSV file
with open('result_converted.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

