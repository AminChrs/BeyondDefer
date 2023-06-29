import random
import csv

# selecting the list of k selections
num_k_lists = 20
n_dataset = 10
k = 5

k_lists = []
for i in range(num_k_lists):
    k_list = set(random.sample(list(range(n_dataset)), k))
    while k_list in k_lists:
      k_list = set(random.sample(list(range(n_dataset)), k))
    k_lists.append(k_list)

with open('k_lists.csv', 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(k_lists)
