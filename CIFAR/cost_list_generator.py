import random
import csv

# selecting the list of cost selections
num_cost_lists = 20
k = 10

cost_lists = []
for i in range(num_cost_lists):
    # generate k number of numbers that are uniformly distributed between 0 and 1
    cost_list = [random.uniform(0, 1) for i in range(k)]
    cost_lists.append(cost_list)

# for now I set the lists be [1]*10
cost_lists = [[1]*10]

with open('cost_lists.csv', 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(cost_lists)
