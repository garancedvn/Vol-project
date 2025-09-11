countries = [['France', 65], ['Germany', 80], ['Italy', 60]]
for item in countries:
    item[1] += 1
print(countries[1][1])

populations = []
for item in countries:
     populations.append(item[1])
print(populations)