
onets = []
with open("Occupation Data.txt") as f:
    for line in f:
        onets.append(line.split()[0].replace('-', '').replace('.', ''))

onets =  onets[1:]