f = open("synset_words.txt", "r")
lines = f.read().split("\n")
m = {}
a = {}
for class_index, line in enumerate(lines):
    try:
        m[class_index] = {}
        m[class_index]["name"] = line.split()[0]
        m[class_index]["human_names"] = line[10:].split(", ")

        name = line.split()[0]
        a[name] = {}
        a[name]["index"] = class_index
        a[name]["human_names"] = line[10:].split(", ")

    except:
        pass

print(m)
print(a)
