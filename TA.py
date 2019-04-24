import glob

data = []
path = 'F:/Grace Tika/TA/Dataset Berita/Budaya/*.txt'
file = glob.glob(path)
for name in file:
    a = open(name, 'r')
    s = a.read()
    data.append(s)
#print(data)

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopword = [line.rstrip('\n\r') for line in open('stopwords.txt')]

for i in range(len(data)):
    data[i] = data[i].rstrip()
    data[i] = data[i].lower()
    data[i] = re.sub(r'[^a-z]',' ',data[i])
    data[i] = data[i].split()
    temp = []
#     print(data[i])
    for j in range(len(data[i])):
        if data[i][j] not in stopword:
            temp.append(data[i][j])
    data[i] = temp
#     print(data[i])
    for j in range(len(data[i])):
        data[i][j] = stemmer.stem(data[i][j])

tas = []
for i in range(len(data)):
    tas += data[i]
tas = list(set(tas))
# print(tas)

bow = []
for i in range(len(data)):
    tem = []
    for j in range(len(tas)):
        tem.append(data[i].count(tas[j]))
    bow.append(tem)
for i in range(len(bow)):
    print(bow)