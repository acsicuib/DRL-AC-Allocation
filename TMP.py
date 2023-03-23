weigthRange = list(range(11))
combinationsWeightTC = list(zip(weigthRange,weigthRange[::-1])) #[(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1), (10, 0)]

print(combinationsWeightTC)


for e,(wt,wc) in enumerate(combinationsWeightTC):
    codeW = str(int(wt))+str(int(wc))
    text = "scp isaac@172.30.248.28:/home/isaac/projects/DRL-AC-Allocation/savedModels/E99_9_9_99_w%s.pth ."%codeW
    print(text)