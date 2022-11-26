res = []

for i in range(200):
    a = i
    invalid = False
    for j in range(4):
        if a % 2 == 1:
            res.append((i, False))
            invalid = True
            break
        a = a // 2
    if not invalid:
        res.append((i, True))

for i in res:
    print(i)
