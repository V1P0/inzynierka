def minPair(a, b):
    if(a[0] > b[0]):
        return (a, b)
    elif(a[0] < b[0]):
        return (b, a)
    elif(a[1] > b[1]):
        return (a, b)
    else:
        return (b, a)

def main():
    width = 9
    height = 13
    for j in range(1, height-1):
        print(minPair((0, j), (width-1, j)), end=', ')
    for i in range(1, 3):
        print(minPair((i, 1), (i, height-2)), end=', ')


if __name__ == '__main__':
    main()