import numpy as np
import matplotlib.pyplot as plt

# y = mx + b
# alter m and x until loss minimized

# arbitrary data
x = np.array([ 2,  4,  5,  7,  9, 10, 12, 14, 15, 17,
     18, 20, 21, 23, 24, 26, 28, 29, 31, 33,
     34, 36, 37, 39, 41, 42, 44, 46, 47, 49,
     50, 52, 54, 55, 57, 59, 60, 62, 64, 65,
     67, 68, 70, 72, 73, 75, 77, 78, 80, 82 ])
y = ([ 7, 11, 10, 15, 18, 17, 23, 25, 28, 27,
     31, 34, 32, 39, 38, 43, 46, 44, 50, 53,
     51, 57, 59, 61, 65, 63, 68, 72, 70, 76,
     79, 77, 83, 86, 84, 91, 89, 94, 97, 96,
    101, 105, 102, 109, 112, 110, 117, 115, 121, 125 ])

m = 0.0
b = 0.0
rate = 0.00001
cycles = 1000

for i in range(cycles):
    line = m * x + b

    residual = (line-y)
    total_error = np.mean(residual**2)
    if(i%10==0):
        print(f"Cycle: {i} | Error: {total_error}")
    # error = sum(y-hat - y)**2
    # error = sum(y-hat**2 - 2(y-hat)(y) + y**2)
    # error(m,b) = sum((mx + b)**2 - 2(mx+b)(y) + y**2)
    # de/dm = sum(2x(y-hat)-2yx)
    # de/dm = sum(2x(y-hat - y)) = sum(2x * error)
    # de/db = sum(2(mx + b)-2y)
    # de/db = sum(2(y-hat - y)) = sum(2 * error)

    dm = np.mean(2 * x * residual)
    db = np.mean(2 * residual)

    m -= rate * (dm)
    b -= rate * (db)

print("m:", m, "| b:", b)

plt.scatter(x,y)
plt.plot(x, m*x+b)
plt.show()
