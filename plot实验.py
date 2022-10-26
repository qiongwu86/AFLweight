# 导入相应的库(包)
import numpy as np  # 生成数据的包
import matplotlib.pyplot as plt  # 作图的包

x = np.arange(1,6) * 2


y = [91.94, 95.66, 96.71, 97.46, 97.85]
y1 = [92.89, 95.78, 97.13, 97.56, 97.90]



plt.bar(
    x, y,

    width=0.5,

    color="gold",
    label="AFL")

plt.bar(x + 0.5, y1,
        width=0.5,
        color="skyblue",
        label="our scheme")

plt.xticks(x + 0.5 / 2, x)

plt.legend()

plt.xlabel(
    "round",

    size=12,

    color="black")

plt.ylabel("accuracy",
           size=12,
           color="black")


plt.ylim(85, 100)

for a, b in zip(x, y):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=6)
for a, b in zip(x + 0.5, y1):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=6)


plt.show()



x1 = np.arange(1,6) * 2

yy = [0.542, 0.362, 0.281, 0.244, 0.221]
yy1 = [0.526, 0.356, 0.280, 0.243, 0.217]



plt.bar(
    x1, yy,

    width=0.5,

    color="gold",

    label="AFL")

plt.bar(x1 + 0.5, yy1,
        width=0.5,
        color="skyblue",
        label="our scheme")

plt.xticks(x1 + 0.5 / 2, x1)

plt.legend()

plt.xlabel(
    "round",

    size=12,

    color="black")

plt.ylabel("loss",
           size=12,
           color="black")


plt.ylim(0, 0.8)

for a, b in zip(x1, yy):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=6)
for a, b in zip(x1 + 0.5, yy1):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=6)


plt.show()


