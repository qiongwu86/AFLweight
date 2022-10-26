import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.rcParams['axes.unicode_minus']=False

x_axis=[1,2,3,4,5,6,7,8,9]

y_axis=[97.67,97.73,97.74,97.77,97.90,97.75,97.49,97.22,96.28]

plt.bar(x_axis, y_axis, color='skyblue', tick_label=['0.1','0.2','0.3','0.4','0.5','0.6', '0.7', '0.8', '0.9'], alpha=0.5)

plt.xlabel('beta',size=12)

plt.ylabel('accuracy',size=12)

plt.ylim(90, 100)

for a, b in zip(x_axis, y_axis):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)

plt.show()