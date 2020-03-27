import matplotlib.pyplot as plt
import pandas as pd
from math import pi


values = [0.07224128, 0.26003403, 0.24291298, 0.4248117]
k = 4
categories = [i/k for i in range(1, k+1)]
N = len(categories)
values += values[:1]

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi + pi/2 for n in range(N)]
angles += angles[:1]

ax = plt.subplot(221, polar=True)
ax.set_title('Artificial (n=4)', pad=15)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, size=10)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=0)
plt.ylim(0, max(values))

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

###########################################################################
values = [0.05001922, 0.04998047, 0.04995556, 0.04991261, 0.04986967, 0.04986084,
 0.04987006, 0.04986381, 0.04988073, 0.04992097, 0.04995678, 0.04998309,
 0.04999234, 0.0500224,  0.05006647, 0.05008881, 0.05013887, 0.05015089,
 0.05022192, 0.05024446]
k = 20
n_labels = 8
_step = int(k/n_labels)
categories = list(reversed([(str(round(1-i/(1.5*k), 2)) if i%_step == 0 else '') for i in range(1, k+1)]))
N = len(categories)
values += values[:1]

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [((n / float(N) * 2 * pi + pi/2) if (n / float(N) * 2 * pi < 3*pi/2) else (n / float(N) * 2 * pi -3*pi/2))
          for n in range(N)]
angles += angles[:1]

ax = plt.subplot(222, polar=True)
ax.set_title('Diabetic Retinopathy (n=20)', pad=15)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, size=10)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=0)
plt.ylim(0, max(values))

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

###########################################################################
values = [0.02954461, 0.03050432, 0.02914616, 0.02920798, 0.02912963, 0.02948952,
 0.02851929, 0.0301881,  0.02964462, 0.02967163, 0.03227828, 0.03245265,
 0.03166857, 0.03185653, 0.0323494,  0.03228316, 0.0334429,  0.03413522,
 0.0333294,  0.03437672, 0.03292534, 0.03362983, 0.03401273, 0.0340814,
 0.03518591, 0.03453589, 0.03383726, 0.03455713, 0.03427709, 0.03443068,
 0.03530805]
k = 31
n_labels = 8
_step = int(k/n_labels)
categories = [(str(round(i/(2*k), 2)) if i%_step == 0 else '') for i in range(1, k+1)]
N = len(categories)
values += values[:1]

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [((n / float(N) * 2 * pi + pi/2) if (n / float(N) * 2 * pi < 3*pi/2) else (n / float(N) * 2 * pi -3*pi/2))
          for n in range(N)]
angles += angles[:1]

ax = plt.subplot(223, polar=True)
ax.set_title('Breast Cancer (n=31)', pad=15)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, size=10)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=0)
plt.ylim(0, max(values))

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

###########################################################################
values = [0.03596113, 0.03507108, 0.03522439, 0.03623687, 0.03594248, 0.03521154,
 0.03584657, 0.03594892, 0.036092,   0.03542879, 0.03539941, 0.03570591,
 0.03573843, 0.03538225, 0.03569333, 0.03577379, 0.03632802, 0.03596115,
 0.03616053, 0.03580305, 0.03545496, 0.03533748, 0.03598798, 0.03555118,
 0.03558715, 0.03600788, 0.03565907, 0.03550467]
k = 28
n_labels = 8
_step = int(k/n_labels)
categories = [(str(round(i/k, 2)) if i%_step == 0 else '') for i in range(1, k+1)]
N = len(categories)
values += values[:1]

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [((n / float(N) * 2 * pi + pi/2) if (n / float(N) * 2 * pi < 3*pi/2) else (n / float(N) * 2 * pi -3*pi/2))
          for n in range(N)]
angles += angles[:1]

ax = plt.subplot(224, polar=True)
ax.set_title('Parkinson Speech (n=28)', pad=15)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, size=10)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=0)
plt.ylim(0, max(values))

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

###########################################################################

plt.tight_layout()
plt.show()
