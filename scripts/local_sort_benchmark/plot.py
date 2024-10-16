import matplotlib.pyplot as plt
import numpy as np

# Data setup
x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])  # Number of tuples (2^20 multiples)
avx_sort = np.array([35, 33, 30, 28, 25, 22, 20, 18, 16])  # AVX sort throughput
cpp_sort = np.array([10, 10, 9, 9, 8, 8, 7, 7, 6])         # C++ STL sort throughput
speedup = avx_sort / cpp_sort                               # Calculated speedup

# Create the figure and a set of subplots
fig, ax1 = plt.subplots()

# Plotting the line plots (AVX and C++ sorts) on the left y-axis
ax1.set_xlabel('number of tuples in $R$ (in $2^{20}$)')
ax1.set_ylabel('sort throughput [M. tuples/sec]')
ax1.plot(x, avx_sort, 'o-', label='AVX sort', color='black')
ax1.plot(x, cpp_sort, 's-', label='C++ STL sort', color='black')

# Modify the spines to remove the top and bottom
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(True)

# Remove the bottom ticks
ax1.tick_params(bottom=False)
ax1.set_xticks(x)
ax1.tick_params(axis='y')

# Adding arrows to the y-axes
ax1.yaxis.set_label_position("left")
ax1.spines['left'].set_position(('outward', 5))
ax1.annotate('', xy=(0, 1), xycoords='axes fraction', xytext=(0, 1.05),
             arrowprops=dict(arrowstyle="->", color='black'))

ax1.legend(loc='upper right')

# Create another y-axis for the bar chart (right side)
ax2 = ax1.twinx()
ax2.set_ylabel('speedup')
ax2.bar(x, speedup, color='gray', width=0.7, alpha=0.7)

# Remove the spines and ticks from ax2 (except for right)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(True)

# Adding an arrow to the right y-axis
ax2.annotate('', xy=(1, 1), xycoords='axes fraction', xytext=(1, 1.05),
             arrowprops=dict(arrowstyle="->", color='black'))

ax2.tick_params(axis='y')

# Show plot
plt.show()
