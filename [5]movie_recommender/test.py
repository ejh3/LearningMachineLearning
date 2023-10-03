'red', 'cyan', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray', 'olive', 'cyan'
# Close the progress bar
COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
import matplotlib.pyplot as plt
import numpy as np

# Create x values
x = np.linspace(0, 10, 100)

# Generate y values for each line
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x)
y4 = np.sqrt(x)
y5 = np.log(x)

# Plot each line with different colors
plt.plot(x, y1, color='c', label='Line 1')
plt.plot(x, y2, color='m', label='Line 2')
plt.plot(x, y3, color='y', label='Line 3')
plt.plot(x, y4, color='purple', label='Line 4')
plt.plot(x, y5, color='orange', label='Line 5')

# Add legend and title
plt.legend()
plt.title('Five Different Lines')

# Display the plot
plt.show()