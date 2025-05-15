import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(array)

# visualizar el array 
plt.figure(figsize=(5, 5))
# agregamos un data frame 
df = pd.DataFrame(array)
print(df)
# el data frame se muestra con matplotlib
plt.imshow(df, cmap='gray', interpolation='nearest')

plt.imshow(array, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.title('Array Visualization')
plt.show()
# guardar la imagen
plt.savefig('array_visualization.png')
# guardar el array en un archivo de texto
np.savetxt('array.txt', array, fmt='%d')

