import numpy as np
import matplotlib.pyplot as plt

array = np.load('train_loss_backup.npz')

print('x:\n', array['x'])
print('y:\n', array['y'])

plt.figure()
plt.plot(array['y'], '-', color='#00a0ff')
plt.title('Train Loss')
plt.xlabel('Train Step')
plt.ylabel('Loss')
plt.grid()
plt.show()
plt.savefig('Train_Loss.png')