import laspy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the two LAS files
las_pred = laspy.read("")
las_gt   = laspy.read("")

# Extract classification arrays
y_pred = las_pred.label
y_true = las_gt.classification

# Check lengths
print(len(y_pred), len(y_true))



cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()
