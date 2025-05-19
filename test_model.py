import numpy as np

# 9. Make predictions on the test set
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# 10. Display a few test images with predicted and true labels
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predicted_labels[i]}, True: {y_test[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
