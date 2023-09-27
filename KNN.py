import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance

training = []
fingerprint = []

with open('trainingData.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if row[523] == '0':
            training.append(row)

with open('validationData.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if row[523] == '0':
            fingerprint.append(row)

for i in range(len(training)):
    for j in range(len(training[i])-1):
        training[i][j] = float(training[i][j])
for i in range(len(fingerprint)):
    for j in range(len(fingerprint[i])-1):
        fingerprint[i][j] = float(fingerprint[i][j])

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(x, y):
    return distance.euclidean(x, y)

# Define a function to find the k closest fingerprints to the new fingerprint
def find_k_closest_fingerprints(new_fingerprint, training_data, k):
    distances = []
    for data_point in training_data:
        distance_to_data_point = euclidean_distance(new_fingerprint[0:520], data_point[0:520])
        distances.append((data_point, distance_to_data_point))
    distances.sort(key=lambda x: x[1])
    return [x[0] for x in distances[:k]]

# Define a function to calculate the estimated floor for a new fingerprint using k-NN
def knn_predict_floor(new_fingerprint, training_data, k):
    closest_fingerprints = find_k_closest_fingerprints(new_fingerprint[:-1], training_data, k)
    floor_counts = {}
    for fingerprint in closest_fingerprints:
        floor = fingerprint[522]
        if floor in floor_counts:
            floor_counts[floor] += 1
        else:
            floor_counts[floor] = 1
    return max(floor_counts, key=floor_counts.get)

# Calculate the estimated floor for each fingerprint in the validation data using k-NN with k = 1, 5, and 9
true_floors = []
predicted_floors_k1 = []
predicted_floors_k5 = []
predicted_floors_k9 = []
for key, new_fingerprint in enumerate(fingerprint):
    true_floor = new_fingerprint[522]
    predicted_floor_k1 = find_k_closest_fingerprints(new_fingerprint[:-1], training, 1)[0][522]
    predicted_floor_k5 = knn_predict_floor(new_fingerprint, training, 5)
    predicted_floor_k9 = knn_predict_floor(new_fingerprint, training, 9)
    true_floors.append(true_floor)
    predicted_floors_k1.append(predicted_floor_k1)
    predicted_floors_k5.append(predicted_floor_k5)
    predicted_floors_k9.append(predicted_floor_k9)
    print("enter" + str(key))

# Calculate the accuracy of the floor predictions for each value of k
accuracy_k1 = sum([1 for i in range(len(true_floors)) if true_floors[i] == predicted_floors_k1[i]]) / len(true_floors)
accuracy_k5 = sum([1 for i in range(len(true_floors)) if true_floors[i] == predicted_floors_k5[i]]) / len(true_floors)
accuracy_k9 = sum([1 for i in range(len(true_floors)) if true_floors[i] == predicted_floors_k9[i]]) / len(true_floors)
print(f"The accuracy of the floor predictions for k = 1 is {accuracy_k1:.2%}")
print(f"The accuracy of the floor predictions for k = 5 is {accuracy_k5:.2%}")
print(f"The accuracy of the floor predictions for k = 9 is {accuracy_k9:.2%}")

# Calculate the confusion matrix for each value of k
confusion_mat_k1 = confusion_matrix(true_floors, predicted_floors_k1)
confusion_mat_k5 = confusion_matrix(true_floors, predicted_floors_k5)
confusion_mat_k9 = confusion_matrix(true_floors, predicted_floors_k9)
print("Confusion matrix for k = 1:")
print(confusion_mat_k1)
print("Confusion matrix for k = 5:")
print(confusion_mat_k5)
print("Confusion matrix for k = 9:")
print(confusion_mat_k9)