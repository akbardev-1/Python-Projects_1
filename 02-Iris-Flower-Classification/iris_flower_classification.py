from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load Iris flower dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Machine Learning model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

# Train the AI model
model.fit(X_train, y_train)

# Test the AI model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("   IRIS FLOWER CLASSIFICATION")
print("================================")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Predict three example flowers
print("\n--- Example Predictions ---")

flowers = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.0, 5.2, 2.3]
]

example_predictions = model.predict(flowers)

for i, prediction in enumerate(example_predictions):
    print(f"Flower {i + 1}: {iris.target_names[prediction].capitalize()}")

# Interactive flower prediction
print("\n================================")
print("   TEST YOUR OWN FLOWER")
print("================================")

while True:

    choice = input("\nWould you like to test a flower? (yes/no): ").lower().strip()

    if choice == "no":
        print("\nThank you for using Iris Flower AI!")
        break

    elif choice == "yes":

        try:
            print("\nEnter flower measurements in centimeters:")

            sepal_length = float(input("Sepal length: "))
            sepal_width = float(input("Sepal width: "))
            petal_length = float(input("Petal length: "))
            petal_width = float(input("Petal width: "))

            if min(sepal_length, sepal_width, petal_length, petal_width) <= 0:
                print("\nMeasurements must be positive numbers.")
                continue

            new_flower = [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]]

            prediction = model.predict(new_flower)

            flower_name = iris.target_names[prediction[0]]

            print("\n------------------------------")
            print("AI Prediction:", flower_name.upper())
            print("------------------------------")

        except ValueError:
            print("\nInvalid input! Please enter numbers only.")

    else:
        print("\nPlease type yes or no.")