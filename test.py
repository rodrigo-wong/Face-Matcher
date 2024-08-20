from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

# Extract faces from the reference and target images
faces_reference = DeepFace.extract_faces("Test/Group-Elon.jpeg")

# Paths to the images
img1_path = "Database/ElonMusk/Elon-Musk1.png"
img2_path = "Test/Group-Elon.jpeg"
faces = DeepFace.extract_faces(img2_path,target_size=(224, 224), detector_backend="retinaface")
# Display each face separately
for i, face in enumerate(faces):
    plt.subplot(1, len(faces), i + 1)  # Create a subplot for each face
    plt.imshow(face['face'])
    plt.axis('off')
plt.show()

results = DeepFace.verify(img1_path, img2_path, model_name="VGG-Face", detector_backend="retinaface")
# Display each face separately

print(results)

# Get the facial area coordinates from the reference image
img1_facial_area = results['facial_areas']["img1"]
img2_facial_area = results['facial_areas']['img2']

# Read the image and draw a rectangle around the facial area
img1 = cv2.imread(img1_path)
img1 = cv2.rectangle(img1, (img1_facial_area['x'], img1_facial_area['y']),
                     (img1_facial_area['x'] + img1_facial_area['w'], img1_facial_area['y'] + img1_facial_area['h']),
                     (0, 0, 255), 2)
img2 = cv2.imread(img2_path)
img2 = cv2.rectangle(img2, (img2_facial_area['x'], img2_facial_area['y']),
                     (img2_facial_area['x'] + img2_facial_area['w'], img2_facial_area['y'] + img2_facial_area['h']),
                     (0, 0, 255), 2)
# Display the image with the rectangle
cv2.imshow("Image 1", img1)
cv2.waitKey(0)
cv2.imshow("Image 2", img2)
cv2.waitKey(0)


