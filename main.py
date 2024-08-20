from deepface import DeepFace
import os
import shutil

backends = ['opencv', 'retinaface', 'ssd', 'mtcnn']
models = ["Facenet", "Facenet512", "OpenFace", "DeepID", "SFace"]
test_folder_path = "Test"
database_path = "Database"

for folder_path, _, filenames in os.walk(database_path):
    for filename in filenames:
        # Get the full path of the file
        file_path = os.path.join(folder_path, filename)

        print("Processing file:", file_path)
        if os.path.isfile(file_path):

            for test_image_name in os.listdir(test_folder_path):
                test_image_path = os.path.join(test_folder_path, test_image_name)
                verified_count = 0
                print(test_image_name)
                for model in models:
                    print("Testing model:", model)
                    results = DeepFace.verify(file_path, test_image_path, model_name=model,
                                              detector_backend="retinaface")
                    if results['verified']:
                        verified_count += 1
                if verified_count > 1:
                    shutil.copy(test_image_path,folder_path)
