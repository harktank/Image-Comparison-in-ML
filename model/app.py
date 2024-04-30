import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
from flask import Flask, request, jsonify
import asyncio
from playwright.sync_api import sync_playwright
import tempfile
import urllib.parse
import json


def load_image(file, target_size=(224, 224)):
    img = Image.open(file)
    # Convert image to RGB mode if it has an alpha channel
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Function to compare images with content
def compare_images_with_content(original_paths, duplicate_image):
    # Load pre-trained ResNet50 model
    try:
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    except Exception as e:
        print("Error occurred while loading ResNet50 model:", str(e))
        return None

    similarities = []
    for original_path in original_paths:
        try:
            # Load original image
            original_image = load_image(original_path)

            # Get feature vectors for images using the pre-trained model
            original_features = model.predict(original_image)
            duplicate_features = model.predict(duplicate_image)

            # Compute cosine similarity between feature vectors
            similarity_score = cosine_similarity(original_features, duplicate_features)[
                0
            ][0]
            image_name = os.path.basename(original_path)
            similarities.append(
                (image_name, similarity_score * 100)
            )  # Convert similarity score to percentage
        except Exception as e:
            print(f"Error occurred while processing image {original_path}: {str(e)}")

    return similarities


# Function to scan for similarity with content
def scan_for_similarity_with_content(originals_dir, duplicate_image):
    try:
        # List all image files in the originals directory
        original_files = [
            os.path.join(originals_dir, file)
            for file in os.listdir(originals_dir)
            if file.endswith((".jpg", ".jpeg", ".png"))
        ]

        # Compute similarity score for all images in the directory
        similarity_scores = compare_images_with_content(original_files, duplicate_image)

        return similarity_scores
    except Exception as e:
        print(f"Error occurred while scanning for similarity: {str(e)}")
        return None


# Streamlit app
# def main():
#     st.title("Website Similarity Analysis")

#     # Upload the image for comparison
#     uploaded_image = st.file_uploader("Upload an image for comparison", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         # Display the uploaded image
#         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#         # Button to trigger similarity comparison
#         if st.button("Compare with Originals"):
#             originals_directory = r'../screenshots/screenshots'  # Original images directory path

#             if os.path.exists(originals_directory):
#                 # Load and preprocess the uploaded image
#                 duplicate_image = load_image(uploaded_image)

#                 # Compute similarity score with images in the directory
#                 similarity_scores = scan_for_similarity_with_content(originals_directory, duplicate_image)
#                 if similarity_scores is not None:
#                     if len(similarity_scores) > 0:
#                         similarity_scores.sort(key=lambda x: x[1], reverse=True)  # Sort similarity scores in descending order
#                         highest_similarity = similarity_scores[0]
#                         st.write(f"The highest similarity score is: {highest_similarity[1]:.2f}% with image: {highest_similarity[0]}")
#                     else:
#                         st.write("No similar images found.")
#             else:
#                 st.warning("The specified directory does not exist.")


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_filtered(string_list, target_string, n):
    filtered_elements = {}
    for string in string_list:
        distance = levenshtein_distance(string, target_string)
        if distance < n:
            filtered_elements[string] = distance
    return filtered_elements


def main():
    app = Flask(__name__)
    sites_list = json.load(open("../screenshots/ss.json", "r"))

    @app.route("/analyze", methods=["POST"])
    def analyze():
        url = request.args.get("url")
        domain = urllib.parse.urlparse(url).netloc
        screenshot = tempfile.TemporaryFile()

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            html = page.content()
            screenshot_bytes = page.screenshot()
            screenshot.write(screenshot_bytes)

            loaded_image = load_image(screenshot)
            originals_directory = r"../screenshots/screenshots"
            similarity_scores = scan_for_similarity_with_content(
                originals_directory, loaded_image
            )
            scores = {}
            for site_path, similarity_score in similarity_scores:
                site_name = sites_list.get(os.path.basename(site_path))
                if site_name:
                    scores[site_name] = similarity_score

            # Calculate Levenshtein distances for domain names
            distances = levenshtein_filtered(scores.keys(), domain, 5)

            most_matching_site = None
            reason = "NONE"  # Default reason

            # Select site based on conditions
            for site_name, similarity_score in scores.items():
                if similarity_score > 90:
                    most_matching_site = site_name
                    reason = "IMG"
                    return jsonify(
                        {
                            "domain": domain,
                            "most_matching_site": most_matching_site,
                            "reason": reason,
                            #  "html": html,
                            "image_scores": scores,
                            "distances": distances
                        }
                    )
                elif similarity_score > 70 and distances.get(site_name, float('inf')) < 5:
                    most_matching_site = site_name
                    reason = "LEV_IMG"
                    return jsonify(
                        {
                            "domain": domain,
                            "most_matching_site": most_matching_site,
                            "reason": reason,
                            #  "html": html,
                            "image_scores": scores,
                            "distances": distances
                        }
                    )

        return jsonify(
            {
                "domain": domain,
                "most_matching_site": most_matching_site,
                "reason": reason,
                #  "html": html,
                "image_scores": scores,
                "distances": distances
            }
        )

    app.run()


if __name__ == "__main__":
    main()
