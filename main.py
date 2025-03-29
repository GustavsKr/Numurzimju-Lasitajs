import cv2
import torch
import requests
import numpy as np
import re
from collections import Counter
from ultralytics import YOLO
import easyocr

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    improved_gray = clahe.apply(gray)
    improved_gray = cv2.bilateralFilter(improved_gray, 9, 75, 75)
    kernel = np.ones((3,3), np.uint8)
    improved_gray = cv2.morphologyEx(improved_gray, cv2.MORPH_OPEN, kernel)

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    improved_gray = cv2.LUT(improved_gray, table)

    return improved_gray

def extract_plate_text(image, reader):
    results = reader.readtext(image, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", decoder='beamsearch')
    if not results:
        return None, 0.0

    sorted_result = sorted(results, key=lambda x: x[0][0][0])
    text = "".join([re.sub(r'[^a-zA-Z0-9.]', '', text[1]).strip() for text in sorted_result]).upper()
    confidence = max([r[2] for r in results])
    return text, confidence

def format_plate(plate_text):
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    if plate_text.startswith("LV") and not plate_text[2:].isdigit():
        plate_text = plate_text[2:]

    if plate_text.isalpha() and 2 < len(plate_text) < 9:
        return plate_text
    elif 2 < len(plate_text) < 7:
        formatted_plate = "".join([dict_int_to_char.get(c, c) for c in plate_text[:2]])
        formatted_plate += "".join([dict_char_to_int.get(c, c) for c in plate_text[2:]])
        if plate_text[:2].isalpha() and plate_text[2:].isdigit() and 1 <= len(plate_text[2:]) <= 4:
            return formatted_plate
    return None

def license_plate_reader(image_urls, model, reader):
    plate_candidates = []
    confidence_scores = {}

    for url in image_urls:
        try:
            response = requests.get(url.strip(), stream=True)
            if response.status_code != 200:
                continue

            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue

            results = model(img, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()

            plates = [boxes[i] for i, conf in enumerate(confidences) if conf > 0.5]
            if not plates:
                continue

            x1, y1, x2, y2 = map(int, plates[0][:4])
            cropped_plate = img[y1:y2, x1:x2]

            preprocessed = preprocess_image(cropped_plate)

            # cv2.imshow("plate", preprocessed)  # Uncomment for debugging
            # cv2.waitKey(0)

            plate_text, conf = extract_plate_text(preprocessed, reader)

            if plate_text:
                formatted = format_plate(plate_text)
                if formatted:
                    plate_candidates.append(formatted)
                    confidence_scores[formatted] = confidence_scores.get(formatted, []) + [conf]

        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    if not plate_candidates:
        print("❌ NO PLATE CANDIDATES")
        return None

    plate_counts = Counter(plate_candidates)
    most_common = plate_counts.most_common()
    max_count = most_common[0][1]
    top_plates = [p for p, c in most_common if c == max_count]

    best_plate = max(top_plates, key=lambda p: sum(confidence_scores[p])) if len(top_plates) > 1 else top_plates[0]

    print(f"✅ Final detected plate: {best_plate}")
    return best_plate

def main():
    torch.set_printoptions(profile="error")
    model = YOLO("license_plate_detector.pt")
    reader = easyocr.Reader(['ru'])  # Russian reader works better than english in this case

    # Multiple images of the same car improves the result accuracy
    image_urls = [
        "https://i.ss.com/gallery/7/1335/333630/audi-80-66725904.800.jpg",
        "https://i.ss.com/gallery/7/1335/333630/audi-80-66725905.800.jpg",
        "https://i.ss.com/gallery/7/1335/333630/audi-80-66725906.800.jpg"
    ]

    plate = license_plate_reader(image_urls, model, reader)

    if not plate:
        print("❌ No valid plate detected.")

if __name__ == "__main__":
    main()
