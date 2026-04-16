import cv2

def blur_score(image_path):
    """
    Compute blur score using Laplacian variance.
    Lower value = more blurry
    """
    img = cv2.imread(image_path, 0)  # grayscale
    if img is None:
        raise ValueError("Invalid image path")

    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return score


def is_blurry(score, threshold=100):
    return score < threshold