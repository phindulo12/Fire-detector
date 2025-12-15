import cv2

def loAd_fiLe(path):
    try:
        img = cv2.imread(path)

        if img is None:
            print("imread failed. Check path:", path)

        return img

    except Exception as e:
        print("Error:", e)
        return None
