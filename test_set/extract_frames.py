import cv2
import os

# Video ve kayıt dizinleri
video_path = "/home/halil/Desktop/videoTask/test_set/video_50/video_left.avi"
saved_dir = "/home/halil/Desktop/videoTask/test_set/video_50/frames/"

# Klasör yoksa oluştur
os.makedirs(saved_dir, exist_ok=True)

# Video dosyasını aç
cam = cv2.VideoCapture(video_path)

currentframe = 0
skipped_frames = 0

# Frame'leri çıkar ve kaydet
while True:
    ret, frame = cam.read()
    if not ret:
        print(f"Bitti: currentframe={currentframe}, skipped_frames={skipped_frames}, ret={ret}")
        break

    if skipped_frames == 0:
        # Frame dosya adını oluştur
        frame_number_str = str(currentframe).zfill(9)
        filename = f"{saved_dir}_{frame_number_str}.jpg"
        print("Creating...", filename)

        # Kaydet
        cv2.imwrite(filename, frame)
        skipped_frames = 60

    skipped_frames -= 1
    currentframe += 1

# Kaynakları serbest bırak
cam.release()
cv2.destroyAllWindows()

