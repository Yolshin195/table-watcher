import gdown
import os

# Ваши ссылки (gdown сам поймет, что это Google Drive)
VIDEO_URLS = [
    "https://drive.google.com/file/d/1pAPTjESoDgjqhTaqM_graYfMpWcOyzRe/view?usp=sharing",
    "https://drive.google.com/file/d/1rYmJB13vvV96JuDUrBvlEXtoKFPWo75A/view?usp=drive_link",
    "https://drive.google.com/file/d/1xfHTf3vJVlTXXs0Rdq9L_xi816ATX5zD/view?usp=sharing",
]

OUTPUT_DIR = "videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_videos():
    for i, url in enumerate(VIDEO_URLS, start=1):
        filename = f"video_{i}.mp4"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"--- Загрузка файла {i} из {len(VIDEO_URLS)} ---")
        
        # fuzzy=True позволяет скачивать по обычным ссылкам "view"
        # quiet=False показывает красивый индикатор прогресса (проценты)
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
    print("\nГотово! Все видео лежат в папке:", OUTPUT_DIR)

if __name__ == "__main__":
    download_videos()