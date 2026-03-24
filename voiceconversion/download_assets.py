import os
import urllib.request
import sys

BASE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

FILES_TO_DOWNLOAD = {
    "hubert_base.pt": "assets/hubert/hubert_base.pt",
    "rmvpe.pt": "assets/rmvpe/rmvpe.pt",
    "uvr5_weights/HP2_all_vocals.pth": "assets/uvr5/HP2_all_vocals.pth",
    "pretrained_v2/f0G40k.pth": "assets/pretrained/f0G40k.pth",
    "pretrained_v2/f0D40k.pth": "assets/pretrained/f0D40k.pth"
}

def progress_bar(count, block_size, total_size):
    if total_size == -1: return
    percent = min(100, int(count * block_size * 100 / total_size))
    sys.stdout.write(f"\r  -> Progress: [{('=' * (percent // 2)).ljust(50, ' ')}] {percent}%")
    sys.stdout.flush()

def main():
    print("Starting automated asset downloads for RVC Backend...\n")
    for url_path, local_path in FILES_TO_DOWNLOAD.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            print(f"Skipping {local_path} (Already exists)")
            continue
            
        download_url = BASE_URL + url_path
        print(f"Downloading {url_path}...")
        try:
            urllib.request.urlretrieve(download_url, local_path, reporthook=progress_bar)
            print("\n  -> Download complete!\n")
        except Exception as e:
            print(f"\n  -> Error downloading {url_path}: {e}\n")

    print("All base assets verified and downloaded successfully!")

if __name__ == "__main__":
    main()