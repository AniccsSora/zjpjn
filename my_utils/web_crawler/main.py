import requests
import os
from PIL import Image
from io import BytesIO
from imagehash import average_hash
import time
from bs4 import BeautifulSoup

def save_image(url, folder, filename):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save(os.path.join(folder, filename))

def download_images(query, folder, count):
    # 設定 Google 圖片搜尋的 URL
    url = 'https://www.google.com/search'
    params = {
        'q': query,
        'tbm': 'isch',
        'ijn': '0',
        'start': '0'
    }

    total_downloaded = 0
    hashes = set()  # 用於存儲已下載圖片的哈希值

    while total_downloaded < count:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.text

            soup = BeautifulSoup(data, 'html.parser')
            image_tags = soup.find_all('img')

            for image_tag in image_tags:
                thumbnail_url = image_tag['src']

                try:
                    # 透過連結進一步訪問原始圖片所在的頁面
                    response = requests.get(thumbnail_url)
                    response.raise_for_status()
                    image_page = response.text

                    start_index = image_page.find(',"ou":"') + len(',"ou":"')
                    end_index = image_page.find('","ow"')
                    image_url = image_page[start_index:end_index]

                    # 檢查圖片的哈希值，確保不下載重複的圖片
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                    image_hash = average_hash(image)

                    if image_hash in hashes:
                        continue

                    # 下載圖片並保存到指定資料夾
                    filename = f'{total_downloaded + 1}.jpg'
                    save_image(image_url, folder, filename)

                    hashes.add(image_hash)
                    total_downloaded += 1

                    if total_downloaded == count:
                        break

                    # 在下載完每張圖片後休息一段時間
                    time.sleep(0.3)  # 休息0.3秒

                except requests.exceptions.RequestException as e:
                    print(f'Error: {e}')
                    continue

            # 在翻頁之前休息一段時間
            time.sleep(0.2)  # 休息0.2秒

        except requests.exceptions.HTTPError as e:
            print(f'HTTP Error: {e}')
            break
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')
            break


if __name__ == '__main__':
    # 執行爬蟲
    query = 'qr code in life scene'
    folder = 'images'
    count = 2

    if not os.path.exists(folder):
        os.makedirs(folder)

    download_images(query, folder, count)
