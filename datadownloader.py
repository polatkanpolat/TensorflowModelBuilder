from icrawler.builtin import GoogleImageCrawler
from utils import createDirectoryIfNotExists
import argparse
import os


class ImageDownloader:
    """
    This class uses icrawler library and lets you to download
    images from google
    """

    def __init__(self, output_directory="data") -> None:
        self.output_directory = output_directory
        createDirectoryIfNotExists(self.output_directory)

    def downloadByName(self, name, number=100):
        download_path = os.path.join(self.output_directory, name)
        if os.path.exists(download_path):
            print(name, "Directory already exists")
            if len(os.listdir(download_path)) >= 100:
                print(name, "Directory already contains at least 100 images")
                return
        else:
            createDirectoryIfNotExists(download_path)
        google_crawler = GoogleImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": download_path},
        )
        google_crawler.crawl(keyword=name, max_num=number, file_idx_offset=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="downloads some images from google")
    parser.add_argument(
        "--number", type=int, help="number of images to download", default=20
    )
    parser.add_argument("name", help="description of images to download")
    args = parser.parse_args()
    downloader = ImageDownloader()
    downloader.downloadByName(args.name, args.number)
