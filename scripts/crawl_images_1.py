"""
crawl_images.py — Web Image Crawler for Forestry Dataset
=========================================================
Usage:
    python crawl_images.py --per_class 2000 --output ./dataset/raw

Dependencies:
    pip install icrawler
"""

import os
import argparse
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

CLASSES = {
    'healthy_forest' : 'healthy green forest dense canopy aerial view',
    'deforested'     : 'deforestation cleared land logging aerial view',
    'forest_fire'    : 'forest fire burning wildfire smoke aerial',
    'flooded_forest' : 'flooded forest waterlogged trees aerial view',
    'diseased_forest': 'diseased dead trees forest blight bark beetle',
}

def crawl(class_name, query, num_images, output_dir, engine='bing'):
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n🔍 [{class_name}] — "{query}" ({num_images} images)')
    try:
        if engine == 'google':
            crawler = GoogleImageCrawler(
                feeder_threads=2, parser_threads=2, downloader_threads=4,
                storage={'root_dir': output_dir}
            )
        else:
            crawler = BingImageCrawler(
                feeder_threads=2, parser_threads=2, downloader_threads=4,
                storage={'root_dir': output_dir}
            )
        crawler.crawl(keyword=query, max_num=num_images,
                      min_size=(150, 150), file_idx_offset=0)
        count = len(os.listdir(output_dir))
        print(f'   ✅ {count} images downloaded')
    except Exception as e:
        print(f'   ❌ Error: {e}')

def main():
    parser = argparse.ArgumentParser(description='Crawl forestry images')
    parser.add_argument('--per_class', type=int, default=200, help='Images per class')
    parser.add_argument('--output',    type=str, default='./dataset/raw', help='Output directory')
    parser.add_argument('--engine',    type=str, default='bing', choices=['bing', 'google'])
    args = parser.parse_args()

    print(f'🌲 Forestry Image Crawler')
    print(f'   Engine    : {args.engine}')
    print(f'   Per class : {args.per_class}')
    print(f'   Classes   : {len(CLASSES)}')
    print(f'   Total     : ~{args.per_class * len(CLASSES)} images')

    for class_name, query in CLASSES.items():
        out = os.path.join(args.output, class_name)
        crawl(class_name, query, args.per_class, out, args.engine)

    print('\n✅ Crawling complete!')

if __name__ == '__main__':
    main()
