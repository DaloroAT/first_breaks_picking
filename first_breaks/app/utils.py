import base64
import re
from pathlib import Path
from typing import Union


def markdown_images(markdown: str):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path: Path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path: Path, img_alt: str):
    img_format = Path(img_path).suffix
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown: str) -> str:
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if Path(image_path).exists():
            markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
    return markdown


def read_markdown(fname: Union[Path, str]) -> str:
    with open(fname) as f:
        intro = f.read()
    return markdown_insert_images(intro)
