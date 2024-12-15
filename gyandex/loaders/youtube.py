import xml.etree.ElementTree as ET
from urllib.parse import unquote_plus

import requests
import yt_dlp

from .types import Document


def fetch_youtube(url: str) -> Document:
    """
    Fetch a YouTube video transcript
    :param url:
    :return:
    """
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "srv1",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError("Could not extract video information")

        r = requests.get(info["requested_subtitles"]["en"]["url"])

        # Parse XML
        root = ET.fromstring(r.text)
        # Extract and decode text from each <text> element
        texts = [unquote_plus(text.text or "") for text in root.findall("text")]

        # Join all texts with space
        return Document(
            title=info.get("title", ""),
            content=" ".join(texts),
            metadata={
                "url": url,
                "description": info.get("description", ""),
            },
        )
