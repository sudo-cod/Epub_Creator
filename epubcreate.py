import json
import os
from ebooklib import epub

# Define the path to the JSON files
json_dir = "/Users/helga/PycharmProjects/epub_creator"

# Initialize the EPUB book
book = epub.EpubBook()

# Set metadata
book.set_title("My Scraped Novel")
book.set_language("en")
book.add_author("Unknown Author")

# Create a list to hold chapters
chapters = []

# Loop through JSON files in the directory
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(json_dir, filename)

        # Load JSON data
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create a chapter for each JSON file
        chapter_title = data["title"]
        chapter_content = data["content"]

        # Replace \r\n or \n with <br> tags for line breaks
        formatted_content = chapter_content.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")

        # Wrap the content in <p> tags for paragraphs
        formatted_content = f"<p>{formatted_content}</p>"

        # Create an EPUB chapter
        chapter = epub.EpubHtml(
            title=chapter_title,
            file_name=f"{chapter_title}.xhtml",
            lang="en"
        )
        chapter.content = f"<h1>{chapter_title}</h1>{formatted_content}"

        # Add the chapter to the book
        book.add_item(chapter)
        chapters.append(chapter)

# Define the table of contents
book.toc = tuple(chapters)

# Add default NCX and Nav files
book.add_item(epub.EpubNcx())
book.add_item(epub.EpubNav())

# Define the book spine (order of content)
book.spine = ["nav"] + chapters

# Save the EPUB file
epub.write_epub("MyScrapedNovel.epub", book, {})
print("EPUB file created successfully!")
