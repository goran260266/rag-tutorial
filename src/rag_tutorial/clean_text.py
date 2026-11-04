import re
from collections.abc import Callable

import ftfy
import requests
import trafilatura
from cleantext import clean
from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace.

    Args:
        text: Raw input text, possibly containing HTML markup.

    Returns:
        The text without tags, with collapsed spaces and at most one blank line
        between paragraphs, stripped of leading and trailing whitespace.
    """
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\S\n]+", " ", text)  # Replace multiple whitespace tabs with single space
    text = re.sub(r"\n +", "\n", text)  # Remove trailing whitespace from lines
    text = re.sub(r" +\n", "\n", text)  # Remove leading whitespace from lines
    text = re.sub(r"\n{3,}", "\n\n", text)  # Replace multiple newlines with double newlines
    return text.strip()  # Remove leading and trailing whitespace


def clean_text_extended(text: str) -> str:
    """Clean text with `clean_text` and additionally unwrap lines and unify quotes.

    Args:
        text: Raw input text.

    Returns:
        The cleaned text with single newlines replaced by spaces (paragraph breaks
        are kept) and typographic quotation marks replaced by a straight single quote.
    """
    text = clean_text(text)
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)  # Replace single newlines with space
    text = re.sub(r'[""«»„‟`"]', "'", text)  # Replace various quotation marks with single quote
    text = re.sub(r" +", " ", text)  # Replace multiple spaces with single space
    return text


def thorough_clean(text: str) -> str:
    """Clean text using ftfy, `clean_text_extended` and the cleantext library.

    Args:
        text: Raw input text, possibly with encoding errors, URLs, emails or emoji.

    Returns:
        The cleaned text with fixed encoding, URLs replaced by `<URL>`, emails
        replaced by `<EMAIL>` and emoji removed.
    """
    text = ftfy.fix_text(text)
    text = clean_text_extended(text)
    text = clean(
        text,
        no_urls=True,
        no_emails=True,
        no_emoji=True,
        to_ascii=False,
        lower=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
    )
    return text


def clean_documents(
    docs: list[Document],
    clean_fn: Callable[[str], str] = clean_text_extended,
) -> list[Document]:
    """Apply a cleaning function to the content of each document in place.

    Args:
        docs: Documents whose `page_content` should be cleaned.
        clean_fn: Function that takes a string and returns the cleaned string.

    Returns:
        The same list of documents with cleaned `page_content`.
    """
    for doc in docs:
        doc.page_content = clean_fn(doc.page_content)
    return docs


def print_text(
    text: str,
    snippet_length: int = 200,
    keyword: str = "Inhaltsverzeichnis",
) -> None:
    """Print a snippet of the text, starting at the first occurrence of a keyword.

    Args:
        text: Text to take the snippet from.
        snippet_length: Number of characters to print.
        keyword: Marker where the snippet starts. If it is not found, the snippet
            starts at the beginning of the text.
    """
    start_idx = text.find(keyword)
    if start_idx == -1:
        print(text[:snippet_length])
        return
    print(text[start_idx : start_idx + snippet_length])


def extract_text_from_url(url: str) -> str | None:
    """Fetch a page and extract its main text with trafilatura.

    Args:
        url: Address of the page to fetch.

    Returns:
        The extracted text, or None if trafilatura finds no text content.

    Raises:
        requests.HTTPError: If the server responds with an error status.
    """
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
    )
    response.raise_for_status()
    html = response.text
    return trafilatura.extract(html, include_tables=False)


if __name__ == "__main__":
    # Example with html text
    loader = WebBaseLoader("https://de.wikipedia.org/wiki/Python_(Programmiersprache)")
    docs = loader.load()
    docs = clean_documents(docs)
    print_text(docs[0].page_content)

    # Example with dirty_text
    dirty_text = """
       Machine Learning ist ein Teilgebiet der KI.
    \t\tEs lernt automatisch aus Daten.<br>

          Neurale Netzwerke   sind     eine <em>wichtige</em> Methode.
    <p>Sie sind
       vom Gehirn
       `inspiriert'.</p>

       „Large Language Models‟    verwenden sehr große Netzwerke.
    """

    cleaned = thorough_clean(dirty_text)
    print(f"cleaned: {cleaned}")

    # example with trafilatura
    text = extract_text_from_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
    if text is None:
        print("trafilatura: Keine Textinhalte extrahiert.")
    else:
        print(f"trafilatura: {text}")

