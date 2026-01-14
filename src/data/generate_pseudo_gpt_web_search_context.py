import logging
import argparse
import pandas as pd
from tqdm import tqdm
from time import sleep
from pathlib import Path
from ..common.web_search import extract_text_from_url
from ..eval.response_metadata_parser import get_cited_urls_from_response_detailed


tqdm.pandas()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)


def extract_text_from_urls(urls):
    texts = []
    for url in urls:
        texts.append(extract_text_from_url(url))
        sleep(0.5) 
    return texts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo GPT web search context based on urls cited in openai response metadata."
    )
    parser.add_argument(
        "input_fp",
        type=str,
        help="Path to the input JSON file.",
    )

    args = parser.parse_args()
    return args



def create_context_from_extracted_texts(texts):

    errors = ["Error extracting content:", "WebParserClient error:", 
            "Error extracting content:", "WebParserClient error:", 
            "HTTP error occurred:", "Error: Connection error occurred", 
            "Error: Request timed out after 20 seconds", "Unexpected error:"]
    valid_texts = [text for text in texts if not any(
        error in text for error in errors) and len(text.strip()) > 0]
    output_text = ""

    for i, text in enumerate(valid_texts):
        output_text += f"======== Extracted Web Search Result#{i+1} ========\n\n{text}\n\n"
    return output_text.strip()


def main():
    args = parse_args()
    input_fp = Path(args.input_fp)
    output_fp = args.input_fp.replace(".json", "_cited_web_search_context.json")

    logging.info(f"Loading data from {input_fp}...")
    df = pd.read_json(input_fp)

    assert "response_detailed" in df.columns, \
        "The input file must contain a 'response_detailed' column."

    logging.info("Extracting cited URLs from response metadata...")
    df["urls_cited"] = df["response_detailed"].apply(
        get_cited_urls_from_response_detailed
    )

    logging.info("Extracting texts from cited URLs...")
    df["texts_from_cited_urls"] = df["urls_cited"].progress_apply(
        extract_text_from_urls
    )

    logging.info(f"Saving updated data with web search context to {output_fp}...")
    df["web_search_context"] = df["texts_from_cited_urls"].apply(
        create_context_from_extracted_texts
    )
    df = df[[
        "identifier", "prompt", "constraints", "solutions", "solution_count",
        "urls_cited", "web_search_context"
    ]]
    df.to_json(output_fp, orient="records", lines=False, indent=4)

    logging.info("Done.")


if __name__ == "__main__":
    main()
