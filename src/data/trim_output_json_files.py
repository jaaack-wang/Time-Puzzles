import os
import glob
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Union, Iterable
from ..eval.response_metadata_parser import append_or_load_json_output_fp_with_token_usage


logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



def trim_output_json_file(input_fp: str, 
                          cols_to_drop: Union[str, Iterable[str]], 
                          overwrite: bool = False) -> None:
    '''The input file is a JSON file, not saved in lines format.
    The trimmed output file will be saved as "trimmed_{input_fp}".'''
    output_fp = f"trimmed_{input_fp}"

    if Path(output_fp).exists() and not overwrite:
        print(f"Trimmed file {output_fp} already exists. Skipping.")
        return
    else:
        os.makedirs(Path(output_fp).parent, exist_ok=True)
    
    df = pd.read_json(input_fp)
    if isinstance(cols_to_drop, str):
        cols_to_drop = [cols_to_drop]
    
    cols_to_drop_ = []
    for col in cols_to_drop:
        if col not in df.columns:
            logging.warning(f"[Output Trimming] Column {col} not found in {input_fp}. Skipping drop for this column.")
        else:
            cols_to_drop_.append(col)

    cols_to_drop = cols_to_drop_
    if not cols_to_drop:
        logging.info(f"[Output Trimming] No columns to drop from {input_fp}. Skipping trimming.")
        return

    df.drop(cols_to_drop, axis=1, inplace=True)
    logging.info(f"[Output Trimming] Dropped columns {cols_to_drop} from {input_fp}.")
    df.to_json(output_fp, orient="records", lines=False, indent=4)
    logging.info(f"[Output Trimmed] Trimmed file saved to {output_fp}.")


def main():
    parser = argparse.ArgumentParser(
        description=("Trim output JSON files by dropping specified columns."
                     "The trimmed files will be saved with prefix 'trimmed_'."))
    parser.add_argument(
        "--input_dire", type=str, default="outputs",
        help="Path to the input directory containing JSON files to be trimmed.")
    
    parser.add_argument(
        "--cols_to_drop", type=str, nargs='+', default=["response_detailed"],
        help="Column names to drop from the JSON files.")

    parser.add_argument(
        "--add_token_usage", action='store_true',
        help="Whether to add token usage columns before trimming."
    )

    parser.add_argument(
        "--overwrite", action='store_true',
        help="Whether to overwrite existing trimmed files.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    json_files = glob.glob(f"{args.input_dire}/**/*.json", recursive=True)
    logging.info(f"Found {len(json_files)} JSON files in {args.input_dire} to trim.")

    for json_fp in json_files:

        try:
            logging.info(f"Processing file: {json_fp}")
            if args.add_token_usage:
                append_or_load_json_output_fp_with_token_usage(json_fp)
            
            trim_output_json_file(json_fp, args.cols_to_drop, overwrite=args.overwrite)
        except Exception as e:
            logging.error(f"Failed to process {json_fp}: {e}")
    
    logging.info("All files trimmed.")
    

if __name__ == "__main__":
    main()