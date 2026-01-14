import os
import uuid
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from IPython.display import display, HTML


def pretty_print(df: pd.DataFrame):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def read_json(fp: str) -> pd.DataFrame:
    if not os.path.exists(fp):
        logging.info(f"File {fp} does not exist. Returning empty DataFrame.")
        return pd.DataFrame()

    return pd.read_json(fp)


def save_dict_as_json(dic: dict, fp: str, 
                      indent: Optional[int]=4, 
                      print_message=True) -> None:
    
    os.makedirs(Path(fp).parent.absolute(), exist_ok=True)
    
    with open(fp, "w") as f:
        json.dump(dic, f, indent=indent)

        if print_message:
            logging.info(f"{fp} saved!")


def generate_unique_id() -> str:
    return str(uuid.uuid4())