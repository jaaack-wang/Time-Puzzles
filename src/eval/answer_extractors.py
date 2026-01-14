import re
import logging
from typing import Optional
from ..llms import get_llm_response_function


def rule_based_answer_extractor(response: str, 
                                loose_extraction: Optional[bool] = False, 
                                num_of_expected_ans: Optional[int] = None) -> str:
    '''Rule-based answer extractor for date extraction.

    Args:
        response: LLM response string
        loose_extraction: whether to use loose extraction (extract last N dates found). default is False.
                          when False, extract all and only dates found after "MY ANSWER: ".
        num_of_expected_ans: number of expected answers. default is None, extract all found answers.
                          only used when loose_extraction is True.
    
    Returns:
        extracted answers as a list of strings, or "CANNOT_PARSE" if extraction fails.
    '''
    if num_of_expected_ans is None:
        num_of_expected_ans = 0

    if num_of_expected_ans == 0:
        # turn off loose extraction if no answers are expected
        loose_extraction = False
    
    # first, extract anything that comes after "MY ANSWER: ", but not including "MY ANSWER: "
    # we also see cases where models say "My answer is"

    matches = re.findall(r"answer:.*", response, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"answer is.*", response, flags=re.IGNORECASE)

    response = matches if matches else [response]
    # extract None or YYYY-MM-DD from the last part

    if not loose_extraction:
        if len(matches) == 0:
            return "CANNOT_PARSE"
    
    response = response[-1]
    response = response.replace("‑", "-")

    if loose_extraction:
        ans = re.findall(r"(None|\d{4}-\d{2}-\d{2})", response, 
                         re.IGNORECASE)[-num_of_expected_ans:]
    else:
        ans = re.findall(r"(None|\d{4}-\d{2}-\d{2})", response)
    
    ans = [a.lower() for a in ans]
    if ans == ["none"]:
        return []
    
    if ans:
        return ans

    return "CANNOT_PARSE"


class LLMAnswerExtractor:
    def __init__(self, extractor_llm: Optional[str] = None, 
                 max_tries: Optional[int] = 3):
        
        if extractor_llm:
            self.extractor_llm = extractor_llm
        else:
            self.extractor_llm = "gpt-4o-mini-2024-07-18"
            logging.info(f"No extractor_llm specified. Using default: {self.extractor_llm}")
            
        self.get_response = get_llm_response_function(self.extractor_llm, 
                                                      experiment_condition="no_tool_use")
        self.max_tries = max_tries
        self.instruction = ("Given the following response to a date guessing question, "
                            "extract the final answer(s) in the format YYYY-MM-DD. "
                            "If no valid date exists, respond with 'None'. "
                            "If multiple valid dates exist, list them all separated by commas. "
                            "Do not include any other text besides the date(s) or 'None'.")

    def __call__(self, response: str, 
                 return_completion: Optional[bool] = False) -> str:
        
        success = False
        for i in range(self.max_tries):
            prompt = f"{self.instruction}\n\n{response}"
            
            try:
                _, completion = self.get_response(prompt)
                success = True
                break
            except Exception as e:
                completion = f"SOMETHING_WRONG: {e}"
                print(f"LLM call failed with error: {e}. Retrying... {i+1}/{self.max_tries}")
                continue
        
        if not success:
            ans = "CANNOT_PARSE"
        
        ans = re.findall(r"(None|\d{4}-\d{2}-\d{2})", completion, re.IGNORECASE)
        ans = [a.lower() for a in ans]
        
        if ans == ["none"]:
            ans = []

        if return_completion:
            return ans, completion
        
        return ans