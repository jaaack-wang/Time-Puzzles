def generate_prompt(puzzle_sample):
    """
    Generates a natural language prompt based on the puzzle sample.

    Args:
        puzzle_sample (dict): A dictionary containing the puzzle data.
                              Expected to have a 'constraints' key (list of strings).

    Returns:
        str: The formatted prompt string.
    """
    instruction = (
        "From the constraints below, determine all valid dates (if any) that satisfy them. "
        "Depending on the conditions, the result may be zero, one, or multiple dates. "
        "Unless otherwise stated, all time-related constraints follow the Gregorian calendar."
    )

    constraints = puzzle_sample["constraints"]

    # Join constraints with newlines
    constraints_text = "\n".join(constraints)

    # Append the global constraint
    year_range = puzzle_sample["year_range"]
    global_constraint = (
        f"The year is within the range {year_range[0]} to {year_range[1]}."
    )

    return f"{instruction}\n\n{constraints_text}\n{global_constraint}"


def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
