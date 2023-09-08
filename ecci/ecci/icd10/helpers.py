import re
from typing import List


def list_to_regex(codes_list: List[str]):
    return r"|".join([rf"(?:{re.escape(code)})" for code in codes_list])
