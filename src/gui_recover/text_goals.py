"""Visible goal grammar for the supported native MiniWoB text tasks."""

import re


NATIVE_TEXT_TASKS = ("enter-text", "enter-text-2", "enter-text-dynamic")
DIRECT_GOAL = re.compile(r'Enter "(.*)" into the text field and press Submit\.')
CASE_GOAL = re.compile(r'Type "([A-Za-z]+)" in all (upper|lower) case letters in the text input and press Submit\.')


def expected_text(goal):
    if match := DIRECT_GOAL.fullmatch(goal):
        return match[1]
    if match := CASE_GOAL.fullmatch(goal):
        word, case = match.groups()
        return word.upper() if case == "upper" else word.lower()
    return None
