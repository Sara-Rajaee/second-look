import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks import extract_choice  # noqa: E402


class ExtractChoiceTest(unittest.TestCase):
    def test_answer_that_opens_with_the_letter(self):
        for text, letter in (("B", "B"), ("b", "B"), (" (C) ", "C"), ("**D**", "D"), ("A.", "A"),
                             ("C) A stick", "C"), ("D) Weniger als 20 cm", "D"), ("A, B, C", "A")):
            self.assertEqual(extract_choice(text), letter, text)

    def test_answer_after_an_explanation(self):
        self.assertEqual(extract_choice("To find the mean, we add the values. The answer is B."), "B")
        self.assertEqual(extract_choice("Let's use algebra: L = 2S, so the correct option is **D**"), "D")
        self.assertEqual(extract_choice("A stick is the usual tool, so the answer is C"), "C")

    def test_no_choice(self):
        self.assertEqual(extract_choice("I do not know"), "")
        self.assertEqual(extract_choice(""), "")
        self.assertEqual(extract_choice(None), "")


if __name__ == "__main__":
    unittest.main()
