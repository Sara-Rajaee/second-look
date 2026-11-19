import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks import (  # noqa: E402
    polymath_answer_readings,
    polymath_boxed_contents,
    polymath_clean,
    polymath_is_correct,
)


class BoxedContentsTest(unittest.TestCase):
    def test_braces_are_matched(self):
        self.assertEqual(polymath_boxed_contents(r"so $\boxed{\frac{1}{2}}$."), [r"\frac{1}{2}"])

    def test_all_boxes_in_order(self):
        self.assertEqual(polymath_boxed_contents(r"\boxed{3} then \boxed {7}"), ["3", "7"])

    def test_unclosed_box_is_ignored(self):
        self.assertEqual(polymath_boxed_contents(r"\boxed{7} and then \boxed{12"), ["7"])


class CleanTest(unittest.TestCase):
    def test_dollar_delimiters(self):
        self.assertEqual(polymath_clean(r"$2\sqrt{2}$"), r"2\sqrt{2}")
        self.assertEqual(polymath_clean(r"$$ 2n $$"), "2n")

    def test_presentation_commands(self):
        self.assertEqual(polymath_clean(r"\displaystyle \dfrac{1}{2}"), r"\frac{1}{2}")
        self.assertEqual(polymath_clean(r"\left(3, 6\right]"), "(3,6]")
        self.assertEqual(polymath_clean(r"25 \text{ cm}"), "25cm")

    def test_arrows_are_not_delimiters(self):
        self.assertEqual(polymath_clean(r"a \rightarrow b"), r"a\rightarrowb")

    def test_double_braces(self):
        self.assertEqual(polymath_clean("{42}"), "42")

    def test_spacing_commands_and_escaped_braces(self):
        self.assertEqual(polymath_clean(r"(3,\ 6]"), "(3,6]")
        self.assertEqual(polymath_clean(r"\{-4,\ -2,\, -1,~1\}"), "-4,-2,-1,1")
        self.assertEqual(polymath_clean("${-4,-2,-1,1}$"), "-4,-2,-1,1")
        self.assertEqual(polymath_clean(r"\left\{ \frac{1}{x},\ e^{2x} \right\}"), r"{\frac{1}{x},e^{2x}}")


class ReadingsTest(unittest.TestCase):
    def test_literal_reading_comes_first(self):
        self.assertEqual(polymath_answer_readings("0.125")[0], "0.125")
        self.assertEqual(polymath_answer_readings("1,2")[0], "1,2")

    def test_unit_is_removed(self):
        for boxed, number in ((r"25 \text{ cm}", "25"), ("5 个", "5"), ("70美元", "70"), ("57.500 €", "57500"),
                              (r"72\,\text{Inch}^3", "72"), (r"60 \text{ US-Dollar}", "60"),
                              (r"20,00 \text{ US-Dollar}", "20"), (r"10 \text{ ميل/ساعة}", "10"),
                              (r"45^\circ", "45"), ("12 km", "12"), (r"$ 5 \text{ hours} $", "5")):
            self.assertIn(number, polymath_answer_readings(boxed), boxed)

    def test_mathematics_is_not_a_unit(self):
        for boxed in (r"2\sqrt{2}", r"\frac{1}{2}", "3/4", "2^{10}", "5!", "2n+1", "10^5", "2n", "-2i", "3x",
                      r"36\frac{4}{11}", "9 + 16 = 25", r"334 \pm 25 \text{ cm}"):
            self.assertEqual(polymath_answer_readings(boxed), [polymath_clean(boxed)], boxed)

    def test_unit_scan_is_linear(self):
        polymath_answer_readings("5 " + "دقيقة(" * 20000)      # would hang with a backtracking regex

    def test_number_formats(self):
        self.assertIn("1250", polymath_answer_readings(r"\$1,250.00"))
        self.assertIn("12", polymath_answer_readings(r"12\%"))
        self.assertIn("26.5", polymath_answer_readings("26,5"))
        self.assertIn("57", polymath_answer_readings("57{,}00"))
        self.assertIn("75", polymath_answer_readings("75,00$"))


class IsCorrectTest(unittest.TestCase):
    def check(self, response, gold, expected):
        self.assertEqual(polymath_is_correct(response, gold), expected, f"{response!r} vs {gold!r}")

    def test_gold_wrapped_in_dollars(self):
        self.check(r"The answer is $\boxed{2\sqrt{2}}$.", r"$2\sqrt{2}$", True)
        self.check(r"\boxed{2n}", "$2n$", True)
        self.check(r"\boxed{(3, 6]}", "$(3,6]$", True)
        self.check(r"\boxed{\dfrac{1}{2}}", r"$\frac{1}{2}$", True)
        self.check(r"\boxed{\{-4, -2, -1, 1\}}", "${-4,-2,-1,1}$", True)
        self.check(r"\boxed{13/24}", r"$\frac{13}{24}$", True)
        self.check(r"\boxed{-1+\sqrt{2022}}", r"$\sqrt{2022} - 1$", True)
        self.check(r"\boxed{(n-1)m}", "$m(n-1)$", True)

    def test_expression_is_not_cut_at_its_leading_number(self):
        self.check(r"\boxed{2\sqrt{2}}", "2", False)
        self.check(r"\boxed{1/2}", "1", False)
        self.check(r"\boxed{10^5}", "10", False)
        self.check(r"\boxed{-2i}", "-2", False)
        self.check(r"\boxed{2n}", "2", False)

    def test_large_numbers_are_not_compared_as_floats(self):
        self.check(r"\boxed{2^{2022}}", "$2^{2023}$", False)
        self.check(r"\boxed{3^{2017}}", r"$\frac{3^{2017}-2019}{2}$", False)
        self.check(r"\boxed{2^{99}}", "$2^{99}-1$", False)
        self.check(r"\boxed{2^{2023}}", "$2^{2023}$", True)
        self.check(r"\boxed{4^{-2016}}", r"$\frac{1}{4^{2016}}$", True)
        self.check(r"\boxed{0.866}", r"$\frac{\sqrt{3}}{2}$", True)

    def test_last_box_is_the_answer(self):
        response = r"First, \boxed{3} is the number of roots. Hence the sum is \boxed{7}."
        self.check(response, "7", True)
        self.check(response, "3", False)

    def test_units_and_number_formats(self):
        self.check(r"\boxed{25 \text{ cm}}", "25", True)
        self.check(r"\boxed{20,00 \text{ US-Dollar}}", "20", True)
        self.check(r"\boxed{1,250}", "1250", True)
        self.check(r"\boxed{28.00}", "28", True)
        self.check(r"\boxed{0.125}", "0.125", True)
        self.check(r"\boxed{{42}}", "42", True)

    def test_wrong_and_missing_answers(self):
        self.check(r"\boxed{41}", "42", False)
        self.check(r"\boxed{\frac{1}{2}}", "12", False)
        self.check("The answer is 42.", "42", False)
        self.check(r"\boxed{42", "42", False)
        self.check("", "42", False)
        self.check(None, "42", False)


if __name__ == "__main__":
    unittest.main()
