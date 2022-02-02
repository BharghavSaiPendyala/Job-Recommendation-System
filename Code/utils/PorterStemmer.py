from nltk.stem.api import StemmerI

class PorterStemmer(StemmerI):
    NLTK_EXTENSIONS = "NLTK_EXTENSIONS"
    MARTIN_EXTENSIONS = "MARTIN_EXTENSIONS"
    ORIGINAL_ALGORITHM = "ORIGINAL_ALGORITHM"

    def __init__(self, mode=NLTK_EXTENSIONS):
        self.mode = mode
        irregular_forms = {
            "sky": ["sky", "skies"],
            "die": ["dying"],
            "lie": ["lying"],
            "tie": ["tying"],
            "news": ["news"],
            "inning": ["innings", "inning"],
            "outing": ["outings", "outing"],
            "canning": ["cannings", "canning"],
            "howe": ["howe"],
            "proceed": ["proceed"],
            "exceed": ["exceed"],
            "succeed": ["succeed"],
        }

        self.pool = {}
        for key in irregular_forms:
            for val in irregular_forms[key]:
                self.pool[val] = key

        self.vowels = frozenset(["a", "e", "i", "o", "u"])

    def _is_consonant(self, word, i):
        
        if word[i] in self.vowels:
            return False
        if word[i] == "y":
            if i == 0:
                return True
            else:
                return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem):
        
        cv_sequence = ""
        for i in range(len(stem)):
            if self._is_consonant(stem, i):
                cv_sequence += "c"
            else:
                cv_sequence += "v"
        return cv_sequence.count("vc")

    def _has_positive_measure(self, stem):
        return self._measure(stem) > 0

    def _contains_vowel(self, stem):
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _ends_double_consonant(self, word):
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and self._is_consonant(word, len(word) - 1)
        )

    def _ends_cvc(self, word):
        return (
            len(word) >= 3
            and self._is_consonant(word, len(word) - 3)
            and not self._is_consonant(word, len(word) - 2)
            and self._is_consonant(word, len(word) - 1)
            and word[-1] not in ("w", "x", "y")
        ) or (
            self.mode == self.NLTK_EXTENSIONS
            and len(word) == 2
            and not self._is_consonant(word, 0)
            and self._is_consonant(word, 1)
        )

    def _replace_suffix(self, word, suffix, replacement):
        assert word.endswith(suffix), "Given word doesn't end with given suffix"
        if suffix == "":
            return word + replacement
        else:
            return word[: -len(suffix)] + replacement

    def _apply_rule_list(self, word, rules):
        for rule in rules:
            suffix, replacement, condition = rule
            if suffix == "*d" and self._ends_double_consonant(word):
                stem = word[:-2]
                if condition is None or condition(stem):
                    return stem + replacement
                else:
                    
                    return word
            if word.endswith(suffix):
                stem = self._replace_suffix(word, suffix, "")
                if condition is None or condition(stem):
                    return stem + replacement
                else:
                    
                    return word

        return word

    def _step1a(self, word):
        if self.mode == self.NLTK_EXTENSIONS:
            if word.endswith("ies") and len(word) == 4:
                return self._replace_suffix(word, "ies", "ie")

        return self._apply_rule_list(
            word,
            [
                ("sses", "ss", None),  
                ("ies", "i", None),  
                ("ss", "ss", None),  
                ("s", "", None),  
            ],
        )

    def _step1b(self, word):
        if self.mode == self.NLTK_EXTENSIONS:
            if word.endswith("ied"):
                if len(word) == 4:
                    return self._replace_suffix(word, "ied", "ie")
                else:
                    return self._replace_suffix(word, "ied", "i")

        
        if word.endswith("eed"):
            stem = self._replace_suffix(word, "eed", "")
            if self._measure(stem) > 0:
                return stem + "ee"
            else:
                return word

        rule_2_or_3_succeeded = False

        for suffix in ["ed", "ing"]:
            if word.endswith(suffix):
                intermediate_stem = self._replace_suffix(word, suffix, "")
                if self._contains_vowel(intermediate_stem):
                    rule_2_or_3_succeeded = True
                    break

        if not rule_2_or_3_succeeded:
            return word

        return self._apply_rule_list(
            intermediate_stem,
            [
                ("at", "ate", None),  
                ("bl", "ble", None),  
                ("iz", "ize", None),  
                
                
                (
                    "*d",
                    intermediate_stem[-1],
                    lambda stem: intermediate_stem[-1] not in ("l", "s", "z"),
                ),
                
                (
                    "",
                    "e",
                    lambda stem: (self._measure(stem) == 1 and self._ends_cvc(stem)),
                ),
            ],
        )

    def _step1c(self, word):

        def nltk_condition(stem):
            return len(stem) > 1 and self._is_consonant(stem, len(stem) - 1)

        def original_condition(stem):
            return self._contains_vowel(stem)

        return self._apply_rule_list(
            word,
            [
                (
                    "y",
                    "i",
                    nltk_condition
                    if self.mode == self.NLTK_EXTENSIONS
                    else original_condition,
                )
            ],
        )

    def _step2(self, word):
        if self.mode == self.NLTK_EXTENSIONS:
            if word.endswith("alli") and self._has_positive_measure(
                self._replace_suffix(word, "alli", "")
            ):
                return self._step2(self._replace_suffix(word, "alli", "al"))

        bli_rule = ("bli", "ble", self._has_positive_measure)
        abli_rule = ("abli", "able", self._has_positive_measure)

        rules = [
            ("ational", "ate", self._has_positive_measure),
            ("tional", "tion", self._has_positive_measure),
            ("enci", "ence", self._has_positive_measure),
            ("anci", "ance", self._has_positive_measure),
            ("izer", "ize", self._has_positive_measure),
            abli_rule if self.mode == self.ORIGINAL_ALGORITHM else bli_rule,
            ("alli", "al", self._has_positive_measure),
            ("entli", "ent", self._has_positive_measure),
            ("eli", "e", self._has_positive_measure),
            ("ousli", "ous", self._has_positive_measure),
            ("ization", "ize", self._has_positive_measure),
            ("ation", "ate", self._has_positive_measure),
            ("ator", "ate", self._has_positive_measure),
            ("alism", "al", self._has_positive_measure),
            ("iveness", "ive", self._has_positive_measure),
            ("fulness", "ful", self._has_positive_measure),
            ("ousness", "ous", self._has_positive_measure),
            ("aliti", "al", self._has_positive_measure),
            ("iviti", "ive", self._has_positive_measure),
            ("biliti", "ble", self._has_positive_measure),
        ]

        if self.mode == self.NLTK_EXTENSIONS:
            rules.append(("fulli", "ful", self._has_positive_measure))
            rules.append(
                ("logi", "log", lambda stem: self._has_positive_measure(word[:-3]))
            )

        if self.mode == self.MARTIN_EXTENSIONS:
            rules.append(("logi", "log", self._has_positive_measure))

        return self._apply_rule_list(word, rules)

    def _step3(self, word):
        return self._apply_rule_list(
            word,
            [
                ("icate", "ic", self._has_positive_measure),
                ("ative", "", self._has_positive_measure),
                ("alize", "al", self._has_positive_measure),
                ("iciti", "ic", self._has_positive_measure),
                ("ical", "ic", self._has_positive_measure),
                ("ful", "", self._has_positive_measure),
                ("ness", "", self._has_positive_measure),
            ],
        )

    def _step4(self, word):
        measure_gt_1 = lambda stem: self._measure(stem) > 1

        return self._apply_rule_list(
            word,
            [
                ("al", "", measure_gt_1),
                ("ance", "", measure_gt_1),
                ("ence", "", measure_gt_1),
                ("er", "", measure_gt_1),
                ("ic", "", measure_gt_1),
                ("able", "", measure_gt_1),
                ("ible", "", measure_gt_1),
                ("ant", "", measure_gt_1),
                ("ement", "", measure_gt_1),
                ("ment", "", measure_gt_1),
                ("ent", "", measure_gt_1),
                
                (
                    "ion",
                    "",
                    lambda stem: self._measure(stem) > 1 and stem[-1] in ("s", "t"),
                ),
                ("ou", "", measure_gt_1),
                ("ism", "", measure_gt_1),
                ("ate", "", measure_gt_1),
                ("iti", "", measure_gt_1),
                ("ous", "", measure_gt_1),
                ("ive", "", measure_gt_1),
                ("ize", "", measure_gt_1),
            ],
        )

    def _step5a(self, word):
        if word.endswith("e"):
            stem = self._replace_suffix(word, "e", "")
            if self._measure(stem) > 1:
                return stem
            if self._measure(stem) == 1 and not self._ends_cvc(stem):
                return stem
        return word

    def _step5b(self, word):
        return self._apply_rule_list(
            word, [("ll", "l", lambda stem: self._measure(word[:-1]) > 1)]
        )

    def stem(self, word, to_lowercase=True):
        stem = word.lower() if to_lowercase else word

        if self.mode == self.NLTK_EXTENSIONS and word in self.pool:
            return self.pool[stem]

        if self.mode != self.ORIGINAL_ALGORITHM and len(word) <= 2:
            return stem

        stem = self._step1a(stem)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)

        return stem