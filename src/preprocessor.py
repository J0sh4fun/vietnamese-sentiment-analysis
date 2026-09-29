import re
import warnings
from importlib.metadata import version
import unicodedata
from underthesea import word_tokenize
from functools import lru_cache
from assets.vietnamese_stopwords import VIETNAMESE_STOPWORDS

VOWELS = "aăâeêioôơuưy"
    
TONE_TABLE = {
    "a": ["a", "á", "à", "ả", "ã", "ạ"],
    "ă": ["ă", "ắ", "ằ", "ẳ", "ẵ", "ặ"],
    "â": ["â", "ấ", "ầ", "ẩ", "ẫ", "ậ"],
    "e": ["e", "é", "è", "ẻ", "ẽ", "ẹ"],
    "ê": ["ê", "ế", "ề", "ể", "ễ", "ệ"],
    "i": ["i", "í", "ì", "ỉ", "ĩ", "ị"],
    "o": ["o", "ó", "ò", "ỏ", "õ", "ọ"],
    "ô": ["ô", "ố", "ồ", "ổ", "ỗ", "ộ"],
    "ơ": ["ơ", "ớ", "ờ", "ở", "ỡ", "ợ"],
    "u": ["u", "ú", "ù", "ủ", "ũ", "ụ"],
    "ư": ["ư", "ứ", "ừ", "ử", "ữ", "ự"],
    "y": ["y", "ý", "ỳ", "ỷ", "ỹ", "ỵ"]
}

REVERSE_TONE = {}
for base, forms in TONE_TABLE.items():
    for idx, char in enumerate(forms):
        REVERSE_TONE[char] = (base, idx)

def find_tone_position(chars, vowel_indices):
    """
    Legacy heuristic only; known to corrupt correctly spelled words.

    Retained for explicit opt-in comparisons, not a validated orthography rule.

    Args:
        chars (list of str): A list of characters forming the syllable.
        vowel_indices (list of int): The position indices of vowels in the syllable.

    Returns:
        int: The heuristic's chosen index; this is not guaranteed to be correct.
    """
    vowels = [chars[i] for i in vowel_indices]

    # Rule 1: Prioritize e, o, ơ
    for i in vowel_indices:
        if chars[i] in ["ê", "ô", "ơ"]:
            return i

    # Rule 2: 3 vowels
    if len(vowel_indices) == 3:
        v1, v2, v3 = vowels
        if v3 in ["i", "y"]:
            pair = v1 + v2
            if pair in ["oa", "oe", "uy"]:
                return vowel_indices[1]
            return vowel_indices[0]
        return vowel_indices[1]

    # Rule 3: 2 vowels
    if len(vowel_indices) == 2:
        has_final = chars[-1] not in VOWELS
        if has_final:
            return vowel_indices[1]
        else:
            return vowel_indices[0]

    return vowel_indices[0]

@lru_cache(maxsize=50000)
def _legacy_reposition_word_tone(word):
    """
    Unvalidated legacy tone repositioning, including the known khuay bug.

    Args:
        word (str): The raw input syllable.

    Returns:
        str: The legacy heuristic output, which may be linguistically incorrect.
    """
    if all(c not in REVERSE_TONE for c in word):
        return word

    if word.startswith("<") and word.endswith(">"):
        return word

    tone = 0
    vowel_indices = []
    chars = []

    for i, c in enumerate(word):
        lower_c = c 

        if lower_c in REVERSE_TONE:
            base, tone_idx = REVERSE_TONE[lower_c]
            if tone == 0 and tone_idx != 0:
                tone = tone_idx
            chars.append(base)
            if base in VOWELS:
                vowel_indices.append(i)
        else:
            chars.append(lower_c)
            if lower_c in VOWELS:
                vowel_indices.append(i)

    if not vowel_indices:
        return word

    # Process the exceptions: qu, gi
    if len(chars) >= 2:
        if chars[0] == "q" and chars[1] == "u":
            vowel_indices = [i for i in vowel_indices if i != 1]
        if chars[0] == "g" and chars[1] == "i":
            vowel_indices = [i for i in vowel_indices if i != 1]

    if not vowel_indices:
        return word

    pos = find_tone_position(chars, vowel_indices)
    base_char = chars[pos]
    
    if base_char in TONE_TABLE:
        chars[pos] = TONE_TABLE[base_char][tone]

    return "".join(chars)


def normalize_unicode(text: str) -> str:
    """Canonical Unicode composition only; never move a tone to another letter."""
    return unicodedata.normalize("NFC", text)


def normalize_word_tone(word: str, *, reposition: bool = False) -> str:
    """Compatibility API: NFC by default, unvalidated legacy behavior by opt-in."""
    word = normalize_unicode(word)
    return _legacy_reposition_word_tone(word) if reposition else word


NEGATION_WORDS = frozenset({"không", "chẳng", "chưa", "chớ", "đừng"})
ENTITY_PATTERN = re.compile(
    r"(?P<email>[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+)"
    r"|(?P<url>\b(?:https?://|www\.)[^\s<>]+)"
    r"|(?P<phone>(?<!\w)(?:0|\+84)\d{8,10}(?!\w))",
    re.IGNORECASE,
)


def stopword_key(text: str) -> str:
    """Compare a whole token/entry in one NFC, lowercase, underscore form."""
    return "_".join(normalize_unicode(text).lower().replace("_", " ").split())


class VietnameseTextProcessor:
    """Stateless NFC/cleaning/segmentation/filtering; preserves one output per input.

    Empty results are empty strings, not discarded rows. Tone repositioning is
    disabled because the legacy heuristic has known counterexamples.
    """

    IMPLEMENTATION_VERSION = 3

    def __init__(self, *, tone_repositioning=False, remove_stopwords=True, stopwords=None,
                 word_segmentation=True):
        if any(type(value) is not bool for value in
               (tone_repositioning, remove_stopwords, word_segmentation)):
            raise TypeError("Preprocessing switches must be booleans.")
        self.tone_repositioning = tone_repositioning
        self.remove_stopwords = remove_stopwords
        self.word_segmentation = word_segmentation
        self.negation_words = NEGATION_WORDS
        words = VIETNAMESE_STOPWORDS if stopwords is None else stopwords
        canonical_words = {stopword_key(word) for word in words}
        self.stopwords = frozenset(word for word in canonical_words if word
                                  and not self.negation_words.intersection(word.split("_")))
        if tone_repositioning:
            warnings.warn("Legacy tone repositioning is unvalidated and can corrupt correct words; "
                          "enable only for explicit comparisons.", UserWarning, stacklevel=2)

    def to_config(self) -> dict:
        """JSON-safe settings plus the exact effective stopword snapshot/runtime."""
        return {
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "settings": {
                "unicode_normalization": "NFC",
                "tone_repositioning": self.tone_repositioning,
                "tokenizer_token_normalization": False,
                "word_segmentation": self.word_segmentation,
                "lowercase": True,
                "remove_stopwords": self.remove_stopwords,
                "empty_policy": "keep",
                "stopword_matching": "whole_token_underscore",
                "masking": "structured_spans_v2",
            },
            "stopwords": sorted(self.stopwords),
            "protected_negations": sorted(self.negation_words),
            "runtime": {"underthesea": version("underthesea"), "unicode": unicodedata.unidata_version},
        }

    @classmethod
    def from_config(cls, config: dict):
        """Restore exactly; fail rather than silently substitute settings/runtime."""
        settings = config["settings"]
        processor = cls(tone_repositioning=settings["tone_repositioning"],
                        remove_stopwords=settings["remove_stopwords"], stopwords=config["stopwords"],
                        word_segmentation=settings.get("word_segmentation", True))
        restored = processor.to_config()
        if config.get("implementation_version") == 2:
            # Version 2 always segmented. This explicit migration preserves that
            # exact behavior; an old model never gains a new preprocessing path.
            restored["implementation_version"] = 2
            del restored["settings"]["word_segmentation"]
        if restored != config:
            raise ValueError("Unsupported preprocessing configuration or runtime mismatch; "
                             "use the recorded implementation and dependency versions.")
        return processor

    def _segment_text(self, text: str) -> list[str]:
        # NFC must precede this regex: decomposed combining marks are not \w.
        text = re.sub(r"[^\w\s]", " ", text)
        text = " ".join(text.split())
        if not text:
            return []
        if self.tone_repositioning:
            # An explicit experimental rewrite happens BEFORE segmentation so
            # the segmenter sees exactly the spelling that reaches TF-IDF.
            text = re.sub(r"[^\W\d_]+", lambda match: normalize_word_tone(
                match.group(), reposition=True), text)
        # The segmenter also has a spelling/tone normalizer enabled by default.
        # Disable that independently; NFC is already handled explicitly above.
        tokens = (word_tokenize(text, format="text", use_token_normalize=False).split()
                  if self.word_segmentation else text.split())
        return [token for token in tokens if not self.remove_stopwords
                or stopword_key(token) not in self.stopwords
                or self.negation_words.intersection(stopword_key(token).split("_"))]

    def transform(self, text_list) -> list[str]:
        if isinstance(text_list, (str, bytes)):
            raise TypeError("transform expects an iterable of strings, not one string.")
        cleaned_documents = []
        for document in text_list:
            if not isinstance(document, str):
                raise TypeError("Each document must be a string; validate missing values before preprocessing.")
            document = normalize_unicode(document).lower()
            document = re.sub(r"</?[a-zA-Z]+[^>]*>", " ", document)
            tokens = []
            position = 0
            # Keep real entity markers out of segmentation and stopword filtering.
            # Literal 'TOKURL' text is never interpreted as a generated marker.
            for match in ENTITY_PATTERN.finditer(document):
                tokens.extend(self._segment_text(document[position:match.start()]))
                tokens.append(f"<{match.lastgroup}>")
                position = match.end()
            tokens.extend(self._segment_text(document[position:]))
            cleaned_documents.append(" ".join(tokens))
        return cleaned_documents

