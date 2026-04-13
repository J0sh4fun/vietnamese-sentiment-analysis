import re
import unicodedata
from underthesea import word_tokenize
from functools import lru_cache

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
    Determines the correct index to place the tone mark within a Vietnamese syllable.

    Follows standard Vietnamese orthography rules to handle specific vowel 
    combinations and ending consonants.

    Args:
        chars (list of str): A list of characters forming the syllable.
        vowel_indices (list of int): The position indices of vowels in the syllable.

    Returns:
        int: The correct index in the 'chars' list where the tone should be applied.
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

lru_cache(maxsize=50000)
def normalize_word_tone(word):
    """
    Standardizes the tone placement for a single word or syllable.

    Args:
        word (str): The raw input syllable.

    Returns:
        str: The syllable with the tone mark shifted to the linguistically 
            correct position. If no tones are detected or it's a special token, 
            returns the original word.
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

class VietnameseTextProcessor:
    """
    A text processing pipeline specificially designed for Vietnamese NLP tasks.
    
    This class handels noise removal (HTML tags), text masking(URLs, emails, phone numbers),
    Unicode normalization (NFC), word segmentation, tone normalization, and stopword filtering.
    """

    def __init__(self, stopwords_path='vietnamese-stopwords.txt', remove_punctuation=False):
        """
        Initializes the preprocessor pipeline.

        Loads the stopwords list and builds the necessary mapping dictionaries
        for tone normalization to optimize processing speed.

        Args:
            stopwords_path (str): The file path to the Vietnamese stopwords list. 
                                Defaults to 'vietnamese-stopwords.txt'.
            remove_punctuation (bool): If True, strips all punctuation (Good for TF-IDF).
                                       If False, keeps structural marks (Good for RAG/LLM).
        """
        self.remove_punctuation = remove_punctuation
        self.stopwords = self._load_stopwords(stopwords_path)
        self.negation_words = {'không', 'chẳng', 'chưa'}
        self.stopwords = self.stopwords - self.negation_words
        

    def _load_stopwords(self, path):
        """
        Loads a set of stopwords for a specified text file.

        Args:
            path (str): The exact path to the text file containing stopwords

        Returns:
            set: a collection of unique stopwords. Returns an empty set if the file cannot be found.
        """
        try:
            with open(path, encoding='utf-8') as f:
                return set(line.strip().replace(" ", "_") for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: Stopword file not found at {path}. Proceeding without it.")
            return set()
    
    def _split_sentences(self, text):
        """
        Splits a raw paragraph into individual sentences based on structural punctuation.

        Uses positive lookbehind regular expressions to identify sentence 
        boundaries (periods, question marks, exclamation points) without 
        consuming the punctuation marks themselves.

        Args:
            text (str): The raw text string

        Returns:
            list of str: A list of individual sentences extracted from the paragraph. 
                        Empty strings are filtered out.
        """

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        return [s for s in sentences if s.strip()]
        
    def _clean_and_mask(self, text):
        """
        Removes HTML entities, masks structured data, and filters out noise.

        Unlike standard aggressive cleaning, this method preserves structural 
        punctuation (.,!?) for downstream tasks like sentence segmentation. 

        It replaces sensitive information (URLs, emails, phones) with uppercase 
        text-based placeholders (e.g., TOKPHONE) to prevent tokenization errors.

        Args:
            text (str): The input sentence string.

        Returns:
            str: The cleaned sentence with masked entities and preserved punctuation.
        """
        text = re.sub(r"</?[a-zA-Z]+.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", " TOKURL ", text)
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+", " TOKEMAIL ", text)
        text = re.sub(r"(0|\+84)\d{8,10}", " TOKPHONE ", text)

        # Handle punctuation based on config
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        else:
            # Preserve structural marks
            text = re.sub(r"[^\w\s.,!?]", " ", text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _restore_special_tokens(self, text):
        """
        Restores text-based placeholders back to standard special tokens.

        Converts internal tokens like 'tokphone' back to the conventional 
        '<phone>' format after the text has safely passed through the tokenizer.

        Args:
            text (str): The preprocessed sentence containing lowercase placeholders.

        Returns:
            str: The final sentence with standardized special tokens.
        """
        text = text.replace("tokurl", "<url>")
        text = text.replace("tokemail", "<email>")
        text = text.replace("tokphone", "<phone>")
        return text
    
    def transform(self, text_list):
        """
        Executes the end-to-end preprocessing pipeline on a batch of documents.

        The pipeline processes text hierarchically: it first splits each document 
        into sentences, then cleans, segments, and normalizes each sentence individually. 
        This preserves the contextual boundary of sentences, which is essential for 
        Retrieval-Augmented Generation (RAG) or sequence-to-sequence models.

        Args:
            text_list (list of str): A batch containing raw Vietnamese documents/paragraphs.

        Returns:
            list of list of str: A nested list where each element represents a document, 
                                and each document is a list of its constituent, fully 
                                preprocessed sentences.
        """
        cleaned_documents = []

        for document in text_list:
            document = unicodedata.normalize('NFC', document)

            # Break the paragraph into a list of sentences.
            sentences = [document] if self.remove_punctuation else self._split_sentences(document)
            
            cleaned_sentences = []
            for sentence in sentences:
                # Clean and mask for each sentence
                sentence = self._clean_and_mask(sentence).lower()

                # Tokenize
                segmented_sentence = word_tokenize(sentence, format='text')

                # Normalize Tone & Filter stopwords
                tokens = segmented_sentence.split()
                final_tokens = []
                for t in tokens:
                    # Split compound words to normalize tone for each syllable
                    syllables = t.split('_')
                    normalized_syllables = [normalize_word_tone(s) for s in syllables]
                    normalized_tokens = '_'.join(normalized_syllables)

                    # Filter stopwords while keeping masks
                    if (normalized_tokens not in self.stopwords) or normalized_tokens.startswith('tok'):
                        final_tokens.append(normalized_tokens)

                # Join sentence & restore masking
                joined_sentence = " ".join(final_tokens)
                final_sentence = self._restore_special_tokens(joined_sentence)
                # Ignore empty sentences(only stopwords and noise)
                if final_sentence.strip():
                    cleaned_sentences.append(final_sentence)

            # Format output based on the mode
            if self.remove_punctuation:
                cleaned_documents.append(" ".join(cleaned_sentences))
            else:
                cleaned_documents.append(cleaned_sentences)

        return cleaned_documents
            

    
# How to call class 
# preprocessor = VietnameseTextPreprocessor(remove_punctuation=True)
# clean_texts = preprocessor.transform(["Raw text 1", "Raw text 2"])


