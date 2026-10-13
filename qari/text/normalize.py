"""Arabic normalizer applied identically to Quran text and ASR output.

Lossy by design: the result is only a matching key. Never display it.
"""
import re
import unicodedata

_DROP = (
    set(range(0x0610, 0x061B))      # Quranic annotation signs
    | set(range(0x064B, 0x0660))    # harakat, shadda, sukun, maddah, hamza above/below
    | {0x0640}                      # tatweel
    | set(range(0x06D6, 0x06EE))    # small high/low marks, small waw/yeh, end of ayah
    | set(range(0x08D3, 0x0900))    # extended Arabic marks
    | {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF}  # zero-width / bidi
)
_FOLD = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ی": "ي", "ئ": "ي",
    "ة": "ه", "ؤ": "و", "ک": "ك",
}
_DAGGER = 0x0670


def normalize(text: str) -> str:
    out: list[str] = []
    prev = ""  # last kept raw character, before folding
    for ch in unicodedata.normalize("NFKC", text):
        o = ord(ch)
        if o == _DAGGER:
            # Explicit alef in the simple script (مالك, العالمين), except after alef
            # maqsura (على, إلى). After a real yeh it stays: القيامة, يا قوم.
            if prev != "ى":
                out.append("ا")
            continue
        if o in _DROP:
            continue
        prev = ch
        ch = _FOLD.get(ch, ch)
        out.append(ch if "ء" <= ch <= "ي" else " ")
    # Uthmani writes alef-madda as hamza + alef (ءامنوا); the simple script writes آمنوا.
    return " ".join(re.sub("ءا", "ا", "".join(out)).split())


def normalize_words(text: str) -> list[str]:
    return normalize(text).split()
