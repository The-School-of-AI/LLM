import regex

word = "कोंबडी"
unicode_chars = list(word)
print(f"Word: {word}")
print(f"Unicode chars: {unicode_chars}")
print(f"Count: {len(unicode_chars)}")

graphemes = regex.findall(r"\X", word)
print(f"Graphemes: {graphemes}")
print(f"Grapheme Count: {len(graphemes)}")
