Threshold tuning:

New in V2.7 (BOOK-FRIENDLY THRESHOLDS):
- Fixed excessive rejections of book content (1610/1738 books were rejected)
- Increased whitespace_ratio threshold: 0.6 → 0.75 (books have chapter breaks, structured layout)
- Increased non_printable_ratio threshold: 0.01 → 0.03 (books have Unicode formatting)
- Increased capitalization_ratio threshold: 0.5 → 0.6 (books have chapter titles)
- Added minimum word_count checks to capitalization and corruption rules
- Expected impact: 90%+ reduction in false book rejections