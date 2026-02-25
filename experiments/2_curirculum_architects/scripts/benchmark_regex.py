import re
import time

# 1. Setup Data
# A mix of code and text, repeated to make it large (~1MB)
sample_text = (
    """
def hello_world():
    print("Hello")
    # Some comments
    # More comments
"""
    * 1000
    + "Some general text without code " * 10000
)

# Refined Pattern
CODE_PATTERN = re.compile(
    r"```|"
    r"def\s+\w+\(|"
    r"class\s+\w+\s*[:{\(]|"
    r"function\s+\w+\s*[\({]|"
    r"^\s*import\s+\w+|"
    r"from\s+[\w.]+\s+import\s+\w+|"
    r"from\s+\.\s+import\s+\w+",
    re.IGNORECASE | re.MULTILINE,
)

ITERATIONS = 100

# 2. Benchmark Search (Current)
start_search = time.perf_counter()
for _ in range(ITERATIONS):
    _ = CODE_PATTERN.search(sample_text)
end_search = time.perf_counter()
avg_search = (end_search - start_search) / ITERATIONS

# 3. Benchmark Findall (List Build)
start_findall = time.perf_counter()
for _ in range(ITERATIONS):
    _ = len(CODE_PATTERN.findall(sample_text))
end_findall = time.perf_counter()
avg_findall = (end_findall - start_findall) / ITERATIONS

# 4. Benchmark Finditer (Count Only)
start_finditer = time.perf_counter()
for _ in range(ITERATIONS):
    _ = sum(1 for _ in CODE_PATTERN.finditer(sample_text))
end_finditer = time.perf_counter()
avg_finditer = (end_finditer - start_finditer) / ITERATIONS

print(f"Doc Size: {len(sample_text)/1024:.2f} KB")
print(f"Search (First Match): {avg_search*1000:.4f} ms")
print(f"Findall (All Matches): {avg_findall*1000:.4f} ms")
print(f"Finditer (Count Only): {avg_finditer*1000:.4f} ms")
print(f"Finditer vs Search Diff: {(avg_finditer - avg_search)*1000:.4f} ms")
