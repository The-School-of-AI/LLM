import argparse
import os
import glob
import duckdb

# -------------------------
# Helpers
# -------------------------

def get_input_files(mode, path):
    files = []
    if mode == "full":
        if not os.path.isdir(path):
             raise NotADirectoryError(f" --full expects a directory, but found: {path}")
        search_path = os.path.join(path, "*.jsonl")
        files = sorted(glob.glob(search_path))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {path}")
    elif mode == "part":
        if not os.path.isfile(path):
            raise FileNotFoundError(f" --part expects a file, but not found: {path}")
        files = [path]
    return files

# -------------------------
# Function 1: Inspection Output (Plain JSONL)
# -------------------------

def generate_inspection_jsonl(con, input_files, output_dir, dataset_name, language):
    """
    Generates a plain, uncompressed JSONL file for human inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ncert_inspection.jsonl")
    output_file_sql = output_file.replace("\\", "/")

    print(f"   👁️  Generating Inspection File...")
    print(f"       -> {output_file}")

    # chr(10) for real newlines; %% in strftime for f-string safety
    query = f"""
    COPY (
        SELECT
            md5(concat(
                '{dataset_name}', '{language}',
                coalesce(Subject, ''), coalesce(Topic, ''), coalesce(Question, ''),
                coalesce(Answer, ''), coalesce(Difficulty, ''), cast(grade as VARCHAR)
            )) AS id,

            'Subject: ' || coalesce(Subject, 'General') || chr(10) ||
            'Topic: ' || coalesce(Topic, '') || chr(10) || chr(10) ||
            'Explanation:' || chr(10) || coalesce(Explanation, '') || chr(10) || chr(10) ||
            'Question:' || chr(10) || coalesce(Question, '') || chr(10) || chr(10) ||
            'Answer:' || chr(10) || coalesce(Answer, '') AS text,

            '{dataset_name}' AS source,
            strftime(current_timestamp, '%%Y-%%m-%%dT%%H:%%M:%%SZ') AS added,
            '2024-01-01T00:00:00Z' AS created,

            {{
                'language': '{language}',
                'grade': cast(grade as VARCHAR),
                'difficulty': Difficulty,
                'student_level': StudentLevel,
                'question_type': QuestionType,
                'license': 'MIT',
                'dataset_type': 'textbook_qa'
            }} AS metadata,

            lower(replace(trim(coalesce(Subject, 'general')), ' ', '_')) AS domain

        FROM read_json_auto({input_files})
    )
    TO '{output_file_sql}'
    (FORMAT JSON, ARRAY false, OVERWRITE_OR_IGNORE true);
    """

    con.sql(query)
    return output_file

# -------------------------
# Function 2: Final Output (Dolma Native GZIP)
# -------------------------

def generate_dolma_native(con, input_files, output_dir, dataset_name, language):
    """
    Generates the official Dolma-compliant .jsonl.gz file in the 'documents' folder.
    """
    documents_dir = os.path.join(output_dir, "documents")
    os.makedirs(documents_dir, exist_ok=True)
    output_file = os.path.join(documents_dir, "ncert_harmonized.jsonl.gz")
    output_file_sql = output_file.replace("\\", "/")

    print(f"   📦 Generating Dolma-Native File...")
    print(f"       -> {output_file}")

    query = f"""
    COPY (
        SELECT
            md5(concat(
                '{dataset_name}', '{language}',
                coalesce(Subject, ''), coalesce(Topic, ''), coalesce(Question, ''),
                coalesce(Answer, ''), coalesce(Difficulty, ''), cast(grade as VARCHAR)
            )) AS id,

            'Subject: ' || coalesce(Subject, 'General') || chr(10) ||
            'Topic: ' || coalesce(Topic, '') || chr(10) || chr(10) ||
            'Explanation:' || chr(10) || coalesce(Explanation, '') || chr(10) || chr(10) ||
            'Question:' || chr(10) || coalesce(Question, '') || chr(10) || chr(10) ||
            'Answer:' || chr(10) || coalesce(Answer, '') AS text,

            '{dataset_name}' AS source,
            strftime(current_timestamp, '%%Y-%%m-%%dT%%H:%%M:%%SZ') AS added,
            '2024-01-01T00:00:00Z' AS created,

            {{
                'language': '{language}',
                'grade': cast(grade as VARCHAR),
                'difficulty': Difficulty,
                'student_level': StudentLevel,
                'question_type': QuestionType,
                'license': 'MIT',
                'dataset_type': 'textbook_qa'
            }} AS metadata,

            lower(replace(trim(coalesce(Subject, 'general')), ' ', '_')) AS domain

        FROM read_json_auto({input_files})
    )
    TO '{output_file_sql}'
    (FORMAT JSON, ARRAY false, COMPRESSION 'GZIP', OVERWRITE_OR_IGNORE true);
    """

    con.sql(query)
    return output_file

# -------------------------
# Main Logic
# -------------------------

def run_conversion(input_files, output_dir, output_format, language="en"):
    con = duckdb.connect()
    dataset_name = "ncert_qa"

    print(f"   📂 Input:  {len(input_files)} file(s)")
    print(f"   📂 Output: {output_dir}")
    print(f"   ⚙️  Action: {output_format.upper()}")

    try:
        if output_format in ["jsonl", "both"]:
            generate_inspection_jsonl(con, input_files, output_dir, dataset_name, language)

        if output_format in ["dolma", "both"]:
            generate_dolma_native(con, input_files, output_dir, dataset_name, language)

        print(f"   ✅ All tasks completed successfully.")

    except Exception as e:
        print(f"   ❌ DuckDB Error: {e}")
        raise e

# -------------------------
# Argument Parsing
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw NCERT shards to Dolma Formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect only
  python convert_ncert.py --full ./data/ncert/ --format jsonl

  # Final build
  python convert_ncert.py --full ./data/ncert/ --format dolma

  # Do BOTH (Recommended)
  python convert_ncert.py --full ./data/ncert/ --format both
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", type=str, metavar="FOLDER_PATH", help="Path to input folder")
    group.add_argument("--part", type=str, metavar="FILE_PATH", help="Path to single file")

    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "dolma", "both"],
        default="both",
        help="Output format: 'jsonl' (inspection), 'dolma' (final gzip), or 'both'"
    )

    parser.add_argument("--output", type=str, default="./dolma_dataset/ncert/", help="Output directory")
    parser.add_argument("--lang", type=str, default="en", help="Language code (e.g., 'en', 'hi')")

    return parser.parse_args()

# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 Dolma Conversion Tool")
    print("=" * 60)

    if args.full:
        mode, path = "full", args.full
    elif args.part:
        mode, path = "part", args.part

    print(f"⚙️  Mode:   {mode}")
    print(f"📁 Source: {path}")
    print(f"📂 Output: {args.output}")
    print(f"🌐 Lang:   {args.lang}")

    try:
        input_files = get_input_files(mode, path)
        run_conversion(input_files, args.output, args.format, language=args.lang)

        print("\n" + "=" * 60)
        print("✅ Status: Complete")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")

if __name__ == "__main__":
    main()
