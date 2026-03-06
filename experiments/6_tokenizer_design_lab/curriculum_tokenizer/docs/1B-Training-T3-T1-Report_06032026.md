======================================================================
T3 SUMMARY
======================================================================
  T3 file count     : 12,231
  T3 total rows     : 19,882,213
  T3 total tokens   : 9,895,522,858  (~9.896 B)

  Rows per T3 file  : min=1  max=41,017  avg=1,626

======================================================================
T1 SUMMARY (from T3 t1_file_path references)
======================================================================
  T1 unique file count : 25,420
  T1 files with size   : 25,420 / 25,420
  T1 average size     : 173.44 MB
  T1 largest size     : 2.68 GB
    -> source=ai-bharath-samanantar/part-00003-affb5b4b-f2d9-4baf-ba7c-b312c6481dfd-c000.zstd.parquet
  T1 smallest size    : 184.78 KB
    -> source=ai-bharath-daily/part-00011-bd9a083c-1676-402d-8e1b-3cfba1618ff1-c000.zstd.parquet
  T1 total size (all) : 4305.63 GB  (4,623,131,716,729 bytes)

======================================================================
TOKENIZATION TRAINING PLANNING
======================================================================
  T3 coreset files     : 12,231
  T3 total rows         : 19,882,213
  Token budget          : 9,895,522,858 (~9.896 B)
  Unique T1 parquets    : 25,420
  T1 data to read       : 4305.63 GB
  Blocks (4096 tokens)  : ~2,415,899
======================================================================

  Written: analyze_t3_t1_1b_report_t3.csv
  Written: analyze_t3_t1_1b_report_t1.csv
  Written: analyze_t3_t1_1b_report_summary.csv