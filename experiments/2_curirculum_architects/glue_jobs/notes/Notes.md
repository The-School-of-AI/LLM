The glue job script that is processing data on 4TB dataset.. using flex executors..

Please rewrite the code instead of updating to avoid missing any of the changes, keep existing file as is, create a _v5 file with the below changes, if you are adding any additional improvements call them out explicitly before the code changes.. Add all changes in the docstring as well.

1. Update the patterns and their usage as per the attached details.



--
1. Remove below metrics and their helper columns and variables, as they are either not effective or not relevant for our use case. Ensure the doc string is updated.. 

- risky_tld_count, not required
- sentence_boundary_coherence, not required
- non_printable_ratio, to support indic texts
- html_tag_density, currently not calculated correctly, we can try to fix it but for now removing the rule that uses it.

To save compute, remove
- punctuation_density
- dependency_depth_estimate, not really something that makes sense.. we can have code with more values than normal text, and it doesn't necessarily indicate spam
- num_numeric_tokens
- citation_count, not tested and not used
- list_marker_count
- step_indicator_count
- ellipsis_count
- dialogue_turn_count


- Remove stage 3 rejections completely


---
- Move rejection_levels inside domain
- Rename metrics folder to bands
- I've 1-1 mapping for domain to source.. I don't need domain and source repartition, my final should look like below

rejections
    - domain
        - rejection_level

bands
    - domain
        - band (passed records with band assignment)

- include text column from source to the bands and rejection output layer

---

```
    df = df.withColumn("tokens_list", F.split(F.lower("text"), r"\s+"))
    df = df.withColumn("unique_tokens", F.size(F.array_distinct("tokens_list")))
```
Do I need to keep tokens_list column, as I mm not using it anywhere else? 
Can't directly calculate unique_tokens without creating tokens_list column. 

-----


To Fix: 
symbol_count - very broad, we can have chars like ", . ; : ?" etc.. in normal text, we can try to find a better way to identify symbol spam

AGENTIC_PATTERN - Very broad, can match normal text as well, need to find a better way to identify this

COT_PATTERN - Very broad, can match normal text as well, need to find a better way to identify this

REASONING_PATTERN - Very broad, can match normal text as well, need to find a better way to identify this

TABLE_PATTERN - Can match delimeted data or other normal text.. have to be more specific

CODE_COMMENT_PATTERN - Can match normal text as well, need to find a better way to identify this

QUESTION_PATTERN - Can match normal text as well, need to find a better way to identify this

CODE_PATTERN - Can be more robust
MATH_PATTERN - Can be more robust




---

- Move rejection_levels inside domain
- Rename metrics folder to bands
- I've 1-1 mapping for domain to source.. I don't need domain and source repartition, my final should look like below

rejections
    - domain
        - rejection_level

bands
    - domain
        - band (passed records with band assignment)

- Add text column to the bands and rejection output layer

---

```
    df = df.withColumn("tokens_list", F.split(F.lower("text"), r"\s+"))
    df = df.withColumn("unique_tokens", F.size(F.array_distinct("tokens_list")))
```
Do I need to keep tokens_list column, as I mm not using it anywhere else? 
I can directly calculate unique_tokens without creating tokens_list column. 




-------

Look at t1 script for their optimzations and how the output data will look like, and then rewrite t2 from scratch, using optimized ways to do the processing.. use the existing t2 script only for reference

Below is the detailed requirement, updated and different from existing workflow in t2 script. Please make sure to read the details carefully and implement the new workflow as per the ask, rather than just making incremental changes to the existing t2 script.

---

I want to compute some metrics on the text column of the parquet files, and create 2 separate metric parquet files as output.. 
1. Rejection File
2. Metrics File (for non-rejected records)

Each record in the metrics file will have 
    - uuid for the record
    - the id of the record from the team1 file - each file can have multiple records
    - the file path of the input parquet file - this is important as we want to be able to join with this file later if needed
    - the computed metrics in their own columns

Each record in the Rejected file will have same columns as above, but with 2 additional columns:
    - is_rejected
    - rejection_reason


List of metrics are provided in 2 files, along with some details on how to calculate it, rejection priority, rejection threshold etc..


I want to compute metrics in the rejection priority order group, and if a record is rejected in lower rejection group, I do not want to compute the rest of the metrics for that record, just upate the is_rejected and rejection_reason columns, add it to rejection file, remove from metrics file and move on to the next record.


We plan to use this to train a 70B model and want to use these metrics for curriculum design and coreset learning.. 

Ensure the new script is optimized to handle ~4TB data efficiently, and run without running into performance issues.  

Just to be clear, I want to run everything on glue directly, no orchestration or anything.. Optimize for cost first and then speed. This will be run on adhoc basis, so we want to make sure the code is optimized for cost and can run within a reasonable time frame, rather than trying to make it run as fast as possible with higher costs.


Some optimization items to think about and implement:
1. eliminate python udf, use vectorized operations and built-in Spark functions where possible
2. Implement "Predicate Pushdown" and Pruning to minimize data read and processed
3. Use Spark's built-in functions for string manipulation, length calculations, and regex operations instead of Python UDFs
4. Handle Memory Management & Caching 
5. Glue-Specific Tuning - I'm using Glue 5.0 with G2X workers, it supports Spark 3.5, which features Adaptive Query Execution (AQE). We can use AQE to automatically handle coalecing partitions and optimizing join/shuffle strategies in real-time 

Explain any trade-offs or design choices, and the rationale behind them in the readme to ensure future teams can understand and maintain the code effectively..

Ensure that all the metrics are covered, and if any of the metric is not needed/not implemented, explain why and give the reason as well in readme.. 
