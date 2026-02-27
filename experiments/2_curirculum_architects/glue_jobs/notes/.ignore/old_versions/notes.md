
This glue script is processed by another team(team 1) in glue job for ~1TB of data.. 

I want to compute some metrics on the text column, and create a separate metrics parquet file for my outcome, without disrupting team 1 file/flow.. 

Each record in the metrics file will have 
    0. uuid for the record
    1. the id of the record from the team1 file (each file can have multiple records)
    2. the file path of the team1 file
    3. the computed metrics in their own columns
    4. is_rejected
    5. rejection_reason


Attached the list of metrics from team 2 and team 3, along with some idea on how to calculate it, rejection priority, rejection threshold etc..


I want to compute metrics in the rejection priority order, and if a record is rejected based on a metric, I do not want to compute the rest of the metrics for that record, just upate the is_rejected and rejection_reason columns and move on to the next record.

Ensure that this should be written to a different parquet file than team 1 file, but this should have the file path of the team 1 file as a column, so that we can join them later if needed. 



Attached is the input folder details that we'll process, number of json.gz files in each folder is also mentioned along with average file size. 

Create the script such that, we process one external source and then go to the next one, if that is the best way to do it.. Output should also be generated separately by the external source level. 

Optimize the code to handle this folder-wise processing efficiently, ensuring that we can scale up to the larger data sizes without running into performance issues. I should be able to add/remove some folders if I don't want to process them

Just to be clear, I want to run everything on glue directly, no orchestration or anything.. 

---

We plan to use this to train a 70B model and want to use these metrics for curriculum design and coreset learning.. 

Ensure the new script is optimized

Some ideas:
1. eliminate python udf, use vectorized operations and built-in Spark functions where possible
2. Implement "Predicate Pushdown" and Pruning to minimize data read and processed
3. Use Spark's built-in functions for string manipulation, length calculations, and regex operations instead of Python UDFs
4. Memory Management & Caching - we currently use df_raw.cache(). On a Teradata scale, cache() will likely overflow the disk/memory of your Glue workers (DPUs) and cause "Executor Lost" errors. Some alternatives to consider:
   - Use `persist(StorageLevel.DISK_ONLY)` for intermediate DataFrames that are too large to fit in memory, if we must reuse the dataframe multiple times. 
   - Or simply process Team 1 and Team 2 sequentially to allow Spark to clear the lineage
   - Unpersist DataFrames as soon as they are no longer needed to free up resources
   - Consider using `checkpoint()` for very large DataFrames to truncate the lineage and reduce memory usage, etc..
5. Standard repartition(NUM_PARTITIONS) can lead to Data Skew if certain source files are much larger than others. 
    - One option is to Aim for Parquet files between 128MB and 256MB. Adjust NUM_PARTITIONS dynamically based on input size:

6. Any Glue-Specific Tuning - I'm using Glue 5.0 with G2X workers, it supports Spark 3.5, which features Adaptive Query Execution (AQE). AQE will automatically handle coalecing partitions and optimizing join/shuffle strategies in real-time 

Do you have any other suggestions for optimizations or improvements to the code? Please provide specific recommendations and explain the rationale behind them in the readme to ensure future teams can understand and maintain the code effectively.



----


If you think any of the non implemented metric is not needed when we consider the tradeoff, explain why and give me reason as well.. I can see some of them need external package..  If we want to add, update readme to ensure the how to add those packages and the dependencies are clear for future teams as well..