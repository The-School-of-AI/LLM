Let's reset few things:

1. We will have 500GB not 1TB of data. That should change the way we do things.
2. We will not do any processing for t1 output.. just add few columns and write to s3. Below is sample script.. Create a new script with optimi
```
df_out = (
    df
    .withColumn("hash", F.sha2(F.col("text"), 256))
    .withColumn("dataset", F.lit("dolma"))
    .withColumn("domain", F.lit(domain))
    .withColumn("source", F.lit(external_source))
    .withColumn("language", F.lit("en"))
    .withColumn("metadata", F.col("metadata").cast("string"))
    .withColumn("version", F.lit(version))
    # Reorder/select columns explicitly
    .select(
        "id",
        "hash",
        "dataset",
        "domain",
        "source",
        "text",
        "language",
        "metadata",
        "added",
        "created",
        "version",
    )
)
```
3. Split the script into 2 scripts, 
    1. Only for writing all the t1 partioned by the source.., I should be able to run this on selected folders only..
    2. Ready the t1 partitioned data, and do the next set of processing steps


3. For processing t2, validate the metrics, rewrite the whole script using best/optimized ways to do the processing ..


My complete dataset is around 500GB.. I'm optimizing for cost and processing time. I'm thinking of 2 ways to do this:
1. If I do not run on the complete folder at once, and plan to run separate job for separate folders. Will that reduce the cost and processing time for me.? By reduce failure overhead due to memory and probably cost by avoiding data shuffles between executors.? What are your thoughts on that?

2. If we split the job into t1 and t2, we can run t1 on smaller cluster with less memory and less cost, and then run t2 on bigger cluster with more 
memory and more cost. This way we can avoid doing any shuffle in t1 and reduce the chances of failure and also reduce cost by avoiding shuffles between executors. Is that a fair assumption?

