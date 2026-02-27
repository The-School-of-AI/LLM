aws glue start-job-run --job-name T1_data_normalization --arguments '{"--DATASETS":"dolmas_books_v1_7"}' --region us-east-1

aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --arguments '{
        "--DATASETS":"dolmas_algebraic-stack_v1_7",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'


aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --arguments '{
        "--DATASETS":"dolmas_open-web-math_v1_7, dolma_cc_news_v1_7, dolma_stackexchange_v1_7, dolma_arxiv_v1_7, dolmas_tulu_flan_v1_7, dolma_megawika_v1_7, dolma_Pes2o_v1_7, dolma_reddit_v1_7",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'


# wrong dataset name, hence one failed
# aws glue start-job-run --job-name T1_data_normalization \
#     --region us-east-1 \
#     --arguments '{
#         "--DATASETS":"dolma_starcoder_v1_7, dolma_C4_v1_7",
#         "--write-shuffle-files-to-s3": "true",
#         "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
#     }'




# failed :
# dolmas_starcoder_v1_7 # wrong name
# dolma_Pes2o_v1_7  # skipped by mistake in the first run, hence re-running
# dolma_cc_en_head_v1_7 # failed with No space left on device during a spill() operation within an UnsafeExternalSorter


# dolmas_starcoder_v1_7 : No space left on device during a spill() operation within an UnsafeExternalSorter
# dolma_Pes2o_v1_7: INCONSISTENT_BEHAVIOR_CROSS_VERSION.WRITE_ANCIENT_DATETIME - Date/timestamp error 

# -------

# No space left on device during a spill() operation
aws glue start-job-run \
    --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 30 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS": "dolma_cc_en_head_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--write-shuffle-spills-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'


# Date issue
aws glue start-job-run \
    --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 10 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS": "dolma_Pes2o_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--write-shuffle-spills-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/ --conf spark.sql.parquet.int96RebaseModeInWrite=CORRECTED"
    }'


# No space left on device during a spill() operation
aws glue start-job-run \
    --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 10 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS": "dolmas_starcoder_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--write-shuffle-spills-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'
# again failed with 10 G2x - : org.apache.spark.SparkException: Job aborted due to stage failure: Task 10 in stage 0.0 failed 4 times, most recent failure: Lost task 10.3 in stage 0.0 (TID 137) (172.34.254.138 executor 2): org.apache.spark.memory.SparkOutOfMemoryError: error while calling spill() on org.apache.spark.util.collection.unsafe.sort.UnsafeExternalSorter@1e8be467 : No space left on device


# --- TODO



aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 40 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS":"dolma_RefineWeb_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'


aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 40 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS":"dolma_cc_en_tail_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'


aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 40 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS":"dolma_cc_en_middle_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'



aws glue start-job-run --job-name T1_data_normalization \
    --region us-east-1 \
    --worker-type G.2X \
    --number-of-workers 30 \
    --execution-class FLEX \
    --arguments '{
        "--DATASETS":"dolma_cc_news_v1_7,dolma_starcoder_v1_7",
        "--enable-auto-scaling": "true",
        "--write-shuffle-files-to-s3": "true",
        "--conf": "spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
    }'


# =======================

Metric Job runs:

# test single file
aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=books/part-00000-b476a77d-9ec8-48a2-bdd3-8b9af02bfe06-c000.zstd.parquet",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=books/part-00001-b476a77d-9ec8-48a2-bdd3-8b9af02bfe06-c000.zstd.parquet",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'


aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=books/part-00001-b476a77d-9ec8-48a2-bdd3-8b9af02bfe06-c000.zstd.parquet",
            "--MANUAL_RESTART": "jr_1a0013ca51b87a113dafe671c0605cbb83f270d55da0ad166170d3ca2ae7a84d",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'



aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=sangraha_te/",
            "--MANUAL_RESTART": "jr_1a0013ca51b87a113dafe671c0605cbb83f270d55da0ad166170d3ca2ae7a84d",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'



aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 3 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=sangraha_ta/",
            "--MANUAL_RESTART": "jr_1a0013ca51b87a113dafe671c0605cbb83f270d55da0ad166170d3ca2ae7a84d",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'


aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=ncert/part-00000-c114471d-f600-4ef3-889c-98730360805c-c000.zstd.parquet",
            "--MANUAL_RESTART": "jr_1a0013ca51b87a113dafe671c0605cbb83f270d55da0ad166170d3ca2ae7a84d",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'





aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 4 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=ncert/part-00000-c114471d-f600-4ef3-889c-98730360805c-c000.zstd.parquet",
            "--enable-auto-scaling": "true",            
            "--write-shuffle-files-to-s3": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3 --conf spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
        }'

aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/",
            "--SOURCE": "ncert",
            "--enable-auto-scaling": "true",            
            "--write-shuffle-files-to-s3": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3 --conf spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
        }'



        "--MANUAL_RESTART": "jr_00f2706f20272517f70b39d628e08bad5a8261e2940d932835640b8c5a01cc86",


aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 5 \
        --execution-class FLEX \
        --arguments '{
            "--SOURCE": "books",
            "--enable-auto-scaling": "true",            
            "--write-shuffle-files-to-s3": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3 --conf spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
        }'

aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 5 \
        --execution-class FLEX \
        --arguments '{
            "--SOURCE": "sangraha_te",
            "--enable-auto-scaling": "true",            
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'



aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--comment": "Test - failed with no source"
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=books/part-00000-b476a77d-9ec8-48a2-bdd3-8b9af02bfe06-c000.zstd.parquet",
            "--enable-auto-scaling": "true",            
            "--write-shuffle-files-to-s3": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3 --conf spark.shuffle.sort.io.plugin.class=com.amazonaws.spark.shuffle.io.cloud.ChopperPlugin --conf spark.shuffle.storage.path=s3://t1-dataacquisition-datasets/processed_dataset/glue_shuffle_temp/"
        }'


------



aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 5 \
        --arguments '{
            "--SOURCE": "books",
            "--ESTIMATED_SIZE_GB":"10",
            "--enable-auto-scaling": "true",            
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'


aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.1X \
        --number-of-workers 2 \
        --execution-class FLEX \
        --arguments '{
            "--SOURCE": "ncert",
            "--ESTIMATED_SIZE_GB":"1",
            "--enable-auto-scaling": "true",            
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'






aws glue start-job-run --job-name T123_metrics_calculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 10 \
        --execution-class FLEX \
        --arguments '{
            "--SOURCE": "stackexchange",
            "--ESTIMATED_SIZE_GB":"23.3",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'


