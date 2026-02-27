aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 100 \
        --arguments '{
            "--SOURCE": "cc_middle",
            "--ESTIMATED_SIZE_GB":"932.2",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 100 \
        --arguments '{
            "--SOURCE": "cc_tail",
            "--ESTIMATED_SIZE_GB":"840.1",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 100 \
        --arguments '{
            "--SOURCE": "refinedweb",
            "--ESTIMATED_SIZE_GB":"829.3",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 100 \
        --arguments '{
            "--SOURCE": "cc_head",
            "--ESTIMATED_SIZE_GB":"723.2",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 60 \
        --arguments '{
            "--SOURCE": "C4",
            "--ESTIMATED_SIZE_GB":"266.7",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 40 \
        --arguments '{
            "--SOURCE": "Starcoder",
            "--ESTIMATED_SIZE_GB":"189.8",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 40 \
        --arguments '{
            "--SOURCE": "reddit",
            "--ESTIMATED_SIZE_GB":"158.0",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 20 \
#         --arguments '{
#             "--SOURCE": "pes2o",
#             "--ESTIMATED_SIZE_GB":"98.8",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 8 \
#         --arguments '{
#             "--SOURCE": "megawika",
#             "--ESTIMATED_SIZE_GB":"28.3",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 8 \
#         --arguments '{
#             "--SOURCE": "flan",
#             "--ESTIMATED_SIZE_GB":"27.1",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 8 \
#         --arguments '{
#             "--SOURCE": "redpajama-arxiv",
#             "--ESTIMATED_SIZE_GB":"24.0",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 8 \
#         --arguments '{
#             "--SOURCE": "stackexchange",
#             "--ESTIMATED_SIZE_GB":"23.3",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 8 \
#         --arguments '{
#             "--SOURCE": "sangraha_hi",
#             "--ESTIMATED_SIZE_GB":"22.2",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 8 \
        --arguments '{
            "--SOURCE": "sangraha_bn",
            "--ESTIMATED_SIZE_GB":"17.7",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 8 \
        --arguments '{
            "--SOURCE": "cc_news",
            "--ESTIMATED_SIZE_GB":"16.5",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 5 \
        --arguments '{
            "--SOURCE": "proof_pile_2-open_web_math",
            "--ESTIMATED_SIZE_GB":"12.1",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

aws glue start-job-run --job-name T2MetricsCalculation \
        --region us-east-1 \
        --worker-type G.2X \
        --number-of-workers 5 \
        --arguments '{
            "--SOURCE": "proof_pile_2-algebraic_stack",
            "--ESTIMATED_SIZE_GB":"10.3",
            "--enable-auto-scaling": "true",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_te",
#             "--ESTIMATED_SIZE_GB":"8.9",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_mr",
#             "--ESTIMATED_SIZE_GB":"7.6",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "books",
#             "--ESTIMATED_SIZE_GB":"7.3",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_ml",
#             "--ESTIMATED_SIZE_GB":"7.1",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_gu",
#             "--ESTIMATED_SIZE_GB":"6.0",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_kn",
#             "--ESTIMATED_SIZE_GB":"4.5",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_ta",
#             "--ESTIMATED_SIZE_GB":"3.0",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'


# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_or",
#             "--ESTIMATED_SIZE_GB":"2.3",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 5 \
#         --arguments '{
#             "--SOURCE": "sangraha_pa",
#             "--ESTIMATED_SIZE_GB":"2.1",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'

# aws glue start-job-run --job-name T2MetricsCalculation \
#         --region us-east-1 \
#         --worker-type G.2X \
#         --number-of-workers 2 \
#         --arguments '{
#             "--SOURCE": "sangraha_as",
#             "--ESTIMATED_SIZE_GB":"0.6",
#             "--enable-auto-scaling": "true",
#             "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
#         }'
