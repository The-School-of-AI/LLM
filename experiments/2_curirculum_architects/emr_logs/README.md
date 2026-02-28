Deleted logs due to size constraints, but here is the README.md that accompanied the logs:

# EMR Serverless Logs
Sharing few logs from some of our EMR Serverless runs. These logs have all the spark logs, including the driver and executor logs. We have compressed them using `tar` with `xz` compression to achieve the best compression ratio, especially since many of the log files are already in .gz format.


To extract any archive later, please use: `tar -xJf <name>.tar.xz`

