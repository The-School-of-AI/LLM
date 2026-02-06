class ControlledS3Writer:
    def __init__(self, bucket: str):
        self.bucket = bucket

    def put(self, key: str, body: bytes):
        raise NotImplementedError("S3 writes disabled in local mode")
