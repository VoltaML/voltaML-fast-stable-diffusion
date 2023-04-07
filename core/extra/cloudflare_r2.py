import logging
from io import BytesIO
from typing import Optional

import boto3

logger = logging.getLogger(__name__)


class R2Bucket:
    """
    Class for interacting with R2 bucket.

    Args:
        endpoint: R2 endpoint. Can be set with `R2_ENDPOINT` env var.
        bucket_name: Bucket name. Can be set with `R2_BUCKET_NAME` env var.
        key: secret access id. Can be set with `AWS_ACCESS_KEY_ID` env var.
        secret: secret access key. Can be set with `AWS_SECRET_ACCESS_KEY` env var.
    """

    def __init__(
        self,
        endpoint: str,
        bucket_name: str,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        dev_addr: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.key = key
        self.secret = secret
        self.bucket_name = bucket_name
        self.bucket = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.key,
            aws_secret_access_key=self.secret,
        )
        self.dev_addr = dev_addr

    def upload_file(self, file: BytesIO, filename: str):
        "Upload file to R2 bucket."
        out = self.bucket.upload_fileobj(file, self.bucket_name, filename)

        logger.debug(f"Out: {out}")

        if self.dev_addr:
            return f"{self.dev_addr}/{filename}"
        return
