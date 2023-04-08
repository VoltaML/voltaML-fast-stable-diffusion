import logging
import os
from io import BytesIO
from typing import Optional
from urllib.parse import quote_plus

import boto3

logger = logging.getLogger(__name__)


class R2Bucket:
    """
    Class for interacting with R2 bucket.

    Args:
        endpoint: R2 endpoint. `R2_ENDPOINT` env var.
        bucket_name: Bucket name. `R2_BUCKET_NAME` env var.
        key: secret access id. `AWS_ACCESS_KEY_ID` env var.
        secret: secret access key. `AWS_SECRET_ACCESS_KEY` env var.
        dev_address: Development address where files can be seen publicly. `R2_DEV_ADDRESS` env var.
    """

    def __init__(
        self,
        endpoint: str,
        bucket_name: str,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        dev_address: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.key = key
        self.secret = secret
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.key,
            aws_secret_access_key=self.secret,
        )
        self.dev_address = dev_address if dev_address else os.getenv("R2_DEV_ADDRESS")

    def upload_file(self, file: BytesIO, filename: str):
        "Upload file to R2 bucket."
        self.client.upload_fileobj(
            file, self.bucket_name, filename, ExtraArgs={"ACL": "public-read"}
        )

        if self.dev_address:
            url = f"{self.dev_address}/{quote_plus(filename)}"
            logger.info(f"Uploaded file to R2: {url}")

            return url
