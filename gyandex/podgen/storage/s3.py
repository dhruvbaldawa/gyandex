from typing import Optional, Dict, Any
import boto3
from botocore.client import Config
import mimetypes
import os


class S3CompatibleStorage:
    """
    A unified storage class for S3-compatible storage services (AWS S3, R2, B2, etc.)
    """

    def __init__(
        self,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: Optional[str] = None,
        region_name: str = "auto",
        custom_domain: Optional[str] = None,
        acl: str = "public-read",
    ):
        """
        Initialize the storage client.

        Args:
            bucket: Name of the bucket to use
            access_key_id: Access key ID for authentication
            secret_access_key: Secret access key for authentication
            endpoint_url: Optional endpoint URL for S3-compatible services
                        (e.g., 'https://xxx.r2.cloudflarestorage.com')
            region_name: AWS region or 'auto' for R2
            custom_domain: Optional custom domain for generating public URLs
            acl: Default ACL for uploaded files
        """
        self.bucket = bucket
        self.custom_domain = custom_domain
        self.acl = acl

        # Configure the S3 client with a generous timeout
        config = Config(
            connect_timeout=10, read_timeout=30, retries={"max_attempts": 3}
        )

        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=config,
        )

    def upload_file(
        self,
        file_path: str,
        destination_path: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to storage and return its public URL.

        Args:
            file_path: Local path to the file
            destination_path: Desired path in the bucket
            metadata: Optional metadata dictionary
            content_type: Optional content type, will be guessed if not provided

        Returns:
            Public URL of the uploaded file
        """
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = "application/octet-stream"

        extra_args = {"ACL": self.acl, "ContentType": content_type}

        if metadata:
            extra_args["Metadata"] = metadata

        self.client.upload_file(
            file_path, self.bucket, destination_path, ExtraArgs=extra_args
        )

        return self.get_public_url(destination_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Download a file from storage.

        Args:
            remote_path: Path of the file in the bucket
            local_path: Local path where the file should be saved
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(self.bucket, remote_path, local_path)

    def get_public_url(self, path: str) -> str:
        """
        Generate a public URL for a file.

        Args:
            path: Path of the file in the bucket

        Returns:
            Public URL for the file
        """
        if self.custom_domain:
            return f"https://{self.custom_domain}/{path}"

        # Get the endpoint URL from the client
        endpoint = self.client.meta.endpoint_url

        if endpoint:
            # For custom endpoints (R2, B2, etc.)
            return f"{endpoint}/{self.bucket}/{path}"
        else:
            # Default AWS S3 URL format
            region = self.client.meta.region_name
            return f"https://{self.bucket}.s3.{region}.amazonaws.com/{path}"

    def list_files(self, prefix: str = "") -> list[Dict[str, Any]]:
        """
        List all files in the bucket with the given prefix.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file information dictionaries
        """
        paginator = self.client.get_paginator("list_objects_v2")
        files = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                files.extend(page["Contents"])

        return files

    def delete_file(self, path: str) -> None:
        """
        Delete a file from storage.

        Args:
            path: Path of the file to delete
        """
        self.client.delete_object(Bucket=self.bucket, Key=path)
