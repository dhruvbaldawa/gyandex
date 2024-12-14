from typing import Union

from ..config.schema import S3StorageConfig
from ..storage.s3 import S3CompatibleStorage


# @TODO: Centralize this type and move this to a common place
def get_storage(config: Union[S3StorageConfig]) -> S3CompatibleStorage:  # pyright: ignore [reportInvalidTypeArguments]
    if config.provider != "s3":  # @TODO: Move this to a enum
        raise NotImplementedError(f"Unsupported storage provider: {config.provider}")

    return S3CompatibleStorage(
        bucket=config.bucket,
        access_key_id=config.access_key,
        secret_access_key=config.secret_key,
        region_name=config.region,
        endpoint_url=config.endpoint,
        custom_domain=config.custom_domain,
    )
