"""KMS envelope encryption for auth profile data.

Uses AWS KMS GenerateDataKey to get a unique data key per profile,
encrypts the data locally with AES-256-GCM, stores the encrypted data key
alongside the ciphertext. Decryption calls KMS Decrypt to unwrap the data key.

The encryption context includes user_id to prevent cross-user decryption.
"""

import json
import logging
import os
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_NONCE_SIZE = 12
_TAG_SIZE = 16


def _get_kms_client():
    """Get boto3 KMS client."""
    import boto3
    return boto3.client("kms", region_name=os.getenv("AWS_REGION", "ca-central-1"))


def encrypt_auth_state(user_id: str, state: dict[str, Any]) -> tuple[bytes, bytes]:
    """Encrypt auth state with KMS envelope encryption.

    Returns:
        (encrypted_key, ciphertext) -- both bytes, store in DB as BYTEA.
    """
    key_arn = settings.KMS_AUTH_KEY_ARN
    if not key_arn:
        raise RuntimeError("KMS_AUTH_KEY_ARN not configured")

    client = _get_kms_client()
    context = {"purpose": "auth_profile", "user_id": user_id}

    response = client.generate_data_key(
        KeyId=key_arn,
        KeySpec="AES_256",
        EncryptionContext=context,
    )
    plaintext_key = response["Plaintext"]
    encrypted_key = response["CiphertextBlob"]

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    plaintext = json.dumps(state).encode("utf-8")
    nonce = os.urandom(_NONCE_SIZE)
    aesgcm = AESGCM(plaintext_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    # Defense-in-depth: reassign key variable (immutable bytes not truly wiped until GC)
    plaintext_key = b"\x00" * len(plaintext_key)

    return encrypted_key, nonce + ciphertext


def decrypt_auth_state(user_id: str, encrypted_key: bytes, ciphertext: bytes) -> dict[str, Any]:
    """Decrypt auth state using KMS envelope decryption.

    Returns:
        The original dict (cookies, localStorage, etc.).
    """
    key_arn = settings.KMS_AUTH_KEY_ARN
    if not key_arn:
        raise RuntimeError("KMS_AUTH_KEY_ARN not configured")

    client = _get_kms_client()
    context = {"purpose": "auth_profile", "user_id": user_id}

    response = client.decrypt(
        CiphertextBlob=encrypted_key,
        EncryptionContext=context,
    )
    plaintext_key = response["Plaintext"]

    nonce = ciphertext[:_NONCE_SIZE]
    ct = ciphertext[_NONCE_SIZE:]

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aesgcm = AESGCM(plaintext_key)
    plaintext = aesgcm.decrypt(nonce, ct, None)

    # Defense-in-depth: reassign key variable (immutable bytes not truly wiped until GC)
    plaintext_key = b"\x00" * len(plaintext_key)

    return json.loads(plaintext.decode("utf-8"))
