"""Cognito authentication helpers for HTTP and WebSocket requests."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException, Request, WebSocket, status

from app.core.config import settings

logger = logging.getLogger(__name__)

JWKS_CACHE_TTL_SECONDS = 3600

_jwks_cache: dict[str, Any] | None = None
_jwks_cache_expiry: float = 0.0
_jwks_lock = asyncio.Lock()


class AuthConfigError(RuntimeError):
    """Raised when auth is enabled but Cognito configuration is incomplete."""


class AuthTokenError(RuntimeError):
    """Raised when token is missing or invalid."""


@dataclass
class AuthPrincipal:
    """Authenticated user details extracted from Cognito JWT."""

    subject: str
    email: str | None
    username: str | None
    claims: dict[str, Any]


def _ensure_auth_configured() -> None:
    """Validate required Cognito settings when auth is enabled."""
    if not settings.AUTH_ENABLED:
        return

    missing: list[str] = []
    if not settings.get_cognito_issuer():
        missing.append("COGNITO_REGION + COGNITO_USER_POOL_ID (or COGNITO_ISSUER)")
    if not settings.COGNITO_APP_CLIENT_ID:
        missing.append("COGNITO_APP_CLIENT_ID")

    if missing:
        raise AuthConfigError(
            "Cognito auth is enabled but missing configuration: " + ", ".join(missing)
        )


def _base64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _decode_jwt_parts(token: str) -> tuple[str, str, str, dict[str, Any], dict[str, Any], bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthTokenError("Malformed JWT")

    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_base64url_decode(header_b64))
        payload = json.loads(_base64url_decode(payload_b64))
        signature = _base64url_decode(signature_b64)
    except Exception as e:
        raise AuthTokenError(f"Failed to decode JWT: {e}") from e

    return header_b64, payload_b64, signature_b64, header, payload, signature


def _build_rsa_public_key(jwk: dict[str, Any]) -> rsa.RSAPublicKey:
    if jwk.get("kty") != "RSA":
        raise AuthTokenError("Unsupported JWT key type")

    n = jwk.get("n")
    e = jwk.get("e")
    if not n or not e:
        raise AuthTokenError("Invalid JWK: missing n/e")

    try:
        n_int = int.from_bytes(_base64url_decode(n), byteorder="big")
        e_int = int.from_bytes(_base64url_decode(e), byteorder="big")
    except Exception as exc:
        raise AuthTokenError(f"Invalid JWK encoding: {exc}") from exc

    numbers = rsa.RSAPublicNumbers(e_int, n_int)
    return numbers.public_key()


async def _get_jwks() -> dict[str, Any]:
    global _jwks_cache, _jwks_cache_expiry

    now = time.time()
    if _jwks_cache and now < _jwks_cache_expiry:
        return _jwks_cache

    async with _jwks_lock:
        now = time.time()
        if _jwks_cache and now < _jwks_cache_expiry:
            return _jwks_cache

        issuer = settings.get_cognito_issuer()
        if not issuer:
            raise AuthConfigError("COGNITO issuer is not configured")

        jwks_url = f"{issuer}/.well-known/jwks.json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(jwks_url)
                response.raise_for_status()
                jwks = response.json()
        except Exception as exc:
            raise AuthConfigError(f"Failed to fetch Cognito JWKS: {exc}") from exc

        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise AuthConfigError("Invalid JWKS payload from Cognito")

        _jwks_cache = jwks
        _jwks_cache_expiry = now + JWKS_CACHE_TTL_SECONDS
        return jwks


def _extract_kid_key(jwks: dict[str, Any], kid: str) -> dict[str, Any]:
    keys = jwks.get("keys", [])
    for key in keys:
        if key.get("kid") == kid:
            return key
    raise AuthTokenError("Token key id not found in Cognito JWKS")


def _validate_claims(claims: dict[str, Any]) -> AuthPrincipal:
    issuer = settings.get_cognito_issuer()
    if not issuer:
        raise AuthConfigError("COGNITO issuer is not configured")

    client_id = settings.COGNITO_APP_CLIENT_ID
    if not client_id:
        raise AuthConfigError("COGNITO_APP_CLIENT_ID is not configured")

    token_issuer = claims.get("iss")
    if token_issuer != issuer:
        raise AuthTokenError("Invalid token issuer")

    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        raise AuthTokenError("Token missing exp claim")
    if time.time() >= float(exp):
        raise AuthTokenError("Token expired")

    token_use = claims.get("token_use")
    if token_use not in {"id", "access"}:
        raise AuthTokenError("Invalid Cognito token_use")

    if token_use == "id":
        aud = claims.get("aud")
        if isinstance(aud, list):
            if client_id not in aud:
                raise AuthTokenError("Invalid token audience")
        elif aud != client_id:
            raise AuthTokenError("Invalid token audience")
    else:
        if claims.get("client_id") != client_id:
            raise AuthTokenError("Invalid access token client_id")

    subject = claims.get("sub")
    if not subject:
        raise AuthTokenError("Token missing subject")

    email = claims.get("email")
    username = claims.get("cognito:username") or claims.get("username")
    return AuthPrincipal(
        subject=str(subject),
        email=str(email) if email else None,
        username=str(username) if username else None,
        claims=claims,
    )


async def verify_cognito_token(token: str) -> AuthPrincipal:
    """Verify Cognito JWT signature and claims."""
    _ensure_auth_configured()

    header_b64, payload_b64, _, header, payload, signature = _decode_jwt_parts(token)

    if header.get("alg") != "RS256":
        raise AuthTokenError("Unsupported JWT algorithm")

    kid = header.get("kid")
    if not kid:
        raise AuthTokenError("Token missing kid")

    jwks = await _get_jwks()
    jwk = _extract_kid_key(jwks, kid)
    public_key = _build_rsa_public_key(jwk)

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    try:
        public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
    except Exception as exc:
        raise AuthTokenError(f"Invalid token signature: {exc}") from exc

    return _validate_claims(payload)


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise AuthTokenError("Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthTokenError("Invalid Authorization header format")

    return token.strip()


async def get_current_user(request: Request) -> AuthPrincipal | None:
    """FastAPI dependency for authenticated requests."""
    if not settings.AUTH_ENABLED:
        return None

    try:
        token = _extract_bearer_token(request.headers.get("Authorization"))
        return await verify_cognito_token(token)
    except AuthConfigError as exc:
        logger.error("Auth config error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is misconfigured on the server",
        ) from exc
    except AuthTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


async def verify_token_string(token: str) -> AuthPrincipal:
    """Verify a raw JWT string (used for SSE query-param auth)."""
    _ensure_auth_configured()
    return await verify_cognito_token(token)


async def authenticate_websocket(websocket: WebSocket) -> AuthPrincipal | None:
    """Authenticate WebSocket connection via query token or Authorization header."""
    if not settings.AUTH_ENABLED:
        return None

    _ensure_auth_configured()

    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get("Authorization")
        token = _extract_bearer_token(auth_header)

    return await verify_cognito_token(token)

