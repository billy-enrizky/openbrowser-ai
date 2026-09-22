"""Loopback-only HTTP server for the local Context Atlas search example."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import mimetypes
import secrets
from collections.abc import Iterable, Mapping
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from examples.context_atlas.credentials import CredentialStore, CredentialStoreError
from examples.context_atlas.documents import DocumentInputError, prepare_passages
from openbrowser.jev.coordinator import BoundedSemanticSearch
from openbrowser.jev.semantic_search import SemanticSearch, SemanticSearchError
from openbrowser.jev.views import SemanticMatch, SemanticSearchResult


MAX_REQUEST_BYTES = 512_000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TOKEN_HEADER = "X-Context-Atlas-Token"
TOKEN_COOKIE_NAME = "context_atlas_token"
TOKEN_COOKIE_ATTRIBUTES = "Path=/; HttpOnly; SameSite=Strict"
API_METHODS = {
	"/api/health": "GET",
	"/api/key/status": "GET",
	"/api/search": "POST",
	"/api/key": "PUT, DELETE",
}
ALLOWED_CORS_HEADERS = "Content-Type, X-Context-Atlas-Token"
ALLOWED_CORS_METHODS = "GET, POST, PUT, DELETE, OPTIONS"


class ContextAtlasError(RuntimeError):
	"""Base class for safe local-server errors."""


class ContextAtlasInvalidRequest(ContextAtlasError):
	"""Raised when a request body does not match the documented shape."""


class ContextAtlasUnauthorized(ContextAtlasError):
	"""Raised when a local API request is not from an authorized client."""


class ContextAtlasNotConfigured(ContextAtlasError):
	"""Raised when a search is requested without a saved Jev key."""


class ContextAtlasRequestTooLarge(ContextAtlasError):
	"""Raised before parsing a request that exceeds the body limit."""


class ContextAtlasUpstreamError(ContextAtlasError):
	"""Raised when storage or Jev cannot complete a bounded search."""


class ContextAtlasRateLimitError(ContextAtlasUpstreamError):
	"""Raised when the Jev provider reports a rate limit."""


class ContextAtlasProviderLimitError(ContextAtlasUpstreamError):
	"""Raised when Jev rejects the bounded source input size."""


class ContextAtlasService:
	"""Validate local requests and delegate bounded searches to the core."""

	def __init__(
		self,
		*,
		searcher: object | None = None,
		credential_store: CredentialStore | None = None,
		model_name: str = "jev-latest",
	) -> None:
		self._credentials = credential_store or CredentialStore()
		self.model_name = model_name
		self._searcher = searcher

	def health(self) -> dict[str, object]:
		return {
			"ok": True,
			"model": self.model_name,
			"jev_configured": self._safe_configured(),
		}

	def key_status(self) -> dict[str, bool]:
		return {"configured": self._safe_configured()}

	def save_api_key(self, payload: Mapping[str, Any]) -> dict[str, bool]:
		if not isinstance(payload, Mapping):
			raise ContextAtlasInvalidRequest("request body must be an object")
		api_key = payload.get("api_key")
		if not isinstance(api_key, str) or not api_key.strip():
			raise ContextAtlasInvalidRequest("api_key must be non-empty")
		try:
			self._credentials.save(api_key)
		except CredentialStoreError as exc:
			raise ContextAtlasUpstreamError("credential could not be saved") from exc
		return {"configured": True}

	def clear_api_key(self) -> dict[str, bool]:
		try:
			self._credentials.clear()
		except CredentialStoreError as exc:
			raise ContextAtlasUpstreamError("credential could not be cleared") from exc
		return {"configured": False}

	async def search_payload(self, payload: Mapping[str, Any]) -> dict[str, object]:
		if not isinstance(payload, Mapping):
			raise ContextAtlasInvalidRequest("request body must be an object")
		query = payload.get("query")
		blocks = payload.get("blocks")
		if not isinstance(query, str) or not isinstance(blocks, (list, tuple)):
			raise ContextAtlasInvalidRequest("request must contain query and blocks")
		try:
			passages = prepare_passages(blocks, query=query)
		except DocumentInputError as exc:
			raise ContextAtlasInvalidRequest(str(exc)) from exc
		searcher = BoundedSemanticSearch(searcher=self._get_searcher())
		try:
			result = await searcher.search(query=query.strip(), passages=passages)
		except SemanticSearchError as exc:
			message = str(exc).lower()
			if "rate" in message or "429" in message:
				raise ContextAtlasRateLimitError("Jev rate limit") from exc
			if any(
				marker in message
				for marker in ("max_tokens", "max tokens", "context length", "input too large", "provider input limit")
			):
				raise ContextAtlasProviderLimitError("Jev provider input limit") from exc
			raise ContextAtlasUpstreamError("Jev search failed") from exc
		return serialize_result(result)

	def _safe_configured(self) -> bool:
		try:
			return self._credentials.configured()
		except CredentialStoreError:
			return False

	def _get_searcher(self) -> object:
		try:
			api_key = self._credentials.load()
		except CredentialStoreError as exc:
			raise ContextAtlasUpstreamError("credential store is unavailable") from exc
		if not api_key:
			raise ContextAtlasNotConfigured("Jev is not configured on the server")
		if self._searcher is not None:
			return self._searcher
		try:
			return SemanticSearch(api_key=api_key, model_name=self.model_name)
		except SemanticSearchError as exc:
			raise ContextAtlasUpstreamError(f"Jev search setup failed: {type(exc).__name__}") from exc


def serialize_result(result: SemanticSearchResult) -> dict[str, object]:
	"""Serialize only stable source-backed result fields."""
	return {
		"scores": [_serialize_match(match) for match in result.scores],
		"matches": [_serialize_match(match) for match in result.matches],
		"threshold": result.threshold,
		"elapsed_ms": result.elapsed_ms,
		"usage": _json_safe(result.usage),
	}


def parse_json_body(body: bytes) -> Mapping[str, Any]:
	try:
		payload = json.loads(body.decode("utf-8"))
	except (UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise ContextAtlasInvalidRequest("request body must be valid JSON") from exc
	if not isinstance(payload, Mapping):
		raise ContextAtlasInvalidRequest("request body must be an object")
	return payload


def validate_request_size(content_length: int | str | None) -> int:
	if content_length is None:
		raise ContextAtlasInvalidRequest("Content-Length is required")
	try:
		length = int(content_length)
	except (TypeError, ValueError) as exc:
		raise ContextAtlasInvalidRequest("Content-Length must be an integer") from exc
	if length < 0:
		raise ContextAtlasInvalidRequest("Content-Length must not be negative")
	if length > MAX_REQUEST_BYTES:
		raise ContextAtlasRequestTooLarge(f"request body exceeds {MAX_REQUEST_BYTES:,} bytes")
	return length


def create_server(
	*,
	host: str = DEFAULT_HOST,
	port: int = DEFAULT_PORT,
	service: ContextAtlasService | None = None,
	static_root: Path | None = None,
	allowed_origins: Iterable[str] | None = None,
	token: str | None = None,
) -> ThreadingHTTPServer:
	if not _is_loopback_host(host):
		raise ValueError("Context Atlas server must bind to a loopback host")
	server = ThreadingHTTPServer((host, port), ContextAtlasRequestHandler)
	server.context_atlas_service = service or ContextAtlasService()  # type: ignore[attr-defined]
	server.context_atlas_static_root = static_root or Path(__file__).with_name("web")  # type: ignore[attr-defined]
	server.context_atlas_token = _validate_token(token or secrets.token_urlsafe(32))  # type: ignore[attr-defined]
	server.context_atlas_allowed_origins = _normalise_allowed_origins(  # type: ignore[attr-defined]
		allowed_origins,
		server.server_port,
	)
	return server


class ContextAtlasRequestHandler(BaseHTTPRequestHandler):
	"""HTTP adapter with JSON errors and no request-body logging."""

	server: ThreadingHTTPServer

	def do_GET(self) -> None:
		path = urlsplit(self.path).path
		if path.startswith("/api/"):
			try:
				self._authorize_request(require_token=path != "/api/health")
			except ContextAtlasUnauthorized as exc:
				self._send_error_for(exc)
				return
		if path == "/api/health":
			self._send_json(HTTPStatus.OK, self.server.context_atlas_service.health())  # type: ignore[attr-defined]
			return
		if path == "/api/key/status":
			self._send_json(HTTPStatus.OK, self.server.context_atlas_service.key_status())  # type: ignore[attr-defined]
			return
		if path.startswith("/api/"):
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		self._serve_static(path)

	def do_POST(self) -> None:
		path = urlsplit(self.path).path
		if path.startswith("/api/"):
			try:
				self._authorize_request()
			except ContextAtlasUnauthorized as exc:
				self._send_error_for(exc)
				return
		if path != "/api/search":
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		try:
			payload = self._read_payload()
			result = asyncio.run(self.server.context_atlas_service.search_payload(payload))  # type: ignore[attr-defined]
			self._send_json(HTTPStatus.OK, result)
		except ContextAtlasError as exc:
			self._send_error_for(exc)

	def do_PUT(self) -> None:
		path = urlsplit(self.path).path
		if path.startswith("/api/"):
			try:
				self._authorize_request()
			except ContextAtlasUnauthorized as exc:
				self._send_error_for(exc)
				return
		if path != "/api/key":
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		try:
			payload = self._read_payload()
			result = self.server.context_atlas_service.save_api_key(payload)  # type: ignore[attr-defined]
			self._send_json(HTTPStatus.OK, result)
		except ContextAtlasError as exc:
			self._send_error_for(exc)

	def do_DELETE(self) -> None:
		path = urlsplit(self.path).path
		if path.startswith("/api/"):
			try:
				self._authorize_request()
			except ContextAtlasUnauthorized as exc:
				self._send_error_for(exc)
				return
		if path != "/api/key":
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		try:
			self._send_json(HTTPStatus.OK, self.server.context_atlas_service.clear_api_key())  # type: ignore[attr-defined]
		except ContextAtlasError as exc:
			self._send_error_for(exc)

	def do_OPTIONS(self) -> None:
		path = urlsplit(self.path).path
		if path not in API_METHODS:
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		try:
			self._authorize_request(require_token=False, require_origin=True)
			requested_method = self.headers.get("Access-Control-Request-Method", "").upper()
			allowed_methods = {method.strip() for method in API_METHODS[path].split(",")}
			if requested_method and requested_method not in allowed_methods:
				raise ContextAtlasUnauthorized("requested method is not allowed")
			requested_headers = {
				header.strip().lower()
				for header in self.headers.get("Access-Control-Request-Headers", "").split(",")
				if header.strip()
			}
			if not requested_headers.issubset({"content-type", TOKEN_HEADER.lower()}):
				raise ContextAtlasUnauthorized("requested header is not allowed")
		except ContextAtlasUnauthorized as exc:
			self._send_error_for(exc)
			return
		self.send_response(HTTPStatus.NO_CONTENT)
		self._send_cors_headers()
		self.send_header("Content-Length", "0")
		self.end_headers()

	def _read_payload(self) -> Mapping[str, Any]:
		content_type = self.headers.get("Content-Type", "")
		if content_type.split(";", 1)[0].strip().lower() != "application/json":
			raise ContextAtlasInvalidRequest("Content-Type must be application/json")
		length = validate_request_size(self.headers.get("Content-Length"))
		body = self.rfile.read(length)
		if len(body) != length:
			raise ContextAtlasInvalidRequest("request body was truncated")
		return parse_json_body(body)

	def _serve_static(self, path: str) -> None:
		root = self.server.context_atlas_static_root  # type: ignore[attr-defined]
		relative = unquote(path).lstrip("/") or "index.html"
		candidate = (root / relative).resolve()
		try:
			candidate.relative_to(root.resolve())
		except ValueError:
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		if not candidate.is_file():
			self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
			return
		body = candidate.read_bytes()
		self.send_response(HTTPStatus.OK)
		self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
		self.send_header("Content-Length", str(len(body)))
		self.send_header("Cache-Control", "no-store")
		self.send_header(
			"Set-Cookie",
			f"{TOKEN_COOKIE_NAME}={self.server.context_atlas_token}; {TOKEN_COOKIE_ATTRIBUTES}",  # type: ignore[attr-defined]
		)
		self.end_headers()
		self.wfile.write(body)

	def _send_error_for(self, error: ContextAtlasError) -> None:
		status = {
			ContextAtlasInvalidRequest: HTTPStatus.BAD_REQUEST,
			ContextAtlasUnauthorized: HTTPStatus.UNAUTHORIZED,
			ContextAtlasRequestTooLarge: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
			ContextAtlasNotConfigured: HTTPStatus.SERVICE_UNAVAILABLE,
			ContextAtlasRateLimitError: HTTPStatus.TOO_MANY_REQUESTS,
			ContextAtlasProviderLimitError: HTTPStatus.BAD_GATEWAY,
			ContextAtlasUpstreamError: HTTPStatus.BAD_GATEWAY,
		}.get(type(error), HTTPStatus.BAD_REQUEST)
		if isinstance(error, ContextAtlasProviderLimitError):
			message = "Jev rejected the source size. Retry with a shorter source or fewer visible passages."
		elif status == HTTPStatus.BAD_GATEWAY:
			message = "Jev search failed. Retry the request."
		else:
			message = {
				HTTPStatus.BAD_REQUEST: str(error),
				HTTPStatus.UNAUTHORIZED: "unauthorized",
				HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "request body is too large",
				HTTPStatus.SERVICE_UNAVAILABLE: "Jev is not configured",
				HTTPStatus.TOO_MANY_REQUESTS: "Jev rate limit reached",
			}.get(status, "request failed")
		self._send_json(status, {"error": message})

	def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
		body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json; charset=utf-8")
		self.send_header("Content-Length", str(len(body)))
		self.send_header("Cache-Control", "no-store")
		self._send_cors_headers()
		self.end_headers()
		self.wfile.write(body)

	def _send_cors_headers(self) -> None:
		origin = self.headers.get("Origin")
		allowed_origins = self.server.context_atlas_allowed_origins  # type: ignore[attr-defined]
		self.send_header("Vary", "Origin")
		if origin in allowed_origins:
			self.send_header("Access-Control-Allow-Origin", origin)
			self.send_header("Access-Control-Allow-Headers", ALLOWED_CORS_HEADERS)
			self.send_header("Access-Control-Allow-Methods", ALLOWED_CORS_METHODS)

	def _authorize_request(self, *, require_token: bool = True, require_origin: bool = False) -> None:
		if not _request_host_is_allowed(self.headers.get("Host"), self.server.server_port):
			raise ContextAtlasUnauthorized("request host is not allowed")
		origin = self.headers.get("Origin")
		if require_origin and not origin:
			raise ContextAtlasUnauthorized("Origin header is required")
		if origin and origin not in self.server.context_atlas_allowed_origins:  # type: ignore[attr-defined]
			raise ContextAtlasUnauthorized("request origin is not allowed")
		if require_token:
			provided_token = self._request_token()
			if not provided_token or not hmac.compare_digest(provided_token, self.server.context_atlas_token):  # type: ignore[attr-defined]
				raise ContextAtlasUnauthorized("request token is invalid")

	def _request_token(self) -> str | None:
		provided_token = self.headers.get(TOKEN_HEADER)
		if provided_token is not None:
			return provided_token
		cookie_header = self.headers.get("Cookie")
		if not cookie_header:
			return None
		cookies = SimpleCookie()
		try:
			cookies.load(cookie_header)
		except CookieError:
			return None
		cookie = cookies.get(TOKEN_COOKIE_NAME)
		return cookie.value if cookie is not None else None

	def log_message(self, format: str, *args: object) -> None:
		return


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Run the local Context Atlas Jev search server")
	parser.add_argument("--host", default=DEFAULT_HOST)
	parser.add_argument("--port", type=int, default=DEFAULT_PORT)
	parser.add_argument("--token", default=None, help=argparse.SUPPRESS)
	args = parser.parse_args(argv)
	server = create_server(host=args.host, port=args.port, token=args.token)
	print(f"Context Atlas server listening at http://{args.host}:{args.port}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		return 0
	finally:
		server.server_close()
	return 0


def _serialize_match(match: SemanticMatch) -> dict[str, object]:
	return {
		"passage_id": match.passage_id,
		"probability": match.probability,
		"sentence_index": match.sentence_index,
		"sentence_text": match.sentence_text,
	}


def _json_safe(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	return None


def _validate_token(token: str) -> str:
	if (
		not isinstance(token, str)
		or not token
		or not token.isascii()
		or not all(character.isalnum() or character in "-_" for character in token)
	):
		raise ValueError("Context Atlas token must be a non-empty string")
	return token


def _normalise_allowed_origins(origins: Iterable[str] | None, port: int) -> frozenset[str]:
	if origins is None:
		origins = (
			f"http://127.0.0.1:{port}",
			f"http://localhost:{port}",
			f"http://[::1]:{port}",
		)
	validated: set[str] = set()
	for origin in origins:
		if not isinstance(origin, str) or not origin or origin == "*":
			raise ValueError("CORS origins must be explicit, non-empty origins")
		parsed = urlsplit(origin)
		if not parsed.scheme or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
			raise ValueError("CORS origins must be scheme and host only")
		validated.add(origin)
	return frozenset(validated)


def _is_loopback_host(host: str) -> bool:
	if host.lower() == "localhost":
		return True
	try:
		return ipaddress.ip_address(host).is_loopback
	except ValueError:
		return False


def _request_host_is_allowed(host_header: str | None, port: int) -> bool:
	if not host_header:
		return False
	try:
		parsed = urlsplit(f"//{host_header}")
		return parsed.port == port and parsed.hostname is not None and _is_loopback_host(parsed.hostname)
	except ValueError:
		return False


if __name__ == "__main__":
	raise SystemExit(main())


__all__ = [
	"DEFAULT_HOST",
	"DEFAULT_PORT",
	"MAX_REQUEST_BYTES",
	"ContextAtlasError",
	"ContextAtlasInvalidRequest",
	"ContextAtlasNotConfigured",
	"ContextAtlasProviderLimitError",
	"ContextAtlasRateLimitError",
	"ContextAtlasRequestTooLarge",
	"ContextAtlasService",
	"ContextAtlasUnauthorized",
	"ContextAtlasUpstreamError",
	"create_server",
	"main",
	"parse_json_body",
	"serialize_result",
	"validate_request_size",
]
