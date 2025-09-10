import asyncio
import socket
import ipaddress
from typing import Dict, Any, Optional, List, Iterable
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse

from agent_engine.agent_logger.agent_logger import AgentLogger
from config import PROXY_SETTINGS


# Initialize logger
logger = AgentLogger(name=PROXY_SETTINGS.get("logger_name", "EricVPNProxy"))


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
}

SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie"}


def _mask_header_value(key: str, value: str) -> str:
    k = key.lower()
    if k in SENSITIVE_HEADERS:
        return "***"
    return value


def _normalize_allowed_headers(headers: Optional[Iterable[str]]) -> Optional[set]:
    if headers is None:
        return None
    return {h.lower() for h in headers}


def _build_forward_headers(incoming: Dict[str, str], allowlist: Optional[set], preserve_host: bool) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in incoming.items():
        kl = k.lower()
        if kl in HOP_BY_HOP_HEADERS:
            continue
        if allowlist is not None and kl not in allowlist:
            continue
        if kl == "host" and not preserve_host:
            # host will be set by httpx based on URL
            continue
        out[k] = v
    return out


def _is_blocked_ip(ip: str, security_cfg: Dict[str, Any]) -> bool:
    ip_obj = ipaddress.ip_address(ip)
    if security_cfg.get("block_loopback", True) and ip_obj.is_loopback:
        return True
    if security_cfg.get("block_link_local", True) and ip_obj.is_link_local:
        return True
    if security_cfg.get("block_multicast", True) and ip_obj.is_multicast:
        return True
    if security_cfg.get("block_unspecified", True) and ip_obj.is_unspecified:
        return True
    if security_cfg.get("block_private_networks", True) and ip_obj.is_private:
        return True
    return False


def _resolve_host_ips(hostname: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(hostname, None)
        addrs = sorted({info[4][0] for info in infos})
        return addrs
    except Exception:
        return []


def _enforce_ssrf_policy(base_url: str, security_cfg: Dict[str, Any], route_allow_private: bool, route_allowed_cidrs: List[str]) -> None:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only http/https are allowed")

        hostname = parsed.hostname
        if not hostname:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upstream hostname")

        if hostname in set(security_cfg.get("blocked_hostnames", [])):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Target hostname is blocked")

        ips = _resolve_host_ips(hostname)
        if not ips:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to resolve upstream host")

        # CIDR allowlist
        cidr_networks = [ipaddress.ip_network(c, strict=False) for c in (route_allowed_cidrs or security_cfg.get("default_allowed_cidrs", []))]

        for ip in ips:
            ip_obj = ipaddress.ip_address(ip)
            if _is_blocked_ip(ip, security_cfg):
                # allow private if explicitly requested and passes CIDR allowlist (if any)
                if ip_obj.is_private and route_allow_private:
                    if cidr_networks and not any(ip_obj in n for n in cidr_networks):
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Private IP not in allowed CIDR")
                    continue
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Target IP blocked by SSRF policy")
            if cidr_networks and not any(ip_obj in n for n in cidr_networks):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Target IP not in allowed CIDR")
    except HTTPException:
            raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upstream URL or SSRF check error")


def _require_api_key_if_enabled(request: Request, cfg: Dict[str, Any]) -> None:
    keys = cfg.get("auth", {}).get("api_keys", [])
    if not keys:
        return
    provided = request.headers.get("X-API-Key")
    if not provided or provided not in set(keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")


def _limit_body_size(request: Request, max_bytes: int) -> None:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Request body too large")
        except ValueError:
            # If invalid content-length, conservatively reject
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Length header")


app = FastAPI(title="Eric Secure Reverse Proxy", version="1.0.0")


# CORS (optional)
cors_cfg = PROXY_SETTINGS.get("cors", {})
if cors_cfg.get("enabled", False):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg.get("allow_origins", ["*"]),
        allow_credentials=cors_cfg.get("allow_credentials", False),
        allow_methods=cors_cfg.get("allow_methods", ["*"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
        expose_headers=cors_cfg.get("expose_headers", []),
        max_age=cors_cfg.get("max_age", 600),
    )


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
    if not request_id:
        # Simple request id
        import uuid
        request_id = uuid.uuid4().hex

    # Basic masked logging
    info_headers = {k: _mask_header_value(k, v) for k, v in request.headers.items()}
    logger.info(f"incoming request {request.method} {request.url.path} request_id={request_id} headers={info_headers}")

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def build_httpx_client_cfg(cfg: Dict[str, Any]):
    timeouts_cfg = cfg["security"]["timeouts"]
    limits_cfg = cfg["security"]["limits"]
    timeout = httpx.Timeout(
        connect=timeouts_cfg.get("connect", 5.0),
        read=timeouts_cfg.get("read", 60.0),
        write=timeouts_cfg.get("write", 60.0),
        pool=timeouts_cfg.get("pool", 60.0),
    )
    limits = httpx.Limits(
        max_connections=limits_cfg.get("max_connections", 128),
        max_keepalive_connections=limits_cfg.get("max_keepalive_connections", 32),
    )
    return timeout, limits


timeout_cfg, limits_cfg = build_httpx_client_cfg(PROXY_SETTINGS)
client = httpx.AsyncClient(
    timeout=timeout_cfg,
    limits=limits_cfg,
    trust_env=False,
    follow_redirects=False,
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def index():
    return {
        "message": "Eric Secure Reverse Proxy",
        "routes": list(PROXY_SETTINGS.get("routes", {}).keys()),
        "usage": "/r/{route_name}/{path}",
    }


@app.api_route("/r/{route_name}/{full_path:path}", methods=PROXY_SETTINGS.get("global_supported_methods", ["GET"]))
async def route_proxy(route_name: str, full_path: str, request: Request):
    # auth
    _require_api_key_if_enabled(request, PROXY_SETTINGS)

    routes_cfg = PROXY_SETTINGS.get("routes", {})
    route_cfg = routes_cfg.get(route_name)
    if not route_cfg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown route name")

    allowed_methods = set(route_cfg.get("allowed_methods") or PROXY_SETTINGS.get("global_supported_methods", []))
    if request.method.upper() not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Method not allowed for this route")

    base_url = route_cfg.get("base_url")
    if not base_url:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Route misconfigured: missing base_url")

    # SSRF policy
    _enforce_ssrf_policy(
        base_url=base_url,
        security_cfg=PROXY_SETTINGS.get("security", {}),
        route_allow_private=bool(route_cfg.get("allow_private", False)),
        route_allowed_cidrs=route_cfg.get("allowed_cidrs") or [],
    )

    # size limit
    _limit_body_size(request, PROXY_SETTINGS.get("security", {}).get("max_request_body_bytes", 10 * 1024 * 1024))

    # Build target URL
    target_url = urljoin(base_url.rstrip('/') + '/', full_path)
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    # headers
    per_route_allowlist = route_cfg.get("request_headers_allowlist")
    allowlist = _normalize_allowed_headers(per_route_allowlist) if per_route_allowlist is not None else _normalize_allowed_headers(PROXY_SETTINGS["security"].get("request_headers_allowlist"))
    preserve_host = bool(route_cfg.get("preserve_host", False))
    fwd_headers = _build_forward_headers(dict(request.headers), allowlist=allowlist, preserve_host=preserve_host)

    # Inject X-Forwarded-*
    if (route_cfg.get("add_x_forwarded_headers") if route_cfg.get("add_x_forwarded_headers") is not None else PROXY_SETTINGS["security"].get("add_x_forwarded_headers", True)):
        client_host = request.client.host if request.client else ""
        proto = request.headers.get("X-Forwarded-Proto") or ("https" if request.url.scheme == "https" else "http")
        fwd_headers.setdefault("X-Forwarded-For", client_host)
        fwd_headers.setdefault("X-Forwarded-Proto", proto)
        fwd_headers.setdefault("X-Forwarded-Host", request.headers.get("host", ""))

    # streaming request body
    async def body_iter():
        async for chunk in request.stream():
            yield chunk

    try:
        async with client.stream(request.method, target_url, headers=fwd_headers, content=body_iter()) as upstream:
            # response headers
            resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}

            # response body size limiter
            max_resp = PROXY_SETTINGS.get("security", {}).get("max_response_body_bytes", 50 * 1024 * 1024)
            total = 0

            async def resp_aiter():
                nonlocal total
                async for part in upstream.aiter_raw():
                    total += len(part)
                    if total > max_resp:
                        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Upstream response too large")
                    yield part

            return StreamingResponse(resp_aiter(), status_code=upstream.status_code, headers=resp_headers)

    except HTTPException:
        raise
    except httpx.ConnectTimeout:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Upstream connect timeout")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Upstream read timeout")
    except httpx.HTTPError as e:
        logger.error(f"httpx error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Upstream error")
    except Exception as e:
        logger.exception(f"unexpected proxy error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal proxy error")


def run():
    import uvicorn
    server_cfg = PROXY_SETTINGS.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 3000))
    logger.info(f"Starting Eric Secure Reverse Proxy at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()


