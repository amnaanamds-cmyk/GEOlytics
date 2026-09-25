"""Server-Side Request Forgery defence for the crawler.

This is the security boundary of the whole product. GEOlytics accepts a URL
from an untrusted customer and fetches it from inside the production network,
which is the textbook SSRF setup: without this module a customer can point an
audit at the cloud metadata endpoint and read the instance's IAM credentials,
or at Redis, Postgres and Qdrant on localhost.

What is defended, and how:

* **Scheme allowlist.** Only http and https. `file://`, `gopher://` and friends
  never reach a socket.
* **Address filtering.** Loopback, private, link-local, reserved, multicast and
  unspecified addresses are refused, in both IPv4 and IPv6, including
  IPv4-mapped IPv6 (`::ffff:169.254.169.254`) and the alternate integer
  notations that `inet_aton` accepts (`http://2130706433/` is 127.0.0.1).
* **DNS rebinding.** Resolving a name and then letting httpx resolve it again
  leaves a window where the second answer points somewhere private. Every
  request therefore connects to the *validated IP* while carrying the original
  `Host` header and TLS SNI, so the certificate is still checked against the
  real hostname. The address that was vetted is the address that is dialled.
* **Redirects.** A public URL that 302s to the metadata endpoint defeats any
  check done only on the input. Redirects are followed manually and every hop
  is revalidated from scratch.
* **Port allowlist.** Stops the crawler being used to probe internal services
  on 22, 3306, 6379 and so on.
* **Response size.** A multi-gigabyte body would exhaust the worker; reading is
  capped and aborted past the limit.

`allow_private_addresses` exists for the test suite, which serves fixture sites
on 127.0.0.1, and for self-hosted installs auditing an intranet. Turning it on
in a multi-tenant deployment removes the protection this module exists to
provide.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

Reason = Literal[
    "scheme",
    "missing_host",
    "port",
    "dns",
    "blocked_address",
    "blocked_host",
    "too_many_redirects",
    "too_large",
]


class UrlPolicyError(Exception):
    """A URL was refused before or during fetching."""

    def __init__(self, reason: Reason, url: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail} ({url})")
        self.reason: Reason = reason
        self.url = url
        self.detail = detail


# Addresses that are publicly routable but still must never be fetched: cloud
# metadata services that hand out credentials to anything that asks.
_EXTRA_BLOCKED = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),   # AWS / GCP / Azure / DO
        ipaddress.ip_address("169.254.170.2"),     # AWS ECS task metadata
        ipaddress.ip_address("100.100.100.200"),   # Alibaba Cloud
        ipaddress.ip_address("192.0.0.192"),       # Oracle Cloud
        ipaddress.ip_address("fd00:ec2::254"),     # AWS IMDS over IPv6
    }
)

_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
        "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
        "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
        "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32",
        "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8", "2001:db8::/32",
    )
)


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    """What the crawler is permitted to fetch."""

    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    # None means "any port". Only sensible alongside allow_private_addresses,
    # for a self-hosted install or the test suite's ephemeral fixture server.
    allowed_ports: frozenset[int] | None = frozenset({80, 443, 8080, 8443})
    allow_private_addresses: bool = False
    # Hosts exempt from the address check, by name or literal. Prefer this over
    # `allow_private_addresses` when an install has to reach one known internal
    # host: it reopens the door for that host only, not for every address a
    # customer can name.
    allowed_hosts: frozenset[str] = frozenset()
    blocked_hosts: frozenset[str] = frozenset()
    max_redirects: int = 5
    max_bytes: int = 5 * 1024 * 1024
    timeout: float = 30.0

    @classmethod
    def from_settings(cls, settings: object) -> UrlPolicy:
        restrict = bool(getattr(settings, "crawl_restrict_ports", True))
        ports = getattr(settings, "crawl_allowed_ports", None) or ()
        return cls(
            allowed_ports=frozenset(ports) if restrict else None,
            allow_private_addresses=bool(
                getattr(settings, "crawl_allow_private_addresses", False)
            ),
            allowed_hosts=frozenset(
                h.lower() for h in getattr(settings, "crawl_allowed_hosts", ()) or ()
            ),
            max_bytes=int(getattr(settings, "crawl_max_bytes", 5 * 1024 * 1024)),
            timeout=float(getattr(settings, "crawl_timeout_seconds", 30.0)),
            blocked_hosts=frozenset(
                h.lower() for h in getattr(settings, "crawl_blocked_hosts", ()) or ()
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidatedTarget:
    """A URL that passed the policy, with the address it must be dialled on."""

    url: str
    scheme: str
    host: str
    port: int
    address: str
    dial_url: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_tls(self) -> bool:
        return self.scheme == "https"


def is_blocked_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address must never be dialled on behalf of a customer."""
    # An IPv4-mapped IPv6 address (::ffff:127.0.0.1) reaches the IPv4 host, so
    # unwrap it and judge the address that will actually be contacted.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    if ip in _EXTRA_BLOCKED:
        return True
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    if ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return True
    return any(ip in network for network in _BLOCKED_NETWORKS)


def resolve_host(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address a hostname resolves to.

    All of them are checked, not just the first: a host with one public and one
    private answer must be refused, because which one gets dialled is not ours
    to choose.
    """
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        return [literal]

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlPolicyError("dns", host, f"could not resolve host: {exc}") from exc

    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not addresses:
        raise UrlPolicyError("dns", host, "host resolved to no usable address")
    return addresses


def validate_url(url: str, policy: UrlPolicy | None = None) -> ValidatedTarget:
    """Check a URL against the policy and pin the address it may be dialled on."""
    policy = policy or UrlPolicy()
    parts = urlsplit(url)

    scheme = parts.scheme.lower()
    if scheme not in policy.allowed_schemes:
        raise UrlPolicyError(
            "scheme", url, f"scheme {scheme or '(none)'!r} is not one of "
            f"{sorted(policy.allowed_schemes)}"
        )

    host = (parts.hostname or "").lower()
    if not host:
        raise UrlPolicyError("missing_host", url, "no host in URL")
    if host in policy.blocked_hosts:
        raise UrlPolicyError("blocked_host", url, f"host {host!r} is blocked")

    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UrlPolicyError("port", url, f"invalid port: {exc}") from exc
    if policy.allowed_ports is not None and port not in policy.allowed_ports:
        raise UrlPolicyError(
            "port", url, f"port {port} is not one of {sorted(policy.allowed_ports)}"
        )

    addresses = resolve_host(host)
    if not policy.allow_private_addresses and host not in policy.allowed_hosts:
        for ip in addresses:
            if is_blocked_address(ip):
                raise UrlPolicyError(
                    "blocked_address", url,
                    f"host resolves to non-public address {ip}",
                )

    chosen = addresses[0]
    literal = f"[{chosen}]" if chosen.version == 6 else str(chosen)
    dial_url = urlunsplit(
        (scheme, f"{literal}:{port}", parts.path or "/", parts.query, "")
    )
    host_header = host if port in (80, 443) else f"{host}:{port}"

    return ValidatedTarget(
        url=url,
        scheme=scheme,
        host=host,
        port=port,
        address=str(chosen),
        dial_url=dial_url,
        headers={"Host": host_header},
    )


def safe_get(
    client: httpx.Client,
    url: str,
    policy: UrlPolicy | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET a URL under the policy, revalidating every redirect hop.

    Redirects are followed here rather than by httpx because each `Location`
    is attacker-controlled and has to go back through `validate_url`. The
    request is sent to the validated IP with the original Host header and TLS
    SNI, so DNS cannot change its answer between the check and the connection.
    """
    policy = policy or UrlPolicy()
    seen: list[str] = []
    current = url

    for _ in range(policy.max_redirects + 1):
        target = validate_url(current, policy)
        seen.append(current)

        request = client.build_request(
            "GET",
            target.dial_url,
            headers={**(headers or {}), **target.headers},
            timeout=policy.timeout,
            # Connect to the vetted IP, but present and verify the real
            # hostname. Without this the certificate check would fail against
            # the bare address and TLS would have to be weakened.
            extensions={"sni_hostname": target.host} if target.is_tls else None,
        )
        response = client.send(request, follow_redirects=False, stream=True)

        if response.is_redirect:
            location = response.headers.get("location", "")
            response.close()
            if not location:
                raise UrlPolicyError("blocked_host", current, "redirect without a Location")
            current = str(httpx.URL(current).join(location))
            continue

        try:
            _read_capped(response, policy.max_bytes)
        except Exception:
            response.close()
            raise
        response.close()
        # Report the URL the customer asked for, not the pinned-IP form.
        response.request.url = httpx.URL(seen[-1])
        return response

    raise UrlPolicyError(
        "too_many_redirects", url, f"more than {policy.max_redirects} redirects"
    )


def _read_capped(response: httpx.Response, max_bytes: int) -> None:
    """Read the body, refusing anything over the cap.

    Content-Length is checked first as a cheap rejection, but it is a hint from
    an untrusted server, so the streamed bytes are counted regardless.
    """
    # A declared length over the cap is a cheap early rejection. It is only a
    # hint from an untrusted server -- it can under-report, or be absent under
    # chunked encoding -- so the streamed bytes are counted either way.
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise UrlPolicyError(
            "too_large", str(response.request.url),
            f"Content-Length {declared} exceeds {max_bytes}",
        )

    total = 0
    chunks: list[bytes] = []
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise UrlPolicyError(
                "too_large", str(response.request.url),
                f"body exceeded {max_bytes} bytes",
            )
        chunks.append(chunk)

    # httpx will not expose .text/.content for a streamed response unless the
    # body is attached back onto it.
    response._content = b"".join(chunks)  # noqa: SLF001 - no public setter exists
    response.is_closed = True
