"""The SSRF policy.

This is the product's security boundary: GEOlytics fetches customer-supplied
URLs from inside the production network. Each test below corresponds to a way
that has historically been exploited.
"""

from __future__ import annotations

import ipaddress

import pytest

from geolytics.crawl.guard import (
    UrlPolicy,
    UrlPolicyError,
    is_blocked_address,
    resolve_host,
    validate_url,
)

PERMISSIVE = UrlPolicy(allow_private_addresses=True, allowed_ports=None)


class TestBlockedAddresses:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1", "127.1.1.1", "0.0.0.0", "10.0.0.1", "172.16.0.1",
            "172.31.255.254", "192.168.1.1", "169.254.169.254", "169.254.170.2",
            "100.100.100.200", "192.0.0.192", "100.64.0.1", "224.0.0.1",
            "240.0.0.1", "255.255.255.255", "198.18.0.1",
        ],
    )
    def test_ipv4_ranges_are_refused(self, address):
        assert is_blocked_address(ipaddress.ip_address(address))

    @pytest.mark.parametrize(
        "address",
        ["::1", "::", "fc00::1", "fd00::1", "fe80::1", "ff02::1", "fd00:ec2::254"],
    )
    def test_ipv6_ranges_are_refused(self, address):
        assert is_blocked_address(ipaddress.ip_address(address))

    @pytest.mark.parametrize(
        "address",
        ["::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:10.0.0.1"],
    )
    def test_ipv4_mapped_ipv6_is_unwrapped_before_judging(self, address):
        """`::ffff:169.254.169.254` reaches the IPv4 metadata endpoint."""
        assert is_blocked_address(ipaddress.ip_address(address))

    @pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700::1"])
    def test_public_addresses_are_allowed(self, address):
        assert not is_blocked_address(ipaddress.ip_address(address))


class TestSchemes:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://127.0.0.1:6379/_SET%20x%20y",
            "ftp://internal/",
            "redis://localhost:6379",
            "dict://localhost:11211/",
            "jar:http://x/!/",
            "//example.com/no-scheme",
        ],
    )
    def test_non_http_schemes_are_refused(self, url):
        with pytest.raises(UrlPolicyError) as exc:
            validate_url(url, PERMISSIVE)
        assert exc.value.reason == "scheme"

    def test_scheme_case_is_ignored(self):
        with pytest.raises(UrlPolicyError, match="scheme"):
            validate_url("FILE:///etc/passwd", PERMISSIVE)


class TestAlternateNotations:
    """Encodings that bypass a naive string check but still reach loopback."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",
            "http://127.1/",
            "http://0177.0.0.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
        ],
    )
    def test_loopback_spellings_are_refused(self, url):
        with pytest.raises(UrlPolicyError) as exc:
            validate_url(url, UrlPolicy(allowed_ports=None))
        assert exc.value.reason in {"blocked_address", "dns"}

    def test_decimal_integer_host_is_refused(self):
        """http://2130706433/ is 127.0.0.1 to the resolver."""
        with pytest.raises(UrlPolicyError) as exc:
            validate_url("http://2130706433/", UrlPolicy(allowed_ports=None))
        assert exc.value.reason in {"blocked_address", "dns"}


class TestPorts:
    @pytest.mark.parametrize("port", [22, 25, 3306, 5432, 6379, 6333, 9200, 11211])
    def test_internal_service_ports_are_refused(self, port):
        with pytest.raises(UrlPolicyError) as exc:
            validate_url(f"http://example.com:{port}/", UrlPolicy())
        assert exc.value.reason == "port"

    @pytest.mark.parametrize("url", ["http://example.com/", "https://example.com/"])
    def test_default_web_ports_are_allowed(self, url):
        assert validate_url(url, UrlPolicy()).port in (80, 443)

    def test_port_restriction_can_be_lifted_explicitly(self):
        assert validate_url("http://example.com:9999/", UrlPolicy(allowed_ports=None)).port == 9999

    def test_malformed_port_is_refused(self):
        with pytest.raises(UrlPolicyError) as exc:
            validate_url("http://example.com:notaport/", UrlPolicy())
        assert exc.value.reason == "port"


class TestHosts:
    def test_missing_host_is_refused(self):
        with pytest.raises(UrlPolicyError) as exc:
            validate_url("http:///just-a-path", PERMISSIVE)
        assert exc.value.reason == "missing_host"

    def test_explicit_host_blocklist(self):
        policy = UrlPolicy(allow_private_addresses=True, blocked_hosts=frozenset({"evil.test"}))
        with pytest.raises(UrlPolicyError) as exc:
            validate_url("http://evil.test/", policy)
        assert exc.value.reason == "blocked_host"

    def test_unresolvable_host_is_refused(self):
        with pytest.raises(UrlPolicyError) as exc:
            validate_url("http://no-such-host.invalid/", PERMISSIVE)
        assert exc.value.reason == "dns"


class TestAddressPinning:
    def test_target_dials_the_validated_address(self):
        target = validate_url("http://127.0.0.1:8099/index.html", PERMISSIVE)
        assert target.address == "127.0.0.1"
        assert "127.0.0.1:8099" in target.dial_url
        assert target.host == "127.0.0.1"

    def test_host_header_carries_the_original_name(self):
        target = validate_url("http://127.0.0.1:8099/x", PERMISSIVE)
        assert target.headers["Host"] == "127.0.0.1:8099"

    def test_default_port_is_omitted_from_the_host_header(self):
        target = validate_url("http://example.com/x", UrlPolicy())
        assert target.headers["Host"] == "example.com"

    def test_resolve_returns_every_address(self):
        addresses = resolve_host("localhost")
        assert addresses
        assert all(is_blocked_address(a) for a in addresses)

    def test_ip_literal_needs_no_dns(self):
        assert resolve_host("93.184.216.34") == [ipaddress.ip_address("93.184.216.34")]


def production_settings(**overrides):
    """A Settings object that satisfies the production safety validator.

    Production refuses to start on a placeholder secret or a test-fixture
    embedder, so those have to be supplied to reach the crawl policy at all.
    """
    from geolytics.config import Settings

    return Settings(
        env="production",
        secret_key="p" * 48,
        embedding_backend="sentence-transformers",
        **overrides,
    )


class TestPolicyFromSettings:
    def test_production_defaults_are_restrictive(self):
        policy = UrlPolicy.from_settings(production_settings())
        assert not policy.allow_private_addresses
        assert policy.allowed_ports == frozenset({80, 443, 8080, 8443})

    def test_restriction_is_lifted_only_by_an_explicit_flag(self):
        from geolytics.config import Settings

        policy = UrlPolicy.from_settings(
            Settings(env="development", crawl_restrict_ports=False)
        )
        assert policy.allowed_ports is None

    def test_an_emptied_port_list_does_not_mean_everything(self):
        """Clearing the list must not silently allow every port."""
        policy = UrlPolicy.from_settings(production_settings(crawl_allowed_ports=[]))
        assert policy.allowed_ports == frozenset()
        with pytest.raises(UrlPolicyError, match="port"):
            validate_url("http://example.com/", policy)
