# Security model

What this system defends, how, and where the boundaries are. Read this before
deploying it anywhere a customer can reach.

---

## 1. The crawler is the main attack surface

GEOlytics accepts a URL from an untrusted customer and fetches it from inside
the production network. That is the textbook SSRF setup, and it is the single
most dangerous thing the product does. Everything in
`geolytics.crawl.guard` exists for it.

| Attack | Defence |
|---|---|
| `http://169.254.169.254/` → cloud IAM credentials | link-local and known metadata addresses refused |
| `http://localhost:6379/` → Redis, Postgres, Qdrant | loopback refused, ports restricted |
| `http://10.0.0.0/8` → internal network scanning | private ranges refused |
| `file://`, `gopher://`, `dict://` | scheme allowlist: http and https only |
| `http://2130706433/`, `::ffff:127.0.0.1` | every resolved address is checked, IPv4-mapped IPv6 unwrapped first |
| DNS rebinding (public at check, private at connect) | the request dials the **validated IP**, carrying the original Host header and TLS SNI |
| Public URL that 302s to the metadata endpoint | redirects followed manually, every hop revalidated, chain bounded |
| Multi-gigabyte response exhausting a worker | size cap counted over the stream, not trusted from `Content-Length` |

**Two settings disable this. Neither belongs in a multi-tenant deployment:**

- `GEOLYTICS_CRAWL_ALLOW_PRIVATE_ADDRESSES` — reopens every private range,
  metadata endpoint included. Production refuses to start with it on.
- `GEOLYTICS_CRAWL_RESTRICT_PORTS=false` — lifts the port allowlist.

For a self-hosted install that must audit one known internal host, use
`GEOLYTICS_CRAWL_ALLOWED_HOSTS` instead: it exempts that host only.

**Deploy the crawler away from anything worth stealing.** The worker is the
only component that makes outbound requests. Give it an egress-restricted
network and no instance role. The defences above are the second line, not the
only one.

## 2. Authentication

**Passwords** use Argon2id at the OWASP baseline (19 MiB, t=2, p=1), with
transparent rehash on login when the parameters are raised. Only length is
enforced (12–1024); composition rules push people toward predictable
substitutions and are no longer recommended. The upper bound is a
denial-of-service guard — Argon2 is deliberately expensive, and hashing a
multi-megabyte string from an unauthenticated endpoint burns a worker.

**Tokens** are JWTs with a pinned algorithm and required claims. Access and
refresh tokens are typed, so neither can be presented as the other. Access
tokens are short (15 minutes by default) and revocation is enforced on the
refresh path — a disabled account or a password change stops the refresh, so a
revoked session survives at most one access-token lifetime.

**API keys** are `gk_live_<lookup>.<secret>`. Only the secret's hash is stored.
That hash is SHA-256, not Argon2, and the difference is deliberate: an API key
is 256 bits of machine-generated entropy with no dictionary to attack, and a
slow hash on every API request would be a self-inflicted denial of service. A
password is low-entropy and human-chosen, which is why it gets Argon2.

**Enumeration.** Login answers identically for an unknown account and a wrong
password, and hashes a dummy password when the user does not exist so the two
take the same time. Signup returns a generic conflict rather than "that email
is registered".

## 3. Tenant isolation

Isolation is enforced in the query layer. The rule is absolute:

> The organisation comes from the authenticated `Principal`. Never from a path
> parameter, query string or request body — a caller controls all three.

Scoping goes in the `WHERE` clause, not a check after the fetch, so another
tenant's id reads as **404 rather than 403**. A 403 would confirm the row
exists, which is itself a leak.

`org_id` is denormalised onto `Site`, `Crawl`, `Audit` and `ExperimentRun` so
every scoped query filters one indexed column with no join. A missed join is
how cross-tenant reads happen.

Roles map to scopes. An API key's scopes are re-intersected with its creator's
role on **every request**, so a key can never exceed the member who made it,
and demoting that member narrows the key immediately. Actions that must be
taken by a person — inviting members, changing the plan — refuse API keys
outright, so a leaked key cannot escalate itself.

## 4. Configuration safety

Production refuses to start on any of these:

- a signing key under 32 characters, or one that looks like a placeholder
  (prefixes included — the string people actually commit is
  `change-me-in-production`)
- `*` in the CORS origins, on a credentialed API
- a crawler permitted to reach private addresses, or with ports unrestricted
- the `hashing` embedder, which is a test fixture with no semantics
- billing enabled with no webhook secret

An unset `GEOLYTICS_SECRET_KEY` generates a per-process key. Fine for a dev
restart; fatal across replicas, which would each sign with a different key and
reject the others' tokens. Production therefore requires it explicitly.

## 5. Billing webhooks

Stripe cannot hold a credential, so **the signature is the authentication**.
It is verified — with a timestamp tolerance against replay, and a constant-time
compare — before any of the body is read. Without that, the endpoint is a free
upgrade button for anyone who learns the URL.

Plan state comes from Stripe, never from the client: the checkout endpoint
takes a plan name only to pick a price id. Events are idempotent, and an
out-of-order event describing a superseded period cannot roll a customer back.

## 6. Rate limiting

A Redis token bucket, refill-and-take in a Lua script so it is atomic. Keyed on
a **hash** of the credential, so Redis never holds anything that works as a
token, and so one abusive key does not throttle everyone behind the same NAT.
Auth endpoints get a much tighter bucket, since that is where credential
stuffing lands.

It **fails open**: if Redis is unreachable, requests are allowed. For this
product that is the right trade — Redis being down should not take the API down
— but it means rate limiting is unavailable exactly when the system is already
unhealthy. The limiter reports the degraded state rather than hiding it.

## 7. The dashboard

Tokens live in **httpOnly cookies**, never `localStorage`. That is the whole
reason the dashboard proxies the API through its own route handlers instead of
calling it from the browser: a token readable by JavaScript is a token any XSS
can exfiltrate, and an access token is a bearer credential for an entire
organisation.

Cookies are `httpOnly`, `sameSite=lax` and `secure` in production. Lax rather
than strict, because strict drops the cookie on every inbound link and would
log people out whenever they arrive from an email; it still withholds the
cookie from cross-site POSTs, which is the CSRF case that matters.

## 8. What is NOT covered

Be honest about these before selling anything:

| Gap | Status |
|---|---|
| Email verification | not implemented — accounts are usable immediately |
| Password reset | not implemented — no mail transport is wired up |
| Member invitations | only existing accounts can be added; a real invite flow needs email |
| MFA / SSO | not implemented |
| Per-tenant encryption at rest | not implemented; rely on disk/volume encryption |
| Penetration test | never performed |
| Data export / deletion (GDPR) | no endpoints; deletion is manual SQL |
| Audit-log retention and export | rows are written, nothing prunes or exports them |

## 9. Reporting a vulnerability

Add a `SECURITY.md` with a contact address and a disclosure window before
taking paying customers. A product without one receives its vulnerability
reports on social media.
