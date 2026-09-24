"""Fixture HTML shared by the unit tests.

A module rather than a conftest attribute: two conftest.py files now exist
(one per test package) and a bare `import conftest` cannot say which it means.
"""

from __future__ import annotations

SAMPLE_HTML = """\
<html><head><title>Acme Plumbing</title>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"LocalBusiness","name":"Acme Plumbing",
 "address":{"@type":"PostalAddress","addressLocality":"Lahore"}}
</script></head>
<body>
<nav>Home Services Contact</nav>
<h1>Acme Plumbing</h1>
<p>Acme Plumbing has served Lahore since 2004.</p>
<h2>Emergency callouts</h2>
<p>Acme dispatches a plumber within 90 minutes, 24 hours a day.
According to a 2024 customer survey, 82 percent of emergency jobs were resolved on
the first visit.</p>
<h3>Callout pricing</h3>
<p>An emergency callout costs 2,500 PKR. It includes the first hour of labour.</p>
<h2>Bathroom installation</h2>
<p>A full bathroom installation takes four to six working days.
Acme sources fittings from three certified suppliers.</p>
<footer>Copyright 2024 Acme Plumbing</footer>
</body></html>
"""

