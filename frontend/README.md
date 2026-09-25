# GEOlytics dashboard

Next.js 16 (App Router) front end for the GEOlytics API.

## How it talks to the API

The browser never calls the API directly. Every request goes through this
app's route handlers, which attach the session credential from an httpOnly
cookie:

```
browser ──▶ /api/proxy/<path> ──▶ GEOlytics API
                 │
                 └── reads geo_at / geo_rt cookies, refreshes on 401
```

That indirection is the point: a token in `localStorage` is a token any XSS can
exfiltrate, and an access token is a bearer credential for a whole
organisation. `GEOLYTICS_API_URL` is deliberately **not** a `NEXT_PUBLIC_`
variable.

Initial page data is loaded server-side (`src/lib/server-api.ts`) so there is
no client fetch waterfall and no loading flash. Client components handle only
mutations and the polling of in-flight audits.

## Running

```bash
npm install
cp .env.example .env.local
GEOLYTICS_API_URL=http://localhost:8000 npm run dev
```

```bash
npm run typecheck
npm run lint
npm run build
SHOT_DIR=./shots npm run e2e   # drives a real browser; see e2e/README.md
```

## Charts

`src/components/charts.tsx`. Forms follow the data's job rather than habit:

- **magnitude** (page scores, signal strength) → horizontal bars on one blue
  ordinal ramp
- **a ratio against a limit** (quota) → a meter, not a two-slice pie
- **a single headline value** (overall score) → a stat tile, not a one-bar chart
- **a signed difference with uncertainty** (experiment comparisons) → a dot at
  the estimate with its 95% interval, on an axis centred at zero. Bars would
  imply a magnitude from zero and hide the interval, which is the part that
  decides whether an effect exists at all.

Colours are roles defined once in `globals.css`, with dark values declared
under both the media query and the `data-theme` scope. The palette was checked
with the data-viz validator: the ordinal ramp is monotone in lightness and
clears the surface-contrast floor in both modes.

Every chart ships a table view, a legend is present whenever there are two or
more series, and status is always carried by an icon or label rather than
colour alone.
