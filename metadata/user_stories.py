from typing import List, Dict

from pydantic import BaseModel, Field


class API(BaseModel):
    method: str = Field(
        description=(
            "HTTP verb for the endpoint (e.g. GET / POST / PUT / DELETE).\n"
            "Semantic Intent: Identifies idempotency (e.g. GET should be side‑effect free) and helps tooling decide whether fuzzing mutations are safe.\n"
            "Significance: Core action classifier; drives expectations about side effects, caching, replay safety, and required protective controls (e.g. CSRF tokens for state‑changing verbs).\n"
            "Security Relevance: Non-idempotent verbs (POST/PUT/PATCH/DELETE) should trigger CSRF/authz scrutiny."
        )
    )
    path: str = Field(
        description=(
            "Canonical server-relative endpoint path, may contain placeholder tokens like :orgId or {memberId}.\n"
            "Significance: Encodes hierarchical resource scoping; placeholders usually map to tenant or authorization boundaries.\n"
            "Security Relevance: Path parameters are common IDOR vectors; ensure authorization invariants are applied per placeholder."
        )
    )
    headers: List[str] = Field(
        description=(
            "List of required or exemplar HTTP headers (one per entry). May include value templates like 'Authorization: Bearer <token>'.\n"
            "Significance: Defines contractual request surface—omissions can change server code paths or weaken protections (e.g. missing If-Match).\n"
            "Security Relevance: Absence or manipulation of security headers can lead to auth bypass, cache poisoning, or replay flaws."
        )
    )
    payload: Dict[str, str] | None = Field(
        description=(
            "Representative JSON body schema snippet. Empty when verb conventionally omits a body (e.g. GET).\n"
            "Significance: Establishes expected field names & types for validation, mutation testing, and comparison against responses.\n"
            "Security Relevance: Fields may be privilege sensitive (role, policy flags); improper server-side filtering exposes escalation risk."
        )
    )
    description: str | None = Field(
        description=(
            "Extended narrative: purpose, preconditions, postconditions, error taxonomy, invariants, threat assumptions.\n"
            "Significance: Canonical reference for generating acceptance & negative test assertions.\n"
            "Security Relevance: Explicit invariants provide anchors for detecting silent authorization or data isolation failures."
        )
    )


class Story(BaseModel):
    model_config = {"extra": "forbid"}
    goals: List[str] = Field(
        ...,
        description=(
            "User‑centric intent statements (WHAT is desired, not implementation details).\n"
            "Significance: High-level drivers for deriving concrete interaction sequences and prioritizing coverage.\n"
            "Security Relevance: Mismatch between articulated goal and actual outcome can reveal logic or authorization flaws."
        )
    )
    stories: List['Story'] = Field(
        description=(
            "Recursive sub‑stories decomposing a parent into finer scenarios / acceptance fragments.\n"
            "Significance: Supports hierarchical refinement—enables layered reasoning from broad capability to precise edge case.\n"
            "Security Relevance: Edge-focused sub-stories (revoked user, expired invite, race) expose boundary condition vulnerabilities."
        )
    )
    description: str = Field(
        description=(
            "Narrative plus acceptance criteria, pre/post conditions, expected side-effects and audit requirements.\n"
            "Significance: Direct source for assertion generation and regression guarding.\n"
            "Security Relevance: Provides explicit state transition guarantees whose violation indicates potential exploit paths."
        )
    )


class Persona(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(
        description=(
            "Role / persona label (business or operational identity).\n"
            "Significance: Context key for scoping permissible actions and expected visibility.\n"
            "Security Relevance: Divergence between persona capability and observed access suggests privilege creep or misconfiguration."
        )
    )
    goals: List[str] = Field(
        description=(
            "Strategic objectives motivating this persona's interactions.\n"
            "Significance: Guides prioritization of story execution order and risk-weighted exploration.\n"
            "Security Relevance: Unnecessary capabilities unrelated to goals may reveal excessive privilege."
        )
    )
    routes: List[str] = Field(
        description=(
            "Frontend route patterns legitimately accessed (dynamic segments allowed).\n"
            "Significance: Defines navigation baseline; deviations highlight discovery or potential enumeration phases.\n"
            "Security Relevance: Sensitive or admin routes require stricter enforcement & monitoring."
        )
    )
    apis: List[API] = Field(
        description=(
            "Subset of API endpoints this persona is expected to invoke. Not an exhaustive backend index.\n"
            "Significance: Establishes normative call graph for anomaly detection and coverage metrics.\n"
            "Security Relevance: Calls outside this set can indicate horizontal / vertical privilege probing."
        )
    )
    stories: List[Story] = Field(
        description=(
            "Root user stories owned by this persona (entry nodes of the narrative tree).\n"
            "Significance: Starting points for scenario expansion and orchestration planning.\n"
            "Security Relevance: Each root frames a boundary where access isolation should be verified."
        )
    )
    description: str = Field(
        description=(
            "Expanded operational context: responsibilities, constraints, workflow cadence, and hypothesized abuse angles.\n"
            "Significance: Supplies domain nuance for realistic test sequence shaping.\n"
            "Security Relevance: Highlights threat exposure zones needing deeper validation."
        )
    )


class UserStories(BaseModel):
    product: str = Field(
        description=(
            "Top-level product / tenant identifier (brand / service name).\n"
            "Significance: Unifies context across personas, APIs, and stories; useful as a scope token in prompts or logs.\n"
            "Security Relevance: Anchors multi-tenant separation; misuse could imply cross‑tenant bleed."
        )
    )
    personas: List[Persona] = Field(
        description=(
            "Ordered collection of modeled personas interacting with the product.\n"
            "Significance: Ensures breadth of perspective; drives comparative privilege analysis.\n"
            "Security Relevance: Missing high-impact persona may conceal untested critical attack paths."
        )
    )
    description: str = Field(
        description=(
            "Catalog overview: scope, rationale, embedded invariants and modeling assumptions.\n"
            "Significance: Single source of truth for initial reasoning warm‑up.\n"
            "Security Relevance: Documents non-negotiable guarantees whose violation elevates severity."
        )
    )


def user_stories() -> UserStories:
    """Return the static UserStories model for Halo CMS public + inferred admin personas.

    Source data provided by user (recon of https://blog.vackbot.com/). Transcribed verbatim
    into structured Pydantic models. This is intentionally read‑only and free of side effects.

    Returns:
        UserStories: Fully populated hierarchical user story tree.
    """
    # Helper to build Story recursively
    def S(goals: list[str], description: str, stories: list[dict] | None = None) -> dict:
        return {
            "goals": goals,
            "description": description,
            "stories": [S(**child) for child in stories] if stories else []
        }

    public_reader_stories = [
        {
            "goals": ["Consume public content safely"],
            "description": (
                "Reader navigates SSR pages and static content without authentication. "
                "Global invariants: no state-changing operations via GET; CSP and HSTS should protect HTML responses."
            ),
            "stories": [
                {
                    "goals": ["View homepage and latest posts"],
                    "description": (
                        "Preconditions: public access, no login. Acceptance: GET / returns 200 HTML with recent posts; "
                        "no authentication prompts; no mixed content. Security invariants: no overly permissive CORS needed on HTML; "
                        "any cookies (e.g., JSESSIONID) must be Secure/HttpOnly/SameSite. Evidence: flow e9e14b79-3091-42d9-b68d-730da2ccc56a."),
                    "stories": []
                },
                {
                    "goals": ["Read an individual article"],
                    "description": (
                        "Acceptance: GET /archives/:slug returns 200 HTML; code blocks render via Prism; content is safely escaped to prevent stored XSS. "
                        "Evidence: flow 3ceab052-253b-490a-b2d2-bedc33826ccb."),
                    "stories": []
                },
                {
                    "goals": ["Paginate through posts"],
                    "description": (
                        "Acceptance: GET /page/:n returns consistent listings without exposing drafts; caching works uniformly; no session personalization. "
                        "Evidence: flow d91a8910-5f01-4f97-807e-c34ae3f313b7."),
                    "stories": []
                },
                {
                    "goals": ["Browse archives and static pages"],
                    "description": (
                        "Acceptance: GET /archives and /s/:slug return 200; no admin/editor metadata in DOM; assets load with integrity. "
                        "Evidence: flows e79ff909-0a78-474b-b0d6-f39d9f20f7d4 and d7099b44-2b75-4dfe-83e1-f75e6bf2907d."),
                    "stories": []
                },
                {
                    "goals": ["Handle missing pages gracefully"],
                    "description": (
                        "Acceptance: GET unknown path returns themed 404 without stack traces; headers avoid unnecessary CORS echo. "
                        "Evidence: flow 23cb2b95-1926-4013-b50a-7448d9932138."),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Search site content"],
            "description": (
                "Reader uses server-side search. Security: rate-limit search; no sensitive cookies needed; avoid verbose errors."),
            "stories": [
                {
                    "goals": ["Submit keyword query"],
                    "description": (
                        "Acceptance: GET /search?keyword=:q responds with 2xx and results or 2xx empty state; for invalid q, 4xx; never 5xx. "
                        "Input is validated/escaped; reflected output is encoded. Evidence: flow 31fe8893-56ec-4584-a4c4-da086ae96e38 currently returns 500 for keyword=API."),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Subscribe and discover content"],
            "description": (
                "Syndication and discovery endpoints must not leak private content."),
            "stories": [
                {
                    "goals": ["Subscribe to Atom feed"],
                    "description": (
                        "Acceptance: GET /atom.xml returns 200 with recent public posts only; cacheable; no cookies required. Evidence: flow 473b9770-2ce1-4851-b197-6693584923dd."),
                    "stories": []
                },
                {
                    "goals": ["Inspect sitemap"],
                    "description": (
                        "Acceptance: GET /sitemap.xml enumerates only public routes; excludes any admin/console paths; timestamps are appropriate. "
                        "Evidence: flow 39093a76-fa3d-416f-b235-5e52abe2366e."),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Load assets securely and efficiently"],
            "description": (
                "Static assets should be immutable, cacheable, and safe from injection."),
            "stories": [
                {
                    "goals": ["Retrieve favicon without redirects"],
                    "description": (
                        "Acceptance: GET /favicon.ico returns a cacheable icon without 3xx; current behavior redirects to /upload logo (flow 30c811f9-e741-4915-9392-26bad82f858d)."),
                    "stories": []
                },
                {
                    "goals": ["Load theme JS/CSS and images"],
                    "description": (
                        "Acceptance: Assets under /themes/* and /upload/* served with correct MIME, long-lived caching, X-Content-Type-Options: nosniff, and ideally SRI for third-party libs (Prism). "
                        "Evidence: flows 8bbcabb6-9693-4630-92c2-a294ca9d78f2, 7edcd0c8-178d-4c96-a07c-60021e3e1f0b, 9dadbbd9-ccb0-4ad7-99f0-4b4d611edc0a."),
                    "stories": []
                }
            ]
        }
    ]

    site_operator_stories = [
        {
            "goals": ["Restrict and secure administrative/API surfaces"],
            "description": (
                "Admin persona inferred from platform (Halo) and CORS allowlists; no admin endpoints were observed in the captured traffic. "
                "Security focus: RBAC, CSRF, CORS, and patch hygiene."),
            "stories": [
                {
                    "goals": ["Ensure CORS is scoped to trusted origins and API paths only"],
                    "description": (
                        "Evidence across public responses shows Access-Control-Allow-Headers includes ADMIN-Authorization and API-Authorization with credentials allowed (flows e9e14b79-3091-42d9-b68d-730da2ccc56a and 39093a76-fa3d-416f-b235-5e52abe2366e). "
                        "Invariant: admin/console/API endpoints (e.g., Halo console/APIs) must not inherit permissive CORS for untrusted origins; enforce strict origin allowlist, CSRF, and authentication."),
                    "stories": []
                },
                {
                    "goals": ["Keep platform up-to-date and minimize version disclosure"],
                    "description": (
                        "Generator reveals Halo 1.5.3-SNAPSHOT on public pages (flow e9e14b79-3091-42d9-b68d-730da2ccc56a). Invariant: upgrade to supported stable releases; avoid exposing detailed version strings; review WAF/proxy header 'sec/9999' consistency."),
                    "stories": []
                }
            ]
        }
    ]

    public_reader_persona = {
        "name": "Public Reader",
        "goals": [
            "Read posts and static pages",
            "Search site content",
            "Subscribe to feeds",
            "Discover site structure via sitemap",
            "Load site assets reliably without security risk"
        ],
        "routes": [
            "/", "/archives", "/archives/:slug", "/page/:n", "/s/:slug", "/search?keyword=:q",
            "/sitemap.xml", "/atom.xml", "/favicon.ico", "/themes/*", "/upload/*"
        ],
        "apis": [
            {"path": "/", "method": "GET", "headers": ["Accept: text/html", "Cookie: JSESSIONID=<redacted> (optional; set by server)"], "payload": None, "description": "SSR homepage load; sets JSESSIONID and returns HTML (flow e9e14b79-3091-42d9-b68d-730da2ccc56a). CORS headers are broadly permissive on this document response."},
            {"path": "/archives", "method": "GET", "headers": ["Accept: text/html"], "payload": None, "description": "Archive index SSR page (flow e79ff909-0a78-474b-b0d6-f39d9f20f7d4)."},
            {"path": "/archives/:slug", "method": "GET", "headers": ["Accept: text/html"], "payload": None, "description": "Individual article SSR page (flow 3ceab052-253b-490a-b2d2-bedc33826ccb)."},
            {"path": "/page/:n", "method": "GET", "headers": ["Accept: text/html"], "payload": None, "description": "Pagination for homepage (flow d91a8910-5f01-4f97-807e-c34ae3f313b7)."},
            {"path": "/s/:slug", "method": "GET", "headers": ["Accept: text/html"], "payload": None, "description": "Static page rendering (flow d7099b44-2b75-4dfe-83e1-f75e6bf2907d)."},
            {"path": "/search", "method": "GET", "headers": ["Accept: text/html"], "payload": None, "description": "Server-side search via query param keyword=:q. Example /search?keyword=API returned 500 Internal Server Error (flow 31fe8893-56ec-4584-a4c4-da086ae96e38). Input validation and error handling required."},
            {"path": "/atom.xml", "method": "GET", "headers": ["Accept: application/atom+xml"], "payload": None, "description": "Atom feed for subscribers (flow 473b9770-2ce1-4851-b197-6693584923dd)."},
            {"path": "/sitemap.xml", "method": "GET", "headers": ["Accept: application/xml"], "payload": None, "description": "Sitemap enumerating archives, tags, categories (flow 39093a76-fa3d-416f-b235-5e52abe2366e)."},
            {"path": "/favicon.ico", "method": "GET", "headers": ["Accept: image/*"], "payload": None, "description": "Redirects (302) to uploaded logo under /upload/2022/08/logo.png (flow 30c811f9-e741-4915-9392-26bad82f858d)."},
            {"path": "/themes/*", "method": "GET", "headers": ["Accept: text/css, application/javascript"], "payload": None, "description": "Static theme assets (flows 8bbcabb6-9693-4630-92c2-a294ca9d78f2, 7edcd0c8-178d-4c96-a07c-60021e3e1f0b, 9dadbbd9-ccb0-4ad7-99f0-4b4d611edc0a)."},
            {"path": "/upload/*", "method": "GET", "headers": ["Accept: image/*"], "payload": None, "description": "Uploaded images and logo target of favicon redirect (flow 30c811f9-e741-4915-9392-26bad82f858d)."}
        ],
        "stories": [S(**s) for s in public_reader_stories],
        "description": (
            "Unauthenticated visitor consuming public SSR pages, feeds, and static assets. All observed traffic was GET. "
            "Cookies: server sets JSESSIONID on first load. CORS headers present on document responses are permissive (allowing credentials and admin/API headers), which is unnecessary for this persona."
        )
    }

    site_operator_persona = {
        "name": "Site Operator (inferred)",
        "goals": [
            "Maintain and publish content (admin/editor workflows)",
            "Protect administrative interfaces from public access",
            "Ensure API and console endpoints are origin-restricted and authenticated"
        ],
        "routes": [],
        "apis": [],  # Not observed in captured traffic
        "stories": [S(**s) for s in site_operator_stories],
        "description": (
            "Administrator/editor managing site content. Not observed in traffic; existence inferred via platform fingerprint (Halo) and CORS headers advertising ADMIN-Authorization and API-Authorization."
        )
    }

    data = {
        "product": "blog.vackbot.com (Halo CMS)",
        "personas": [public_reader_persona, site_operator_persona],
        "description": (
            "Scope: https://blog.vackbot.com/ (Halo CMS). Recon via Playwright with mitmproxy captured 30+ flows, all GET. Observed SSR documents: /, /archives, /archives/:slug, /page/:n, /s/:slug; feeds: /atom.xml; discovery: /sitemap.xml; static assets under /themes/* and /upload/*. "
            "An error on /search?keyword=API returned 500 (flow 31fe8893-56ec-4584-a4c4-da086ae96e38). Server sets JSESSIONID (Set-Cookie on /, flow e9e14b79-3091-42d9-b68d-730da2ccc56a). CORS headers are globally permissive with credentials allowed and admin/API headers listed. "
            "Third-party analytics (Baidu hm.js/hm.gif) executed in-browser but were not captured in the mitm whitelist. Security invariants: no state-changing operations for public users; cookies hardened (Secure/HttpOnly/SameSite); strict CSP/HSTS on HTML; scoped CORS for any actual APIs; robust error handling for search." 
        )
    }

    # Build and return the Pydantic model
    return UserStories(**data)


def scantist_v4_user_stories() -> UserStories:
    """Return the static UserStories model for Scantist v4 staging pre-auth flows.

    Source data from staging endpoint v4staging.scantist.io. Captures anonymous visitor,
    unverified user, SSO user, and adversarial actor personas. Focuses on pre-auth security
    posture including registration, login, captcha, password reset, and SSO initiation.

    Returns:
        UserStories: Fully populated hierarchical user story tree for Scantist v4 staging.
    """
    # Helper to build Story recursively
    def S(goals: list[str], description: str, stories: list[dict] | None = None) -> dict:
        return {
            "goals": goals,
            "description": description,
            "stories": [S(**child) for child in stories] if stories else []
        }

    anonymous_visitor_stories = [
        {
            "goals": ["Navigate to public SPA routes for authentication"],
            "description": "Public navigation baseline. Ensure client-side routing resilience does not mask genuine errors.",
            "stories": [
                {
                    "goals": ["Load /login even if server returns 404"],
                    "description": (
                        "As an Anonymous Visitor, I can GET /login and the SPA still renders the login module despite edge 404s "
                        "(flows 7d6de143-2601-4406-b4b2-628393402ea6, 7a4d7fa0-0cea-466a-b252-70a0bfb7bd8f). "
                        "Security invariant: no sensitive data in edge 404 bodies; secure headers present (X-Frame-Options: DENY, "
                        "X-Content-Type-Options: nosniff, COOP: same-origin)."
                    ),
                    "stories": []
                },
                {
                    "goals": ["View Terms & Conditions client-side despite server 404"],
                    "description": (
                        "GET /terms-and-conditions can 404 at the edge (aeb50295-6504-417b-82e8-6a56fafb5555) while SPA termsAndConditions.js "
                        "renders content. Ensure consistent 404 handling and no leakage."
                    ),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Register a local account to start onboarding"],
            "description": "Anonymous registration path gated by email verification.",
            "stories": [
                {
                    "goals": ["Submit registration JSON and receive verification prompt"],
                    "description": (
                        "POST /v2/rest-auth/registration/ with email and SHA-256 hashed passwords (flow 0703b0f1-f0f4-473f-8a3f-f190b0c5c404) "
                        "returns id, email, and detail '验证邮件已发送。'. Sets HttpOnly; Secure; SameSite=Lax sessionid. "
                        "Invariants: server must re-hash with salt; pre-verification session has minimal privileges; rate limiting in place."
                    ),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Prepare captcha for subsequent login attempts"],
            "description": "Captcha acquisition is unauthenticated; abuse controls required.",
            "stories": [
                {
                    "goals": ["Generate captcha image using verify_key"],
                    "description": (
                        "POST /v2/rest-auth/generate_verify_code/ returns base64 PNG (flows b8686dc0-972f-426a-b533-fe6b426ab529, "
                        "41ed1e47-df9c-4627-a5a5-56f2477d9f66). Invariants: bind verify_key to IP/session, enforce TTL and rate limits."
                    ),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Reset a forgotten password securely"],
            "description": "Anonymous password reset initiation.",
            "stories": [
                {
                    "goals": ["Trigger password reset email without revealing account existence"],
                    "description": (
                        "POST /v2/rest-auth/password/reset/ returns 201 with generic detail (flow 64b63543-1426-47bd-b38e-7437a30d7b97). "
                        "Invariants: JSON-only acceptance to mitigate CSRF; strict rate limiting."
                    ),
                    "stories": []
                }
            ]
        },
        {
            "goals": ["Discover SSO options"],
            "description": "Surface SSO provider metadata to drive UI.",
            "stories": [
                {
                    "goals": ["Fetch public integration configs"],
                    "description": (
                        "GET /v2/integration-configs/ lists providers and client_id (flows 5606d437-56e9-468e-8a4d-1e19df1ccae6, "
                        "911396b6-1a78-459d-a6c9-b8746026628d). Note: client_id is not secret; OIDC base_url uses ngrok in staging."
                    ),
                    "stories": []
                }
            ]
        }
    ]

    unverified_user_stories = [
        {
            "goals": ["Resend verification email without leaking registration status"],
            "description": "POST /v1/rest-auth/resend-confirmation/ responds generically (flow 103e2700-6bbb-42af-a354-f1023648b54b). Invariants: rate limit; consistent security headers; no user enumeration leakage.",
            "stories": []
        },
        {
            "goals": ["Attempt login while unverified"],
            "description": "Login gate enforces verification state and captcha.",
            "stories": [
                {
                    "goals": ["Ensure server validates captcha first and returns generic errors"],
                    "description": (
                        "Observed distinct message 'E-mail is not verified.' (flow 6c6f1454-5bad-4346-8b3b-9d99ef7abaa1). "
                        "Invariant: use generic failure to avoid verification-state disclosure; throttle attempts."
                    ),
                    "stories": []
                }
            ]
        }
    ]

    sso_user_stories = [
        {
            "goals": ["Obtain IdP authorization URL from backend"],
            "description": "SSO initiation must be robust and privacy-preserving.",
            "stories": [
                {
                    "goals": ["Reject malformed initiation with 4xx not 500"],
                    "description": (
                        "Observed 500 on POST /v2/rest-auth/managed-oidc/login/ (flow 26b1b4b1-b533-47d8-b0aa-bfb1088a5f3c) and frontend JSON parse error. "
                        "Harden validation and error handling; redact internals."
                    ),
                    "stories": []
                }
            ]
        }
    ]

    adversarial_actor_stories = [
        {
            "goals": ["Abuse captcha generation at scale"],
            "description": "Rapid POSTs to /v2/rest-auth/generate_verify_code/ should hit throttles and anomaly detection. Invariant: verify_key TTL and per-IP/session caps.",
            "stories": []
        },
        {
            "goals": ["Enumerate verification status via login responses"],
            "description": (
                "Compare wrong password vs unverified vs wrong captcha messages. Invariant: use generic errors to reduce signal; enforce backoff."
            ),
            "stories": []
        },
        {
            "goals": ["Test CSRF protections on cookie-bearing auth endpoints"],
            "description": (
                "Attempt non-JSON Content-Types (form/multipart) to ensure 400/415 rejection and no state changes. "
                "No explicit CSRF token observed; relies on SameSite and JSON-only acceptance."
            ),
            "stories": []
        }
    ]

    anonymous_visitor_persona = {
        "name": "Anonymous Visitor",
        "goals": [
            "Access the login screen and supporting public pages",
            "Register a local account",
            "Fetch captcha image required for login",
            "Reset a forgotten password",
            "Discover available SSO providers and attempt SSO initiation"
        ],
        "routes": ["/", "/login", "/terms-and-conditions", "/callback"],
        "apis": [
            {
                "path": "/v2/system/version/",
                "method": "GET",
                "headers": ["Accept: application/json", "Origin: https://v4staging.scantist.io"],
                "payload": None,
                "description": "API version/health probe (no auth). Observed 200 in flows 2d009da1-771e-4caf-bd59-6339b3f236a6, 4ea6f2e4-5c32-4d43-a0d0-da7c81bb933e."
            },
            {
                "path": "/v2/integration-configs/",
                "method": "GET",
                "headers": ["Accept: application/json", "Origin: https://v4staging.scantist.io"],
                "payload": None,
                "description": "Public SSO provider configuration discovery (github, gitlab, oidc). Observed 200 in flows 5606d437-56e9-468e-8a4d-1e19df1ccae6, 911396b6-1a78-459d-a6c9-b8746026628d, 43b34489-9e82-4f66-8fc4-886c56cf2a22."
            },
            {
                "path": "/v2/rest-auth/generate_verify_code/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io", "Accept-Language: en|cn"],
                "payload": {"verify_key": "uuid"},
                "description": "Unauthenticated captcha image generator; returns base64 PNG for a verify_key. Observed 200 in flows b8686dc0-972f-426a-b533-fe6b426ab529, d77db9f1-bf72-4e3e-b675-f0f3117e72b5, 22992488-77f1-4ef5-a166-ee680a356ff8."
            },
            {
                "path": "/v2/rest-auth/registration/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io", "Accept-Language: en|cn"],
                "payload": {"email": "string", "password1": "sha256(password)", "password2": "sha256(password)"},
                "description": "Create local account; sets minimal-privilege session cookie pre-verification. Observed 200 in flow 0703b0f1-f0f4-473f-8a3f-f190b0c5c404. Set-Cookie: sessionid; HttpOnly; Secure; SameSite=Lax."
            },
            {
                "path": "/v2/rest-auth/password/reset/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io", "Accept-Language: en|cn"],
                "payload": {"email": "string"},
                "description": "Initiate password reset; indistinguishable response for existence. Observed 201 in flow 64b63543-1426-47bd-b38e-7437a30d7b97."
            }
        ],
        "stories": [S(**s) for s in anonymous_visitor_stories],
        "description": (
            "Unregistered or logged-out user path. Security posture observed: CORS allowlist to SPA origin; HSTS max-age=60 (staging); "
            "X-Frame-Options DENY; COOP same-origin."
        )
    }

    unverified_user_persona = {
        "name": "Unverified Registered User",
        "goals": [
            "Resend email verification",
            "Attempt login (should be blocked until verified)"
        ],
        "routes": ["/login"],
        "apis": [
            {
                "path": "/v1/rest-auth/resend-confirmation/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io", "Accept-Language: en|cn"],
                "payload": {"email": "string"},
                "description": "Resend confirmation email. Observed 200 with messages cookie set (flow 103e2700-6bbb-42af-a354-f1023648b54b). API version drift (/v1 vs /v2)."
            },
            {
                "path": "/v2/rest-auth/login/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io", "Accept-Language: en|cn"],
                "payload": {"password": "sha256(password)", "username": "email", "verify_key": "uuid", "verify_code": "string"},
                "description": "Login attempt returns 400 'E-mail is not verified.' for unverified users (flow 6c6f1454-5bad-4346-8b3b-9d99ef7abaa1). Enumeration risk; captcha parameters present."
            }
        ],
        "stories": [S(**s) for s in unverified_user_stories],
        "description": (
            "User has an account but email not verified; interactions limited to verification and public endpoints."
        )
    }

    sso_user_persona = {
        "name": "SSO User (Managed OIDC)",
        "goals": [
            "Initiate OIDC login via managed provider",
            "Complete redirect-based authentication handshake"
        ],
        "routes": ["/login", "/callback"],
        "apis": [
            {
                "path": "/v2/rest-auth/managed-oidc/login/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io", "Accept-Language: cn|en"],
                "payload": {"prompt": "optional", "provider": "string (expected)", "redirect_uri": "string (expected)"},
                "description": "Initiates OIDC login but currently returns 500 with HTML body when invoked with malformed payload (flow 26b1b4b1-b533-47d8-b0aa-bfb1088a5f3c). Invariant: backend should 4xx with JSON error map on bad input."
            },
            {
                "path": "/v2/integration-configs/",
                "method": "GET",
                "headers": ["Accept: application/json", "Origin: https://v4staging.scantist.io"],
                "payload": None,
                "description": "Provider metadata (github, gitlab, oidc) used to configure SSO buttons. Observed 200 (e.g., 43b34489-9e82-4f66-8fc4-886c56cf2a22)."
            }
        ],
        "stories": [S(**s) for s in sso_user_stories],
        "description": (
            "Users leveraging Managed OIDC path; current staging misconfiguration yields server error on initiation."
        )
    }

    adversarial_actor_persona = {
        "name": "Abuse/Adversarial Actor",
        "goals": [
            "Probe rate limits and captcha abuse potential",
            "Attempt user/account state enumeration via error message differences",
            "Probe CSRF posture on cookie-auth endpoints"
        ],
        "routes": ["/login"],
        "apis": [
            {
                "path": "/v2/rest-auth/generate_verify_code/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io"],
                "payload": {"verify_key": "uuid"},
                "description": "Captcha generation is unauthenticated; target for high-volume abuse testing (flows b8686dc0-972f-426a-b533-fe6b426ab529, 97c5f81e-b423-4395-9131-942052cb71b4)."
            },
            {
                "path": "/v2/rest-auth/login/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io"],
                "payload": {"password": "sha256(password)", "username": "email", "verify_key": "uuid", "verify_code": "string"},
                "description": "Error semantics differ for unverified vs invalid; confirms enumeration vector (flow 6c6f1454-5bad-4346-8b3b-9d99ef7abaa1)."
            },
            {
                "path": "/v1/rest-auth/resend-confirmation/",
                "method": "POST",
                "headers": ["Content-Type: application/json", "Origin: https://v4staging.scantist.io"],
                "payload": {"email": "string"},
                "description": "Check for rate limiting and generic responses under /v1 namespace (flow 103e2700-6bbb-42af-a354-f1023648b54b)."
            }
        ],
        "stories": [S(**s) for s in adversarial_actor_stories],
        "description": (
            "Adversarial lens focused on throttling, enumeration, and CSRF posture pre-auth."
        )
    }

    data = {
        "product": "Scantist v4 (staging) — v4staging.scantist.io",
        "personas": [anonymous_visitor_persona, unverified_user_persona, sso_user_persona, adversarial_actor_persona],
        "description": (
            "Scope: Pre-auth flows on Scantist v4 staging SPA (v4staging.scantist.io) and API (api-v4staging.scantist.io). "
            "Evidence: 80+ flows across GET/POST/OPTIONS with status codes 200/201/400/404/500. Key endpoints: /v2/rest-auth/registration/, "
            "/v2/rest-auth/login/, /v2/rest-auth/generate_verify_code/, /v2/rest-auth/password/reset/, /v1/rest-auth/resend-confirmation/, "
            "/v2/rest-auth/managed-oidc/login/, and public GETs /v2/integration-configs/, /v2/system/version/. "
            "Security invariants: enforce JSON-only POSTs (no CSRF token observed), use HttpOnly; Secure; SameSite cookies, gate access by email verification "
            "without state disclosure, rate-limit unauth endpoints (registration/login/reset/resend/captcha), and return structured 4xx for OIDC initiation errors. "
            "Notable anomalies: 500 on managed OIDC initiation (flow 26b1b4b1-b533-47d8-b0aa-bfb1088a5f3c), SPA route 404s at edge for /login and /terms-and-conditions, "
            "and verification-state disclosure in login error (flow 6c6f1454-5bad-4346-8b3b-9d99ef7abaa1). RBAC and tenant APIs not observed yet; requires post-login exploration."
        )
    }

    # Build and return the Pydantic model
    return UserStories(**data)


