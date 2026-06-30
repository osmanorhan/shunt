//! Capability tasks — graded, self-contained fixtures with on-disk ground truth.
//!
//! All tasks in this suite are discriminative: 4B models struggle, 12B models pass.
//! Requires reasoning + full code production (50-200 new lines), checked structurally.
//! **To add a task, append one `CapabilityTask` to `suite()`** — the runner picks it up.

use std::fs;

use tempfile::TempDir;

use crate::fixtures::Workspace;

pub type ContentCheck = (&'static str, fn(&str) -> bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Difficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
}

impl Difficulty {
    pub fn label(self) -> &'static str {
        match self {
            Difficulty::Trivial => "trivial",
            Difficulty::Easy => "easy",
            Difficulty::Medium => "medium",
            Difficulty::Hard => "hard",
        }
    }
}

pub struct CapabilityTask {
    pub name: &'static str,
    pub difficulty: Difficulty,
    pub request: &'static str,
    pub files: &'static [(&'static str, &'static str)],
    pub checks: &'static [ContentCheck],
}

impl CapabilityTask {
    pub fn workspace(&self) -> Workspace {
        let dir = TempDir::new().expect("tempdir");
        for (rel, contents) in self.files {
            let path = dir.path().join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("mkdir");
            }
            fs::write(&path, contents).expect("write fixture");
        }
        Workspace { dir }
    }

    pub fn full_request(&self) -> String {
        self.request.to_string()
    }

    pub fn passed(&self, ws: &Workspace) -> bool {
        self.checks.iter().all(|(rel, check)| {
            let content = fs::read_to_string(ws.root().join(rel)).unwrap_or_default();
            check(&content)
        })
    }
}

// ── Suite ─────────────────────────────────────────────────────────────────────

pub fn suite() -> Vec<CapabilityTask> {
    vec![
        // ── Structural / refactor tasks ────────────────────────────────────────
        CapabilityTask {
            name: "thundering_herd",
            difficulty: Difficulty::Hard,
            request: "The getData function in src/cache/data.ts has a thundering-herd \
                      problem: if 10 concurrent requests arrive for the same uncached key, \
                      all 10 call fetchData in parallel, hammering the backend. Fix it by \
                      keeping an `inFlight` Map of key → Promise so that concurrent requests \
                      for the same key share a single in-flight fetch. Once the fetch resolves \
                      the result goes into cache and the inFlight entry is removed.",
            files: &[
                (
                    "src/cache/data.ts",
                    "const cache = new Map<string, any>();\n\n\
                     async function fetchData(key: string): Promise<any> {\n  \
                     const res = await fetch(`/api/data/${key}`);\n  \
                     if (!res.ok) throw new Error(`fetch failed: ${res.status}`);\n  \
                     return res.json();\n}\n\n\
                     export async function getData(key: string): Promise<any> {\n  \
                     if (cache.has(key)) {\n    return cache.get(key);\n  }\n  \
                     const value = await fetchData(key);\n  \
                     cache.set(key, value);\n  \
                     return value;\n}\n",
                ),
                (
                    "src/routes/data.ts",
                    "import { Router } from 'express';\n\
                     import { getData } from '../cache/data';\n\n\
                     const router = Router();\n\n\
                     router.get('/data/:key', async (req, res, next) => {\n  \
                     try {\n    \
                     const value = await getData(req.params.key);\n    \
                     res.json({ value });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
            ],
            checks: &[("src/cache/data.ts", |c| {
                (c.contains("inFlight") || c.contains("pending") || c.contains("inflight"))
                    && c.contains("Promise")
                    && c.contains("cache.set")
            })],
        },
        CapabilityTask {
            name: "cra_to_vite",
            difficulty: Difficulty::Hard,
            request: "Migrate this project from Create React App to Vite. \
                      Update package.json: replace react-scripts with vite and \
                      @vitejs/plugin-react (devDependencies), update scripts to use \
                      vite/vite build/vitest. Create vite.config.ts using defineConfig \
                      with the react() plugin. Move public/index.html to the project root, \
                      remove %PUBLIC_URL% tokens, and add a <script type=\"module\" src=\"/src/main.tsx\"> \
                      tag. The entry point src/main.tsx should stay as-is.",
            files: &[
                (
                    "package.json",
                    "{\n  \"name\": \"my-app\",\n  \"version\": \"0.1.0\",\n  \
                     \"private\": true,\n  \
                     \"dependencies\": {\n    \
                     \"react\": \"^18.2.0\",\n    \
                     \"react-dom\": \"^18.2.0\",\n    \
                     \"react-scripts\": \"5.0.1\"\n  },\n  \
                     \"scripts\": {\n    \
                     \"start\": \"react-scripts start\",\n    \
                     \"build\": \"react-scripts build\",\n    \
                     \"test\": \"react-scripts test\"\n  }\n}\n",
                ),
                (
                    "public/index.html",
                    "<!DOCTYPE html>\n<html lang=\"en\">\n  <head>\n    \
                     <meta charset=\"utf-8\" />\n    \
                     <link rel=\"icon\" href=\"%PUBLIC_URL%/favicon.ico\" />\n    \
                     <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n    \
                     <title>React App</title>\n  </head>\n  <body>\n    \
                     <div id=\"root\"></div>\n  </body>\n</html>\n",
                ),
                (
                    "src/main.tsx",
                    "import React from 'react';\n\
                     import ReactDOM from 'react-dom/client';\n\
                     import App from './App';\n\n\
                     const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);\n\
                     root.render(<App />);\n",
                ),
                (
                    "src/App.tsx",
                    "import React from 'react';\n\n\
                     export default function App() {\n  \
                     return <h1>Hello</h1>;\n}\n",
                ),
            ],
            checks: &[
                ("package.json", |c| {
                    c.contains("vite") && c.contains("@vitejs/plugin-react") && !c.contains("react-scripts")
                }),
                ("vite.config.ts", |c| {
                    c.contains("defineConfig") && c.contains("react()")
                }),
                ("index.html", |c| {
                    !c.contains("%PUBLIC_URL%") && c.contains("type=\"module\"")
                }),
            ],
        },
        CapabilityTask {
            name: "extract_auth_service",
            difficulty: Difficulty::Hard,
            request: "UserService in src/services/user.ts has grown into a god class mixing \
                      CRUD with authentication. Extract login, logout, and verifyToken into a \
                      new class AuthService in src/services/auth.ts. Update src/routes/auth.ts \
                      to import from the new AuthService instead of userService. Leave the \
                      user-management methods (findById, updateEmail, deleteUser) in UserService.",
            files: &[
                (
                    "src/services/user.ts",
                    "import { db } from '../db';\n\
                     import { mailer } from '../mailer';\n\
                     import jwt from 'jsonwebtoken';\n\n\
                     const JWT_SECRET = process.env.JWT_SECRET ?? 'secret';\n\n\
                     export class UserService {\n  \
                     async login(email: string, password: string): Promise<string> {\n    \
                     const user = await db.users.findUnique({ where: { email } });\n    \
                     if (!user || user.password !== password) throw new Error('invalid credentials');\n    \
                     return jwt.sign({ userId: user.id }, JWT_SECRET);\n  }\n\n  \
                     async logout(token: string): Promise<void> {\n    \
                     await db.tokens.delete({ where: { token } });\n  }\n\n  \
                     async verifyToken(token: string): Promise<{ userId: string }> {\n    \
                     return jwt.verify(token, JWT_SECRET) as { userId: string };\n  }\n\n  \
                     async findById(id: string) {\n    \
                     return db.users.findUnique({ where: { id } });\n  }\n\n  \
                     async updateEmail(id: string, email: string) {\n    \
                     return db.users.update({ where: { id }, data: { email } });\n  }\n\n  \
                     async deleteUser(id: string) {\n    \
                     await db.users.delete({ where: { id } });\n    \
                     await mailer.sendGoodbyeEmail(id);\n  }\n}\n\n\
                     export const userService = new UserService();\n",
                ),
                (
                    "src/routes/auth.ts",
                    "import { Router } from 'express';\n\
                     import { userService } from '../services/user';\n\n\
                     const router = Router();\n\n\
                     router.post('/login', async (req, res, next) => {\n  \
                     try {\n    \
                     const token = await userService.login(req.body.email, req.body.password);\n    \
                     res.json({ token });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     router.post('/logout', async (req, res, next) => {\n  \
                     try {\n    \
                     await userService.logout(req.headers.authorization!);\n    \
                     res.json({ ok: true });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     update: async (_q: any) => ({} as any),\n    \
                     delete: async (_q: any) => {},\n  },\n  \
                     tokens: { delete: async (_q: any) => {} },\n};\n",
                ),
                (
                    "src/mailer.ts",
                    "export const mailer = {\n  \
                     sendGoodbyeEmail: async (_id: string) => {},\n};\n",
                ),
            ],
            checks: &[
                ("src/services/auth.ts", |c| {
                    c.contains("AuthService") && c.contains("login(") && c.contains("verifyToken(")
                }),
                ("src/services/user.ts", |c| {
                    !c.contains("async login(") && !c.contains("async verifyToken(")
                }),
                ("src/routes/auth.ts", |c| {
                    (c.contains("services/auth") || c.contains("authService")) && !c.contains("userService.login")
                }),
            ],
        },
        CapabilityTask {
            name: "callback_to_async",
            difficulty: Difficulty::Medium,
            request: "Three utility modules in src/utils/ still use the old Node.js callback \
                      style for fs.readFile. Convert all three — config.js, templates.js, and \
                      assets.js — to async/await using fs.promises.readFile (or the \
                      'node:fs/promises' import). The function signatures should become async \
                      functions that return Promises; remove the callback parameters entirely.",
            files: &[
                (
                    "src/utils/config.js",
                    "import fs from 'node:fs';\n\
                     import path from 'node:path';\n\n\
                     export function loadConfig(filename, callback) {\n  \
                     fs.readFile(path.join(process.cwd(), filename), 'utf8', (err, data) => {\n    \
                     if (err) return callback(err);\n    \
                     callback(null, JSON.parse(data));\n  });\n}\n",
                ),
                (
                    "src/utils/templates.js",
                    "import fs from 'node:fs';\n\
                     import path from 'node:path';\n\n\
                     export function loadTemplate(name, callback) {\n  \
                     const file = path.join(process.cwd(), 'templates', name + '.html');\n  \
                     fs.readFile(file, 'utf8', (err, data) => {\n    \
                     if (err) return callback(err);\n    \
                     callback(null, data.trim());\n  });\n}\n",
                ),
                (
                    "src/utils/assets.js",
                    "import fs from 'node:fs';\n\n\
                     export function readAsset(filePath, callback) {\n  \
                     fs.readFile(filePath, 'utf8', (err, data) => {\n    \
                     if (err) return callback(err);\n    \
                     callback(null, data);\n  });\n}\n",
                ),
            ],
            checks: &[
                ("src/utils/config.js", |c| {
                    c.contains("async") && c.contains("await") && !c.contains("(err, data)")
                }),
                ("src/utils/templates.js", |c| {
                    c.contains("async") && c.contains("await") && !c.contains("(err, data)")
                }),
                ("src/utils/assets.js", |c| {
                    c.contains("async") && c.contains("await") && !c.contains("(err, data)")
                }),
            ],
        },
        CapabilityTask {
            name: "node_health_route_smoke",
            difficulty: Difficulty::Hard,
            request: "In this small Node service, add a GET /healthz endpoint that returns \
                      JSON with `{ ok: true, version: process.env.APP_VERSION ?? 'dev' }`, \
                      keep the existing /users/:id route working, and add a new smoke script \
                      at scripts/smoke-health.sh that curls the health endpoint.",
            files: &[
                (
                    "package.json",
                    "{\n  \"name\": \"mini-service\",\n  \"version\": \"1.0.0\",\n  \
                     \"type\": \"module\",\n  \
                     \"scripts\": {\n    \"start\": \"node src/server.js\"\n  }\n}\n",
                ),
                (
                    "src/server.js",
                    "import http from 'node:http';\n\n\
                     function json(res, status, body) {\n  \
                     res.writeHead(status, { 'content-type': 'application/json' });\n  \
                     res.end(JSON.stringify(body));\n}\n\n\
                     const server = http.createServer((req, res) => {\n  \
                     if (!req.url) {\n    \
                     return json(res, 400, { error: 'missing_url' });\n  }\n\n  \
                     if (req.url.startsWith('/users/')) {\n    \
                     const id = req.url.slice('/users/'.length);\n    \
                     return json(res, 200, { id, role: 'member' });\n  }\n\n  \
                     return json(res, 404, { error: 'not_found' });\n});\n\n\
                     server.listen(process.env.PORT ?? 3000);\n",
                ),
            ],
            checks: &[
                ("src/server.js", |c| {
                    c.contains("/healthz")
                        && c.contains("APP_VERSION")
                        && c.contains("ok")
                        && c.contains("/users/")
                }),
                ("scripts/smoke-health.sh", |c| {
                    c.contains("curl") && c.contains("healthz")
                }),
            ],
        },
        // ── Production / reasoning tasks ───────────────────────────────────────
        CapabilityTask {
            name: "implement_cursor_pagination",
            difficulty: Difficulty::Hard,
            request: "The GET /items endpoint in src/routes/items.ts uses OFFSET pagination \
                      which becomes slow on large tables and returns duplicate or skipped rows \
                      under concurrent inserts. Replace it with cursor-based pagination: \
                      accept an optional `cursor` query param (opaque string — base64-encode \
                      the last item id), use WHERE id > decodedId in the query instead of \
                      skip/offset, and return a `nextCursor` field in the response \
                      (null when no more pages).",
            files: &[
                (
                    "src/routes/items.ts",
                    "import { Router } from 'express';\n\
                     import { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.get('/items', async (req, res, next) => {\n  \
                     try {\n    \
                     const page  = parseInt(req.query.page  as string) || 1;\n    \
                     const limit = Math.min(parseInt(req.query.limit as string) || 20, 100);\n    \
                     const skip  = (page - 1) * limit;\n    \
                     const items = await db.items.findMany({\n      \
                     skip,\n      take: limit,\n      orderBy: { id: 'asc' },\n    });\n    \
                     res.json({ items, page, limit });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     items: { findMany: async (_q: any) => [] as any[] },\n};\n",
                ),
            ],
            checks: &[("src/routes/items.ts", |c| {
                let encodes = c.contains("base64") || c.contains("btoa(")
                    || (c.contains("Buffer.from(") && c.contains("toString("));
                let returns_next = c.contains("nextCursor") || c.contains("next_cursor");
                let no_offset = !(c.contains("page - 1") || c.contains("page-1"));
                (encodes || returns_next) && no_offset
            })],
        },
        CapabilityTask {
            name: "implement_rate_limiter",
            difficulty: Difficulty::Hard,
            request: "The rate-limit middleware in src/middleware/rateLimit.ts is a stub that \
                      calls next() unconditionally. Implement a sliding-window rate limiter \
                      using the Redis client in src/redis.ts: allow 100 requests per minute \
                      per IP. Use Redis INCR + EXPIRE (or a sorted set) to track the window. \
                      Return 429 with a Retry-After header when the limit is exceeded. \
                      Do NOT use an in-memory Map — it would not work across multiple instances. \
                      Wire the middleware into src/app.ts before the API routes.",
            files: &[
                (
                    "src/middleware/rateLimit.ts",
                    "import { Request, Response, NextFunction } from 'express';\n\n\
                     export function rateLimit(req: Request, res: Response, next: NextFunction): void {\n  \
                     next();\n}\n",
                ),
                (
                    "src/app.ts",
                    "import express from 'express';\n\
                     import apiRouter from './routes/api';\n\n\
                     const app = express();\n\
                     app.use(express.json());\n\
                     app.use('/api', apiRouter);\n\n\
                     export default app;\n",
                ),
                (
                    "src/redis.ts",
                    "export const redis = {\n  \
                     get: async (_key: string) => null as string | null,\n  \
                     set: async (_key: string, _value: string) => {},\n  \
                     incr: async (_key: string) => 1,\n  \
                     expire: async (_key: string, _secs: number) => {},\n  \
                     zadd: async (_key: string, _score: number, _member: string) => 0,\n  \
                     zremrangebyscore: async (_key: string, _min: string, _max: number) => 0,\n  \
                     zcard: async (_key: string) => 0,\n};\n",
                ),
                (
                    "src/routes/api.ts",
                    "import { Router } from 'express';\n\n\
                     const router = Router();\n\n\
                     router.get('/hello', (_req, res) => res.json({ message: 'hello' }));\n\n\
                     export default router;\n",
                ),
            ],
            checks: &[
                ("src/middleware/rateLimit.ts", |c| {
                    (c.contains("redis") || c.contains("Redis"))
                        && (c.contains("incr") || c.contains("zadd") || c.contains("zincrby"))
                        && c.contains("429")
                        && (c.contains("Retry-After") || c.contains("retry-after"))
                        && !c.contains("new Map()")
                }),
                ("src/app.ts", |c| c.contains("rateLimit")),
            ],
        },
        CapabilityTask {
            name: "implement_hmac_webhook",
            difficulty: Difficulty::Hard,
            request: "Webhook delivery in src/webhooks/deliver.ts sends POST requests without \
                      any signature, and the verify stub in src/webhooks/verify.ts always returns \
                      false. Fix both: (1) In deliver.ts, compute an HMAC-SHA256 signature over \
                      the JSON body using the secret from process.env.WEBHOOK_SECRET, and attach \
                      it as the X-Signature-256 header in the format `sha256=<hex>`. \
                      (2) In verify.ts, recompute the expected signature and compare with \
                      crypto.timingSafeEqual to prevent timing attacks. Never use === for \
                      signature comparison.",
            files: &[
                (
                    "src/webhooks/deliver.ts",
                    "import { db } from '../db';\n\n\
                     export async function deliverWebhook(\n  \
                     url: string,\n  eventType: string,\n  payload: unknown,\n): Promise<void> {\n  \
                     const body = JSON.stringify({ event: eventType, data: payload });\n  \
                     const response = await fetch(url, {\n    method: 'POST',\n    \
                     headers: { 'Content-Type': 'application/json' },\n    body,\n  });\n  \
                     if (!response.ok) {\n    \
                     throw new Error(`Webhook delivery failed: ${response.status}`);\n  }\n  \
                     await db.webhookEvents.create({ data: { url, eventType, deliveredAt: new Date() } });\n}\n",
                ),
                (
                    "src/webhooks/verify.ts",
                    "export function verifyWebhookSignature(\n  \
                     payload: string,\n  signature: string,\n  secret: string,\n): boolean {\n  \
                     return false;\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     webhookEvents: { create: async (_data: any) => ({}) },\n};\n",
                ),
            ],
            checks: &[
                ("src/webhooks/deliver.ts", |c| {
                    c.contains("createHmac")
                        && c.contains("sha256")
                        && (c.contains("X-Signature-256") || c.contains("x-signature-256"))
                        && c.contains("sha256=")
                }),
                ("src/webhooks/verify.ts", |c| {
                    c.contains("createHmac")
                        && c.contains("sha256")
                        && c.contains("timingSafeEqual")
                        && !c.contains("=== sig")
                        && !c.contains("== sig")
                }),
            ],
        },
        CapabilityTask {
            name: "write_auth_tests",
            difficulty: Difficulty::Hard,
            request: "Write comprehensive Jest unit tests for AuthService in \
                      src/services/auth.ts. Create src/services/auth.test.ts with at least \
                      5 tests covering: login with valid credentials returns a token; \
                      login with wrong password throws; verifyToken with a valid token \
                      returns the payload; refreshToken with a fresh token returns a new token; \
                      refreshToken with an already-used token throws. Use jest.mock to mock \
                      the db module so no real database is needed.",
            files: &[
                (
                    "src/services/auth.ts",
                    "import { db } from '../db';\n\
                     import jwt from 'jsonwebtoken';\n\n\
                     const JWT_SECRET = process.env.JWT_SECRET ?? 'test-secret';\n\n\
                     export class AuthService {\n  \
                     async login(email: string, password: string): Promise<string> {\n    \
                     const user = await db.users.findUnique({ where: { email } });\n    \
                     if (!user || user.password !== password) throw new Error('wrong password');\n    \
                     return jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: '1h' });\n  }\n\n  \
                     async verifyToken(token: string): Promise<{ userId: string }> {\n    \
                     return jwt.verify(token, JWT_SECRET) as { userId: string };\n  }\n\n  \
                     async refreshToken(token: string): Promise<string> {\n    \
                     const stored = await db.refreshTokens.findUnique({ where: { token } });\n    \
                     if (!stored) throw new Error('token not found');\n    \
                     if (stored.used) throw new Error('refresh token already used');\n    \
                     await db.refreshTokens.update({ where: { token }, data: { used: true } });\n    \
                     return jwt.sign({ userId: stored.userId }, JWT_SECRET, { expiresIn: '1h' });\n  }\n\n  \
                     async logout(userId: string): Promise<void> {\n    \
                     await db.sessions.deleteMany({ where: { userId } });\n  }\n}\n\n\
                     export const authService = new AuthService();\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: { findUnique: async (_q: any) => null as any },\n  \
                     refreshTokens: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     update: async (_q: any) => ({} as any),\n  },\n  \
                     sessions: { deleteMany: async (_q: any) => ({}) },\n};\n",
                ),
            ],
            checks: &[("src/services/auth.test.ts", |c| {
                let count = c.matches("it(").count() + c.matches("test(").count();
                c.contains("describe")
                    && (c.contains("wrong") || c.contains("invalid") || c.contains("incorrect"))
                    && (c.contains("used") || c.contains("refresh"))
                    && (c.contains("jest.mock") || c.contains("mockResolvedValue"))
                    && count >= 5
            })],
        },
        CapabilityTask {
            name: "implement_retry_backoff",
            difficulty: Difficulty::Medium,
            request: "The HTTP client in src/http/client.ts fails immediately on any error. \
                      Add exponential back-off retry: retry up to 3 times on 5xx responses \
                      (do NOT retry on 4xx — those are client errors and retrying is pointless). \
                      Use delays of 2^attempt * 100ms between retries. After 3 failed attempts, \
                      throw the last error.",
            files: &[(
                "src/http/client.ts",
                "export async function httpGet(url: string): Promise<Response> {\n  \
                 const res = await fetch(url);\n  \
                 if (!res.ok) {\n    \
                 throw new Error(`HTTP ${res.status}: ${url}`);\n  }\n  \
                 return res;\n}\n\n\
                 export async function httpPost(url: string, body: unknown): Promise<Response> {\n  \
                 const res = await fetch(url, {\n    method: 'POST',\n    \
                 headers: { 'Content-Type': 'application/json' },\n    \
                 body: JSON.stringify(body),\n  });\n  \
                 if (!res.ok) {\n    \
                 throw new Error(`HTTP ${res.status}: ${url}`);\n  }\n  \
                 return res;\n}\n",
            )],
            checks: &[("src/http/client.ts", |c| {
                let has_retry = c.contains("attempt") || c.contains("retry") || c.contains("retries");
                let has_backoff = (c.contains("2 **") || c.contains("2**") || c.contains("Math.pow(2,"))
                    && (c.contains("setTimeout") || c.contains("sleep") || c.contains("delay") || c.contains("await new Promise"));
                let guards_5xx = c.contains(">= 500") || c.contains("status >= 500") || c.contains("res.status >= 5");
                has_retry && has_backoff && guards_5xx
            })],
        },
        CapabilityTask {
            name: "thread_correlation_id",
            difficulty: Difficulty::Hard,
            request: "Thread a correlation ID through every request. \
                      (1) In src/middleware/correlationId.ts: generate a UUID with \
                      crypto.randomUUID(), attach it to req as (req as any).correlationId, \
                      and set the X-Correlation-Id response header. \
                      (2) In src/services/order.ts: pass correlationId to every logger call \
                      so it appears in log output. \
                      (3) In src/middleware/errorHandler.ts: include correlationId in the \
                      error log entry.",
            files: &[
                (
                    "src/middleware/correlationId.ts",
                    "import { Request, Response, NextFunction } from 'express';\n\n\
                     export function correlationIdMiddleware(\n  \
                     req: Request, res: Response, next: NextFunction,\n): void {\n  \
                     next();\n}\n",
                ),
                (
                    "src/services/order.ts",
                    "import { logger } from '../logger';\n\
                     import { db } from '../db';\n\n\
                     export async function createOrder(userId: string, items: string[]) {\n  \
                     logger.info('creating order', { userId });\n  \
                     const order = await db.orders.create({ data: { userId, items, status: 'pending' } });\n  \
                     logger.info('order created', { orderId: order.id });\n  \
                     return order;\n}\n\n\
                     export async function cancelOrder(orderId: string) {\n  \
                     logger.info('cancelling order', { orderId });\n  \
                     await db.orders.update({ where: { id: orderId }, data: { status: 'cancelled' } });\n  \
                     logger.info('order cancelled', { orderId });\n}\n",
                ),
                (
                    "src/middleware/errorHandler.ts",
                    "import { Request, Response, NextFunction } from 'express';\n\
                     import { logger } from '../logger';\n\n\
                     export function errorHandler(\n  \
                     err: Error, req: Request, res: Response, _next: NextFunction,\n): void {\n  \
                     logger.error('unhandled error', { message: err.message });\n  \
                     res.status(500).json({ error: 'internal_server_error' });\n}\n",
                ),
                (
                    "src/logger.ts",
                    "export const logger = {\n  \
                     info: (msg: string, meta?: object) => \
                     console.log(JSON.stringify({ level: 'info', msg, ...meta })),\n  \
                     error: (msg: string, meta?: object) => \
                     console.error(JSON.stringify({ level: 'error', msg, ...meta })),\n};\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     orders: {\n    \
                     create: async (_data: any) => ({ id: 'order-1' }),\n    \
                     update: async (_data: any) => ({}),\n  },\n};\n",
                ),
            ],
            checks: &[
                ("src/middleware/correlationId.ts", |c| {
                    (c.contains("randomUUID") || c.contains("uuid") || c.contains("crypto"))
                        && c.contains("correlationId")
                        && (c.contains("X-Correlation-Id") || c.contains("x-correlation-id"))
                }),
                ("src/services/order.ts", |c| {
                    c.contains("correlationId") && c.contains("logger")
                }),
                ("src/middleware/errorHandler.ts", |c| c.contains("correlationId")),
            ],
        },
        CapabilityTask {
            name: "implement_optimistic_lock",
            difficulty: Difficulty::Hard,
            request: "updateDocument in src/documents/repository.ts overwrites any concurrent \
                      edit silently. Add optimistic locking: accept a `version` parameter, \
                      use db.documents.updateMany with a WHERE clause that matches both the \
                      document id and the current version number (WHERE id = id AND version = version), \
                      and if the update count is 0 throw an error with HTTP status 409 — it means \
                      the document was modified concurrently and the client should re-fetch.",
            files: &[
                (
                    "src/documents/repository.ts",
                    "import { db } from '../db';\n\n\
                     export interface Document {\n  id: string;\n  title: string;\n  \
                     content: string;\n  version: number;\n}\n\n\
                     export async function getDocument(id: string): Promise<Document | null> {\n  \
                     return db.documents.findUnique({ where: { id } }) as Promise<Document | null>;\n}\n\n\
                     export async function updateDocument(\n  \
                     id: string, title: string, content: string,\n): Promise<Document> {\n  \
                     return db.documents.update({\n    where: { id },\n    \
                     data: { title, content, version: { increment: 1 } },\n  }) as Promise<Document>;\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     documents: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     update: async (_q: any) => ({} as any),\n    \
                     updateMany: async (_q: any) => ({ count: 0 }),\n  },\n};\n",
                ),
            ],
            checks: &[("src/documents/repository.ts", |c| {
                let accepts_version = c.contains("version:") || c.contains(", version)")
                    || c.contains(", version: number");
                let uses_lock = c.contains("updateMany") && c.contains("version");
                let throws_conflict = c.contains("409") || c.contains("Conflict") || c.contains("stale");
                accepts_version && uses_lock && throws_conflict
            })],
        },
        CapabilityTask {
            name: "add_zod_validation",
            difficulty: Difficulty::Medium,
            request: "The POST /users route in src/routes/users.ts uses manual if-checks to \
                      validate name, email, and age. Replace all of it with Zod: define a \
                      z.object() schema, use .safeParse() on req.body, return 422 (not 400) \
                      on validation failures, and include the Zod error details in the response \
                      body. Remove the manual if-checks entirely.",
            files: &[
                (
                    "src/routes/users.ts",
                    "import { Router } from 'express';\n\
                     import { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.post('/users', async (req, res, next) => {\n  \
                     try {\n    \
                     const { name, email, age } = req.body;\n    \
                     if (typeof name !== 'string' || name.length === 0) {\n      \
                     return res.status(400).json({ error: 'name is required' });\n    }\n    \
                     if (!email.includes('@')) {\n      \
                     return res.status(400).json({ error: 'invalid email' });\n    }\n    \
                     if (typeof age !== 'number' || age < 0 || age > 150) {\n      \
                     return res.status(400).json({ error: 'invalid age' });\n    }\n    \
                     const user = await db.users.create({ data: { name, email, age } });\n    \
                     res.status(201).json(user);\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: { create: async (_data: any) => ({ id: '1' }) },\n};\n",
                ),
            ],
            checks: &[("src/routes/users.ts", |c| {
                (c.contains("z.object") || c.contains("from 'zod'") || c.contains("from \"zod\""))
                    && c.contains("422")
                    && (c.contains(".issues") || c.contains(".errors") || c.contains("flatten"))
                    && !c.contains("!email.includes('@')")
                    && !c.contains("typeof name")
            })],
        },
        CapabilityTask {
            name: "implement_cache_aside",
            difficulty: Difficulty::Hard,
            request: "src/products/service.ts hits the database on every call with no caching. \
                      Add a cache-aside layer using the Redis client in src/redis.ts: \
                      in getProduct, check Redis first (JSON.parse the value if present), and \
                      on cache miss populate Redis with JSON.stringify and a 300-second TTL. \
                      In updateProduct and deleteProduct, invalidate the cache key with redis.del. \
                      Do NOT use an in-memory Map — it breaks across server restarts.",
            files: &[
                (
                    "src/products/service.ts",
                    "import { db } from '../db';\n\n\
                     export async function getProduct(id: string) {\n  \
                     return db.products.findUnique({ where: { id } });\n}\n\n\
                     export async function updateProduct(\n  \
                     id: string, data: { name?: string; price?: number },\n) {\n  \
                     return db.products.update({ where: { id }, data });\n}\n\n\
                     export async function deleteProduct(id: string) {\n  \
                     await db.products.delete({ where: { id } });\n}\n",
                ),
                (
                    "src/redis.ts",
                    "export const redis = {\n  \
                     get: async (_key: string): Promise<string | null> => null,\n  \
                     setEx: async (_key: string, _ttl: number, _value: string): Promise<void> => {},\n  \
                     del: async (_key: string): Promise<void> => {},\n};\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     products: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     update: async (_q: any) => ({} as any),\n    \
                     delete: async (_q: any) => {},\n  },\n};\n",
                ),
            ],
            checks: &[("src/products/service.ts", |c| {
                c.contains("redis.get")
                    && (c.contains("redis.setEx") || c.contains("redis.set(") || c.contains("setex"))
                    && c.contains("JSON.stringify")
                    && c.contains("JSON.parse")
                    && c.contains("redis.del")
                    && !c.contains("new Map()")
            })],
        },
        CapabilityTask {
            name: "write_api_integration_tests",
            difficulty: Difficulty::Hard,
            request: "Write supertest integration tests for the auth routes in \
                      src/routes/auth.ts. Create src/routes/auth.test.ts with at least \
                      5 tests: POST /auth/register creates a user and returns a JWT; \
                      POST /auth/register with a duplicate email returns 409; \
                      POST /auth/login with wrong password returns 401; \
                      POST /auth/login with valid credentials returns a token; \
                      GET /auth/me with a valid Bearer token returns the user profile. \
                      Use jest.mock to mock the db module so no real database is required.",
            files: &[
                (
                    "src/routes/auth.ts",
                    "import { Router } from 'express';\n\
                     import { db } from '../db';\n\
                     import jwt from 'jsonwebtoken';\n\n\
                     const JWT_SECRET = process.env.JWT_SECRET ?? 'secret';\n\n\
                     const router = Router();\n\n\
                     router.post('/register', async (req, res, next) => {\n  \
                     try {\n    \
                     const { email, password } = req.body;\n    \
                     const existing = await db.users.findUnique({ where: { email } });\n    \
                     if (existing) return res.status(409).json({ error: 'email already registered' });\n    \
                     const user = await db.users.create({ data: { email, password } });\n    \
                     const token = jwt.sign({ userId: user.id }, JWT_SECRET);\n    \
                     res.status(201).json({ token });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     router.post('/login', async (req, res, next) => {\n  \
                     try {\n    \
                     const { email, password } = req.body;\n    \
                     const user = await db.users.findUnique({ where: { email } });\n    \
                     if (!user || user.password !== password)\n      \
                     return res.status(401).json({ error: 'invalid credentials' });\n    \
                     const token = jwt.sign({ userId: user.id }, JWT_SECRET);\n    \
                     res.json({ token });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     router.get('/me', async (req, res, next) => {\n  \
                     try {\n    \
                     const auth = req.headers.authorization;\n    \
                     if (!auth?.startsWith('Bearer '))\n      \
                     return res.status(401).json({ error: 'unauthorized' });\n    \
                     const { userId } = jwt.verify(auth.slice(7), JWT_SECRET) as { userId: string };\n    \
                     const user = await db.users.findUnique({ where: { id: userId } });\n    \
                     res.json({ id: user?.id, email: user?.email });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/app.ts",
                    "import express from 'express';\n\
                     import authRouter from './routes/auth';\n\n\
                     const app = express();\n\
                     app.use(express.json());\n\
                     app.use('/auth', authRouter);\n\n\
                     export default app;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     create: async (_q: any) => ({ id: '1' } as any),\n  },\n};\n",
                ),
            ],
            checks: &[("src/routes/auth.test.ts", |c| {
                let count = c.matches("it(").count() + c.matches("test(").count();
                (c.contains("supertest") || c.contains("request(app)"))
                    && (c.contains("jest.mock") || c.contains("mockResolvedValue") || c.contains("jest.fn"))
                    && c.contains("409")
                    && c.contains("401")
                    && c.contains("/me")
                    && count >= 5
            })],
        },
        CapabilityTask {
            name: "implement_soft_delete",
            difficulty: Difficulty::Medium,
            request: "deleteUser in src/users/repository.ts permanently removes the record. \
                      Convert it to a soft delete: set deletedAt to new Date() instead of \
                      calling db.users.delete. Update findUser and listUsers to exclude \
                      soft-deleted users by filtering WHERE deletedAt IS NULL. \
                      Add a restoreUser(id) function that sets deletedAt back to null.",
            files: &[
                (
                    "src/users/repository.ts",
                    "import { db } from '../db';\n\n\
                     export interface User {\n  id: string;\n  email: string;\n  \
                     name: string;\n  deletedAt: Date | null;\n}\n\n\
                     export async function findUser(id: string): Promise<User | null> {\n  \
                     return db.users.findFirst({ where: { id } }) as Promise<User | null>;\n}\n\n\
                     export async function listUsers(): Promise<User[]> {\n  \
                     return db.users.findMany({}) as Promise<User[]>;\n}\n\n\
                     export async function deleteUser(id: string): Promise<void> {\n  \
                     await db.users.delete({ where: { id } });\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: {\n    \
                     findFirst: async (_q: any) => null as any,\n    \
                     findMany: async (_q: any) => [] as any[],\n    \
                     update: async (_q: any) => ({} as any),\n    \
                     delete: async (_q: any) => {},\n  },\n};\n",
                ),
            ],
            checks: &[("src/users/repository.ts", |c| {
                c.contains("deletedAt")
                    && c.contains("new Date()")
                    && !c.contains("db.users.delete(")
                    && c.contains("restoreUser")
            })],
        },
        CapabilityTask {
            name: "decouple_email_with_events",
            difficulty: Difficulty::Hard,
            request: "The registration flow in src/auth/register.ts calls mailer.sendWelcomeEmail \
                      synchronously, coupling email delivery to the HTTP response time. \
                      Decouple them with Node.js EventEmitter: \
                      (1) Create src/events/emitter.ts exporting a singleton EventEmitter. \
                      (2) In src/auth/register.ts, emit a 'user.registered' event with the \
                      user object instead of calling sendWelcomeEmail directly. \
                      (3) Create src/events/listeners/welcome-email.ts that listens for \
                      'user.registered' and calls mailer.sendWelcomeEmail.",
            files: &[
                (
                    "src/auth/register.ts",
                    "import { db } from '../db';\n\
                     import { mailer } from '../mailer';\n\n\
                     export async function registerUser(email: string, password: string) {\n  \
                     const user = await db.users.create({ data: { email, password } });\n  \
                     await mailer.sendWelcomeEmail(user.email);\n  \
                     return user;\n}\n",
                ),
                (
                    "src/mailer.ts",
                    "export const mailer = {\n  \
                     sendWelcomeEmail: async (email: string) => {\n    \
                     console.log(`Sending welcome email to ${email}`);\n  },\n};\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: { create: async (_data: any) => ({ id: '1', ...(_data as any).data }) },\n};\n",
                ),
            ],
            checks: &[
                ("src/events/emitter.ts", |c| c.contains("EventEmitter")),
                ("src/auth/register.ts", |c| {
                    c.contains(".emit(")
                        && c.contains("user.registered")
                        && !c.contains("mailer.sendWelcomeEmail")
                }),
                ("src/events/listeners/welcome-email.ts", |c| {
                    c.contains(".on(")
                        && c.contains("user.registered")
                        && c.contains("sendWelcomeEmail")
                }),
            ],
        },
        // ── Research-backed: undisclosed scope ───────────────────────────────
        CapabilityTask {
            name: "n_plus_one_multi_site",
            difficulty: Difficulty::Hard,
            request: "The posts service is causing severe database slowdowns under load. \
                      N+1 queries are hammering the database. Find and fix the performance problem.",
            files: &[
                (
                    "src/posts/service.ts",
                    "import { db } from '../db';\n\
                     import { getUserById, getUsersByIds } from '../users/repository';\n\n\
                     export async function getPublishedPosts() {\n  \
                     const posts = await db.posts.findMany({ where: { published: true } });\n  \
                     return Promise.all(posts.map(async post => ({\n    \
                     ...post,\n    \
                     author: await getUserById(post.authorId),\n  })));\n}\n\n\
                     export async function getPostsByTag(tag: string) {\n  \
                     const posts = await db.posts.findMany({ where: { tags: { has: tag } } });\n  \
                     return Promise.all(posts.map(async post => ({\n    \
                     ...post,\n    \
                     author: await getUserById(post.authorId),\n  })));\n}\n\n\
                     export async function searchPosts(query: string) {\n  \
                     const posts = await db.posts.findMany({\n    \
                     where: { OR: [{ title: { contains: query } }, { body: { contains: query } }] },\n  });\n  \
                     return Promise.all(posts.map(async post => ({\n    \
                     ...post,\n    \
                     author: await getUserById(post.authorId),\n  })));\n}\n",
                ),
                (
                    "src/users/repository.ts",
                    "import { db } from '../db';\n\n\
                     export async function getUserById(id: string) {\n  \
                     return db.users.findUnique({ where: { id } });\n}\n\n\
                     export async function getUsersByIds(ids: string[]) {\n  \
                     return db.users.findMany({ where: { id: { in: ids } } });\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     posts: { findMany: async (_q: any) => [] as any[] },\n  \
                     users: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     findMany: async (_q: any) => [] as any[],\n  },\n};\n",
                ),
            ],
            checks: &[("src/posts/service.ts", |c| {
                !c.contains("await getUserById(post.authorId)")
                    && c.contains("getUsersByIds")
                    && c.contains("getPublishedPosts")
                    && c.contains("getPostsByTag")
                    && c.contains("searchPosts")
            })],
        },
        CapabilityTask {
            name: "prototype_pollution_fix",
            difficulty: Difficulty::Hard,
            request: "A security audit flagged the config merge utility: an attacker who can \
                      supply crafted JSON input can modify Object.prototype, affecting all \
                      objects in the process and potentially escalating privileges. \
                      Fix the vulnerability in src/utils/merge.ts.",
            files: &[
                (
                    "src/utils/merge.ts",
                    "export function deepMerge(\n  \
                     target: Record<string, any>,\n  \
                     source: Record<string, any>,\n): Record<string, any> {\n  \
                     const result = { ...target };\n  \
                     for (const key of Object.keys(source)) {\n    \
                     if (\n      typeof source[key] === 'object' &&\n      \
                     source[key] !== null &&\n      \
                     !Array.isArray(source[key])\n    ) {\n      \
                     result[key] = deepMerge(result[key] ?? {}, source[key]);\n    } else {\n      \
                     result[key] = source[key];\n    }\n  }\n  \
                     return result;\n}\n",
                ),
                (
                    "src/config/loader.ts",
                    "import { deepMerge } from '../utils/merge';\n\n\
                     const defaults = { timeout: 5000, retries: 3, debug: false };\n\n\
                     export function loadConfig(overrides: Record<string, any>) {\n  \
                     return deepMerge(defaults, overrides);\n}\n",
                ),
            ],
            checks: &[("src/utils/merge.ts", |c| {
                c.contains("__proto__") && c.contains("constructor") && c.contains("prototype")
            })],
        },
        CapabilityTask {
            name: "thread_tenant_id",
            difficulty: Difficulty::Hard,
            request: "The system needs data isolation between tenants. Add a required \
                      `tenantId: string` field to the User type in src/types/user.ts and \
                      thread it through the entire user creation flow so it is stored in the \
                      database and returned in the API response.",
            files: &[
                (
                    "src/types/user.ts",
                    "export interface User {\n  \
                     id: string;\n  email: string;\n  name: string;\n  createdAt: Date;\n}\n",
                ),
                (
                    "src/users/repository.ts",
                    "import { db } from '../db';\nimport { User } from '../types/user';\n\n\
                     export async function createUser(email: string, name: string): Promise<User> {\n  \
                     return db.users.create({ data: { email, name } }) as Promise<User>;\n}\n\n\
                     export async function findUserById(id: string): Promise<User | null> {\n  \
                     return db.users.findUnique({ where: { id } }) as Promise<User | null>;\n}\n",
                ),
                (
                    "src/users/service.ts",
                    "import { createUser } from './repository';\n\n\
                     export async function registerUser(email: string, name: string) {\n  \
                     return createUser(email, name);\n}\n",
                ),
                (
                    "src/routes/users.ts",
                    "import { Router } from 'express';\n\
                     import { registerUser } from '../users/service';\n\n\
                     const router = Router();\n\n\
                     router.post('/users', async (req, res, next) => {\n  \
                     try {\n    \
                     const { email, name } = req.body;\n    \
                     const user = await registerUser(email, name);\n    \
                     res.status(201).json(user);\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: {\n    \
                     create: async (_q: any) => ({} as any),\n    \
                     findUnique: async (_q: any) => null as any,\n  },\n};\n",
                ),
            ],
            checks: &[
                ("src/types/user.ts",       |c| c.contains("tenantId")),
                ("src/users/repository.ts", |c| c.contains("tenantId")),
                ("src/users/service.ts",    |c| c.contains("tenantId")),
                ("src/routes/users.ts",     |c| c.contains("tenantId")),
            ],
        },
        CapabilityTask {
            name: "typed_error_hierarchy",
            difficulty: Difficulty::Hard,
            request: "Error handling throws raw Error objects with string messages and the \
                      error handler matches on err.message to set HTTP status codes. A typo \
                      in any message string silently returns 500 instead of the correct status. \
                      Create typed error subclasses NotFoundError, ValidationError, and \
                      ConflictError extending AppError in src/errors.ts. Update all four \
                      service files to throw typed errors. Update the error handler to use \
                      instanceof dispatch instead of string matching.",
            files: &[
                (
                    "src/errors.ts",
                    "export class AppError extends Error {\n  \
                     constructor(message: string, public statusCode: number) {\n    \
                     super(message);\n    this.name = 'AppError';\n  }\n}\n",
                ),
                (
                    "src/middleware/errorHandler.ts",
                    "import { Request, Response, NextFunction } from 'express';\n\n\
                     export function errorHandler(\n  \
                     err: Error, _req: Request, res: Response, _next: NextFunction,\n): void {\n  \
                     if (err.message === 'not_found') {\n    \
                     res.status(404).json({ error: err.message });\n  \
                     } else if (err.message === 'conflict') {\n    \
                     res.status(409).json({ error: err.message });\n  \
                     } else if (err.message === 'validation_error') {\n    \
                     res.status(422).json({ error: err.message });\n  } else {\n    \
                     res.status(500).json({ error: 'internal_server_error' });\n  }\n}\n",
                ),
                (
                    "src/users/user.service.ts",
                    "import { db } from '../db';\n\n\
                     export async function getUser(id: string) {\n  \
                     const user = await db.users.findUnique({ where: { id } });\n  \
                     if (!user) throw new Error('not_found');\n  \
                     return user;\n}\n\n\
                     export async function updateUserEmail(id: string, email: string) {\n  \
                     const existing = await db.users.findUnique({ where: { email } });\n  \
                     if (existing) throw new Error('conflict');\n  \
                     return db.users.update({ where: { id }, data: { email } });\n}\n",
                ),
                (
                    "src/orders/order.service.ts",
                    "import { db } from '../db';\n\n\
                     export async function getOrder(id: string) {\n  \
                     const order = await db.orders.findUnique({ where: { id } });\n  \
                     if (!order) throw new Error('not_found');\n  \
                     return order;\n}\n",
                ),
                (
                    "src/payments/payment.service.ts",
                    "import { db } from '../db';\n\n\
                     export async function createPayment(orderId: string, amount: number) {\n  \
                     const order = await db.orders.findUnique({ where: { id: orderId } });\n  \
                     if (!order) throw new Error('not_found');\n  \
                     if (order.status === 'paid') throw new Error('conflict');\n  \
                     return db.payments.create({ data: { orderId, amount } });\n}\n",
                ),
                (
                    "src/products/product.service.ts",
                    "import { db } from '../db';\n\n\
                     export async function getProduct(id: string) {\n  \
                     const product = await db.products.findUnique({ where: { id } });\n  \
                     if (!product) throw new Error('not_found');\n  \
                     return product;\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     update: async (_q: any) => ({} as any),\n  },\n  \
                     orders: { findUnique: async (_q: any) => null as any },\n  \
                     payments: { create: async (_q: any) => ({} as any) },\n  \
                     products: { findUnique: async (_q: any) => null as any },\n};\n",
                ),
            ],
            checks: &[
                ("src/errors.ts", |c| {
                    c.contains("class NotFoundError")
                        && c.contains("class ConflictError")
                        && c.contains("extends AppError")
                }),
                ("src/middleware/errorHandler.ts", |c| {
                    c.contains("instanceof") && !c.contains("err.message ===")
                }),
                ("src/users/user.service.ts", |c| {
                    c.contains("NotFoundError") && !c.contains("new Error('not_found')")
                }),
                ("src/orders/order.service.ts", |c| {
                    c.contains("NotFoundError") && !c.contains("new Error('not_found')")
                }),
                ("src/payments/payment.service.ts", |c| {
                    (c.contains("NotFoundError") || c.contains("ConflictError"))
                        && !c.contains("new Error('conflict')")
                }),
                ("src/products/product.service.ts", |c| {
                    c.contains("NotFoundError") && !c.contains("new Error('not_found')")
                }),
            ],
        },
        CapabilityTask {
            name: "toctou_refresh_token",
            difficulty: Difficulty::Hard,
            request: "The refresh token endpoint has a race condition: two concurrent requests \
                      with the same token both read `used: false`, both pass the check, and \
                      both issue new access tokens — a replay attack. Fix the race so a token \
                      can only be consumed once even under concurrent load.",
            files: &[
                (
                    "src/auth/token.ts",
                    "import { db } from '../db';\nimport jwt from 'jsonwebtoken';\n\n\
                     const SECRET = process.env.JWT_SECRET ?? 'secret';\n\n\
                     export async function refreshToken(token: string): Promise<string> {\n  \
                     const stored = await db.tokens.findUnique({ where: { token } });\n  \
                     if (!stored || stored.used) throw new Error('invalid token');\n  \
                     await db.tokens.update({ where: { token }, data: { used: true } });\n  \
                     return jwt.sign({ userId: stored.userId }, SECRET, { expiresIn: '1h' });\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     tokens: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     update: async (_q: any) => ({} as any),\n    \
                     updateMany: async (_q: any) => ({ count: 0 }),\n  },\n};\n",
                ),
            ],
            checks: &[("src/auth/token.ts", |c| {
                c.contains("updateMany")
                    && (c.contains(".count") || c.contains("count ===") || c.contains("count >"))
                    && !c.contains("findUnique")
            })],
        },
        CapabilityTask {
            name: "event_listener_leak",
            difficulty: Difficulty::Hard,
            request: "The WebSocket server accumulates event listeners on the shared event bus \
                      for every client connection, triggering MaxListenersExceededWarning after \
                      ~10 connections and causing memory to grow unboundedly. Fix the leak so \
                      listeners added to eventBus for a socket are removed when that socket closes.",
            files: &[
                (
                    "src/ws/handler.ts",
                    "import { EventEmitter } from 'node:events';\n\n\
                     export const eventBus = new EventEmitter();\n\n\
                     export function setupSocket(socket: any): void {\n  \
                     eventBus.on('broadcast', (data: unknown) => {\n    \
                     if (socket.readyState === 1) {\n      \
                     socket.send(JSON.stringify(data));\n    }\n  });\n\n  \
                     socket.on('message', (data: Buffer) => {\n    \
                     eventBus.emit('broadcast', JSON.parse(data.toString()));\n  });\n}\n",
                ),
                (
                    "src/cache/lru.ts",
                    "export class LruCache<K, V> {\n  \
                     private max: number;\n  \
                     private cache = new Map<K, V>();\n\n  \
                     constructor(max: number) { this.max = max; }\n\n  \
                     get(key: K): V | undefined {\n    \
                     if (!this.cache.has(key)) return undefined;\n    \
                     const value = this.cache.get(key)!;\n    \
                     this.cache.delete(key);\n    \
                     this.cache.set(key, value);\n    \
                     return value;\n  }\n\n  \
                     set(key: K, value: V): void {\n    \
                     if (this.cache.has(key)) {\n      \
                     this.cache.delete(key);\n    } else if (this.cache.size >= this.max) {\n      \
                     const firstKey = this.cache.keys().next().value!;\n      \
                     this.cache.delete(firstKey);\n    }\n    \
                     this.cache.set(key, value);\n  }\n}\n",
                ),
            ],
            checks: &[
                ("src/ws/handler.ts", |c| {
                    (c.contains("eventBus.off") || c.contains("removeListener"))
                        && (c.contains("'close'") || c.contains("\"close\""))
                }),
                ("src/cache/lru.ts", |c| {
                    // Red herring — correct LRU must remain untouched
                    c.contains("class LruCache") && c.contains("firstKey")
                }),
            ],
        },
        CapabilityTask {
            name: "discriminated_union_types",
            difficulty: Difficulty::Hard,
            request: "The notification handlers use type assertions (`as EmailNotification`) \
                      because TypeScript cannot narrow the union — the `type` field is typed \
                      as `string` rather than a literal. Fix the type definition so each \
                      subtype carries a literal `type` field, enabling TypeScript to narrow \
                      automatically. Then remove all `as` type assertions from the three \
                      handler files.",
            files: &[
                (
                    "src/types/notification.ts",
                    "export interface BaseNotification {\n  \
                     id: string;\n  type: string;\n  createdAt: Date;\n}\n\n\
                     export interface EmailNotification extends BaseNotification {\n  \
                     to: string;\n  subject: string;\n  body: string;\n}\n\n\
                     export interface SmsNotification extends BaseNotification {\n  \
                     phoneNumber: string;\n  message: string;\n}\n\n\
                     export interface PushNotification extends BaseNotification {\n  \
                     deviceToken: string;\n  title: string;\n  payload: Record<string, unknown>;\n}\n\n\
                     export type Notification = EmailNotification | SmsNotification | PushNotification;\n",
                ),
                (
                    "src/handlers/email.ts",
                    "import { Notification, EmailNotification } from '../types/notification';\n\n\
                     export function handleEmail(n: Notification): void {\n  \
                     if (n.type === 'email') {\n    \
                     const email = n as EmailNotification;\n    \
                     console.log(`To: ${email.to} | Subject: ${email.subject}`);\n  }\n}\n",
                ),
                (
                    "src/handlers/sms.ts",
                    "import { Notification, SmsNotification } from '../types/notification';\n\n\
                     export function handleSms(n: Notification): void {\n  \
                     if (n.type === 'sms') {\n    \
                     const sms = n as SmsNotification;\n    \
                     console.log(`SMS to ${sms.phoneNumber}: ${sms.message}`);\n  }\n}\n",
                ),
                (
                    "src/handlers/push.ts",
                    "import { Notification, PushNotification } from '../types/notification';\n\n\
                     export function handlePush(n: Notification): void {\n  \
                     if (n.type === 'push') {\n    \
                     const push = n as PushNotification;\n    \
                     console.log(`Push to ${push.deviceToken}: ${push.title}`);\n  }\n}\n",
                ),
            ],
            checks: &[
                ("src/types/notification.ts", |c| {
                    c.contains("type: 'email'") && c.contains("type: 'sms'") && c.contains("type: 'push'")
                }),
                ("src/handlers/email.ts", |c| !c.contains(" as EmailNotification")),
                ("src/handlers/sms.ts",   |c| !c.contains(" as SmsNotification")),
                ("src/handlers/push.ts",  |c| !c.contains(" as PushNotification")),
            ],
        },
        CapabilityTask {
            name: "keyof_repository_constraint",
            difficulty: Difficulty::Hard,
            request: "BaseRepository.findBy accepts any string for the field name, so a \
                      typo like `findBy('emial', value)` silently returns null instead of \
                      failing at compile time. Tighten findBy and findAllBy to accept \
                      only `keyof T` so TypeScript catches field-name typos at compile time. \
                      The concrete repositories must continue to work without changes to \
                      their call sites.",
            files: &[
                (
                    "src/db/repository.ts",
                    "export class BaseRepository<T extends Record<string, unknown>> {\n  \
                     protected tableName: string;\n\n  \
                     constructor(tableName: string) { this.tableName = tableName; }\n\n  \
                     async findBy(field: string, value: unknown): Promise<T | null> {\n    \
                     return null;\n  }\n\n  \
                     async findAllBy(field: string, value: unknown): Promise<T[]> {\n    \
                     return [];\n  }\n}\n",
                ),
                (
                    "src/repositories/user.repository.ts",
                    "import { BaseRepository } from '../db/repository';\n\n\
                     interface User { id: string; email: string; name: string; role: string; }\n\n\
                     export class UserRepository extends BaseRepository<User> {\n  \
                     constructor() { super('users'); }\n\n  \
                     async findByEmail(email: string) {\n    \
                     return this.findBy('email', email);\n  }\n}\n",
                ),
                (
                    "src/repositories/product.repository.ts",
                    "import { BaseRepository } from '../db/repository';\n\n\
                     interface Product { id: string; name: string; sku: string; price: number; }\n\n\
                     export class ProductRepository extends BaseRepository<Product> {\n  \
                     constructor() { super('products'); }\n\n  \
                     async findBySku(sku: string) {\n    \
                     return this.findBy('sku', sku);\n  }\n}\n",
                ),
            ],
            checks: &[("src/db/repository.ts", |c| {
                c.contains("keyof T") && !c.contains("field: string")
            })],
        },
        CapabilityTask {
            name: "idempotency_key",
            difficulty: Difficulty::Hard,
            request: "Mobile clients retry payment requests on network timeout, causing \
                      duplicate charges. Add idempotency key support to POST /payments \
                      using the `Idempotency-Key` header and Redis: \
                      (1) if the key is already completed, return the cached response; \
                      (2) if the key is currently in-flight, return 409; \
                      (3) otherwise process normally and cache the response for 24 hours. \
                      Requests without the header must process normally.",
            files: &[
                (
                    "src/routes/payments.ts",
                    "import { Router } from 'express';\nimport { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.post('/payments', async (req, res, next) => {\n  \
                     try {\n    \
                     const { amount, currency, customerId } = req.body;\n    \
                     const payment = await db.payments.create({\n      \
                     data: { amount, currency, customerId, status: 'completed' },\n    });\n    \
                     res.status(201).json({ paymentId: payment.id, status: 'completed' });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/redis.ts",
                    "export const redis = {\n  \
                     get: async (_key: string): Promise<string | null> => null,\n  \
                     set: async (_key: string, _value: string, _opts?: any): Promise<void> => {},\n  \
                     setEx: async (_key: string, _ttl: number, _value: string): Promise<void> => {},\n  \
                     del: async (_key: string): Promise<void> => {},\n};\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     payments: { create: async (_q: any) => ({ id: 'pay_1' }) },\n};\n",
                ),
            ],
            checks: &[("src/routes/payments.ts", |c| {
                c.contains("Idempotency-Key")
                    && c.contains("redis.get")
                    && (c.contains("redis.set") || c.contains("redis.setEx"))
                    && c.contains("409")
                    && (c.contains("in-flight") || c.contains("inflight")
                        || c.contains("PROCESSING") || c.contains("processing")
                        || c.contains(":lock"))
            })],
        },
        CapabilityTask {
            name: "streaming_csv_export",
            difficulty: Difficulty::Hard,
            request: "The CSV export endpoint loads all records into memory before responding. \
                      With 500k rows this causes OOM crashes in production. Rewrite it to \
                      stream records from the database in batches using findByCursor, writing \
                      each batch to the response with res.write() as it arrives, so memory \
                      usage stays constant regardless of dataset size.",
            files: &[
                (
                    "src/routes/export.ts",
                    "import { Router } from 'express';\nimport { db } from '../db';\n\n\
                     function toCsvLine(row: Record<string, unknown>): string {\n  \
                     return Object.values(row).map(String).join(',');\n}\n\n\
                     const router = Router();\n\n\
                     router.get('/export', async (req, res, next) => {\n  \
                     try {\n    \
                     const rows = await db.records.findAll();\n    \
                     res.setHeader('Content-Type', 'text/csv');\n    \
                     res.setHeader('Content-Disposition', 'attachment; filename=\"export.csv\"');\n    \
                     res.send(rows.map(toCsvLine).join('\\n'));\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     records: {\n    \
                     findAll: async () => [] as any[],\n    \
                     findByCursor: async (_cursor: string | null, _take: number) => [] as any[],\n  },\n};\n",
                ),
            ],
            checks: &[("src/routes/export.ts", |c| {
                !c.contains("findAll()")
                    && c.contains("findByCursor")
                    && c.contains("res.write(")
                    && c.contains("res.end(")
                    && (c.contains("while") || c.contains("cursor"))
            })],
        },
        // ── Adversarial / metacognitive tasks ─────────────────────────────────
        CapabilityTask {
            name: "no_op_sql_injection",
            difficulty: Difficulty::Hard,
            request: "A security scanner flagged src/users/query.ts for a SQL injection \
                      vulnerability in the user search endpoint. Review and fix the issue.",
            files: &[
                (
                    "src/users/query.ts",
                    "import { db } from '../db';\n\n\
                     export async function searchUsers(query: string) {\n  \
                     return db.users.findMany({\n    \
                     where: {\n      OR: [\n        \
                     { name: { contains: query, mode: 'insensitive' } },\n        \
                     { email: { contains: query, mode: 'insensitive' } },\n      ],\n    },\n  });\n}\n\n\
                     export async function getUserByEmail(email: string) {\n  \
                     return db.users.findUnique({ where: { email } });\n}\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: {\n    \
                     findMany: async (_q: any) => [] as any[],\n    \
                     findUnique: async (_q: any) => null as any,\n  },\n};\n",
                ),
            ],
            checks: &[("src/users/query.ts", |c| {
                // Good model recognises the ORM already parameterises queries and makes no change.
                // Bad model introduces raw SQL string concatenation — the vulnerability it was asked to fix.
                c.contains("db.users.findMany")
                    && !c.contains("$queryRaw")
                    && !c.contains("+ query")
                    && !c.contains("${query")
            })],
        },
        CapabilityTask {
            name: "no_op_cursor_pagination",
            difficulty: Difficulty::Hard,
            request: "Users are reporting duplicate and skipped items when paginating the \
                      items list under high concurrent load. The root cause is OFFSET \
                      pagination returning inconsistent results during concurrent inserts. \
                      Fix the pagination in src/routes/items.ts.",
            files: &[
                (
                    "src/routes/items.ts",
                    "import { Router } from 'express';\nimport { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.get('/items', async (req, res, next) => {\n  \
                     try {\n    \
                     const cursor = req.query.cursor as string | undefined;\n    \
                     const limit  = Math.min(parseInt(req.query.limit as string) || 20, 100);\n    \
                     const decodedCursor = cursor\n      \
                     ? parseInt(Buffer.from(cursor, 'base64').toString('ascii'), 10)\n      \
                     : undefined;\n    \
                     const items = await db.items.findMany({\n      \
                     take: limit,\n      \
                     ...(decodedCursor ? { where: { id: { gt: decodedCursor } } } : {}),\n      \
                     orderBy: { id: 'asc' },\n    });\n    \
                     const nextCursor = items.length === limit\n      \
                     ? Buffer.from(String(items[items.length - 1].id)).toString('base64')\n      \
                     : null;\n    \
                     res.json({ items, nextCursor });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     items: { findMany: async (_q: any) => [] as any[] },\n};\n",
                ),
            ],
            checks: &[("src/routes/items.ts", |c| {
                // Good model recognises cursor pagination is already in place and makes no change.
                // Bad model reverts to OFFSET or introduces skip/page arithmetic.
                c.contains("nextCursor")
                    && c.contains("base64")
                    && !c.contains("page - 1")
                    && !c.contains("skip =")
            })],
        },
        CapabilityTask {
            name: "red_herring_ws_cache",
            difficulty: Difficulty::Hard,
            request: "The service heap grows by ~50MB per hour under normal load. \
                      A profiler shows the leak is in the networking layer. \
                      Find and fix the memory leak.",
            files: &[
                (
                    "src/ws/handler.ts",
                    "import { EventEmitter } from 'node:events';\n\n\
                     export const eventBus = new EventEmitter();\n\n\
                     export function setupSocket(socket: any): void {\n  \
                     eventBus.on('broadcast', (data: unknown) => {\n    \
                     if (socket.readyState === 1) {\n      \
                     socket.send(JSON.stringify(data));\n    }\n  });\n\n  \
                     socket.on('message', (data: Buffer) => {\n    \
                     eventBus.emit('broadcast', JSON.parse(data.toString()));\n  });\n}\n",
                ),
                (
                    "src/cache/lru.ts",
                    "export class LruCache<K, V> {\n  \
                     private max: number;\n  \
                     private cache = new Map<K, V>();\n\n  \
                     constructor(max: number) { this.max = max; }\n\n  \
                     get(key: K): V | undefined {\n    \
                     if (!this.cache.has(key)) return undefined;\n    \
                     const value = this.cache.get(key)!;\n    \
                     this.cache.delete(key);\n    \
                     this.cache.set(key, value);\n    \
                     return value;\n  }\n\n  \
                     set(key: K, value: V): void {\n    \
                     if (this.cache.has(key)) {\n      \
                     this.cache.delete(key);\n    } else if (this.cache.size >= this.max) {\n      \
                     const firstKey = this.cache.keys().next().value!;\n      \
                     this.cache.delete(firstKey);\n    }\n    \
                     this.cache.set(key, value);\n  }\n}\n",
                ),
            ],
            checks: &[
                ("src/ws/handler.ts", |c| {
                    (c.contains("eventBus.off") || c.contains("removeListener"))
                        && (c.contains("'close'") || c.contains("\"close\""))
                }),
                ("src/cache/lru.ts", |c| {
                    // Red herring — correct LRU must be left untouched
                    c.contains("class LruCache") && c.contains("firstKey")
                }),
            ],
        },
        CapabilityTask {
            name: "red_herring_n_plus_one",
            difficulty: Difficulty::Hard,
            request: "Database monitoring shows some endpoints issue an abnormal number of \
                      queries per request. Find and fix the N+1 query problem.",
            files: &[
                (
                    "src/routes/users.ts",
                    "import { Router } from 'express';\nimport { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.get('/users', async (req, res, next) => {\n  \
                     try {\n    \
                     const users = await db.users.findMany({ where: { active: true } });\n    \
                     const result = await Promise.all(\n      \
                     users.map(async user => ({\n        \
                     ...user,\n        \
                     role: await db.roles.findUnique({ where: { id: user.roleId } }),\n      }))\n    \
                     );\n    \
                     res.json(result);\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/routes/products.ts",
                    "import { Router } from 'express';\nimport { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.get('/products', async (req, res, next) => {\n  \
                     try {\n    \
                     const products = await db.products.findMany({\n      \
                     where: { active: true },\n      \
                     include: { category: true },\n    });\n    \
                     res.json(products);\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     users: { findMany: async (_q: any) => [] as any[] },\n  \
                     roles: {\n    \
                     findUnique: async (_q: any) => null as any,\n    \
                     findMany: async (_q: any) => [] as any[],\n  },\n  \
                     products: { findMany: async (_q: any) => [] as any[] },\n};\n",
                ),
            ],
            checks: &[
                ("src/routes/users.ts", |c| {
                    // N+1 fixed — no per-user role lookup inside the map
                    !c.contains("await db.roles.findUnique({ where: { id: user.roleId } })")
                }),
                ("src/routes/products.ts", |c| {
                    // Red herring — already uses include JOIN, must not be touched
                    c.contains("include: { category: true }")
                }),
            ],
        },
        CapabilityTask {
            name: "vague_stale_test",
            difficulty: Difficulty::Hard,
            request: "The notification service tests are failing in CI after a recent change. \
                      Fix them.",
            files: &[
                (
                    "src/services/notification.ts",
                    "export interface Notification {\n  \
                     id: string;\n  message: string;\n  \
                     severity: 'low' | 'medium' | 'high' | 'critical';\n  \
                     createdAt: Date;\n}\n\n\
                     export function createNotification(\n  \
                     message: string,\n  severity: Notification['severity'],\n): Notification {\n  \
                     return { id: crypto.randomUUID(), message, severity, createdAt: new Date() };\n}\n\n\
                     export function isCritical(n: Notification): boolean {\n  \
                     return n.severity === 'critical';\n}\n",
                ),
                (
                    "src/services/notification.test.ts",
                    "import { createNotification, isCritical } from './notification';\n\n\
                     describe('notification service', () => {\n  \
                     it('creates a notification with level', () => {\n    \
                     const n = createNotification('disk full', 'critical');\n    \
                     expect((n as any).level).toBe('critical');\n  });\n\n  \
                     it('isCritical is true for critical level', () => {\n    \
                     const n = { id: '1', message: 't', level: 'critical', createdAt: new Date() } as any;\n    \
                     expect(isCritical(n)).toBe(true);\n  });\n\n  \
                     it('isCritical is false for non-critical', () => {\n    \
                     const n = { id: '1', message: 't', level: 'low', createdAt: new Date() } as any;\n    \
                     expect(isCritical(n)).toBe(false);\n  });\n});\n",
                ),
            ],
            checks: &[("src/services/notification.test.ts", |c| {
                c.contains("severity") && !c.contains("n.level") && !c.contains("level: '")
            })],
        },
        CapabilityTask {
            name: "vague_slow_checkout",
            difficulty: Difficulty::Hard,
            request: "Users report slow checkout — p99 latency is 3x above SLA. \
                      A trace shows the bottleneck is in the POST /checkout handler. Fix it.",
            files: &[
                (
                    "src/routes/checkout.ts",
                    "import { Router } from 'express';\nimport { db } from '../db';\n\n\
                     const router = Router();\n\n\
                     router.post('/checkout', async (req, res, next) => {\n  \
                     try {\n    \
                     const { cartId, userId } = req.body;\n    \
                     const inventory = await db.inventory.check(cartId);\n    \
                     const pricing   = await db.pricing.calculate(cartId, userId);\n    \
                     const shipping  = await db.shipping.getOptions(userId);\n    \
                     if (!inventory.available) {\n      \
                     return res.status(409).json({ error: 'items_unavailable' });\n    }\n    \
                     const order = await db.orders.create({\n      \
                     data: { cartId, userId, total: pricing.total, shipping: shipping.cheapest },\n    });\n    \
                     res.status(201).json({ orderId: order.id, total: pricing.total });\n  \
                     } catch (err) { next(err); }\n});\n\n\
                     export default router;\n",
                ),
                (
                    "src/db.ts",
                    "export const db = {\n  \
                     inventory: { check: async (_cartId: string) => ({ available: true }) },\n  \
                     pricing:   { calculate: async (_c: string, _u: string) => ({ total: 0 }) },\n  \
                     shipping:  { getOptions: async (_userId: string) => ({ cheapest: 'standard' }) },\n  \
                     orders:    { create: async (_q: any) => ({ id: 'order-1' }) },\n};\n",
                ),
            ],
            checks: &[("src/routes/checkout.ts", |c| {
                c.contains("Promise.all")
                    && c.contains("inventory")
                    && c.contains("pricing")
                    && c.contains("shipping")
            })],
        },
    ]
}

/// All tasks in suite() are discriminative, so bench_suite returns the full suite.
pub fn bench_suite() -> Vec<CapabilityTask> {
    suite()
}
