"""Research-only, loopback OpenAI-compatible request gate for synthetic runs.

The 40-request cap is per process and counts attempted upstream calls, including
failures. It is neither a provider-enforced spending limit nor a containment
boundary. Do not expose this listener beyond loopback or use real/private data.
"""

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CHAT_PATH = "/v1/chat/completions"
EMBED_PATH = "/v1/embeddings"
UPSTREAM = "https://dashscope-intl.aliyuncs.com/compatible-mode"
MAX_REQUESTS = 40
MAX_INPUT_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_TOKENS = 2048


class Rejected(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_body(path, body):
    if path not in (CHAT_PATH, EMBED_PATH):
        raise Rejected(404, "unsupported path")
    if not body or len(body) > MAX_INPUT_BYTES:
        raise Rejected(413, "request size outside limit")
    try:
        data = json.loads(body.decode("utf-8", errors="strict"),
                          object_pairs_hook=_unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite number")))
    except (UnicodeError, ValueError, TypeError):
        raise Rejected(400, "invalid JSON") from None
    if not isinstance(data, dict):
        raise Rejected(400, "JSON object required")
    if path == CHAT_PATH:
        allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "stream", "response_format"}
        if data.get("model") != "qwen3.8-flash" or set(data) - allowed:
            raise Rejected(400, "unsupported chat request")
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) > 128:
            raise Rejected(400, "invalid messages")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str)):
                raise Rejected(400, "text messages only")
        tokens = data.get("max_tokens", MAX_TOKENS)
        if type(tokens) is not int or not 1 <= tokens <= MAX_TOKENS:
            raise Rejected(400, "max_tokens outside limit")
        if type(data.get("stream", False)) is not bool:
            raise Rejected(400, "invalid stream flag")
        response_format = data.get("response_format")
        if response_format is not None:
            if not isinstance(response_format, dict):
                raise Rejected(400, "invalid response format")
            if response_format == {"type": "json_object"}:
                pass
            elif (set(response_format) == {"type", "json_schema"}
                  and response_format["type"] == "json_schema"
                  and isinstance(response_format["json_schema"], dict)
                  and set(response_format["json_schema"]) <= {"name", "schema", "strict"}
                  and isinstance(response_format["json_schema"].get("name"), str)
                  and isinstance(response_format["json_schema"].get("schema"), dict)
                  and type(response_format["json_schema"].get("strict", False)) is bool):
                pass
            else:
                raise Rejected(400, "invalid response format")
        for parameter in ("temperature", "top_p"):
            if parameter in data and (type(data[parameter]) not in (float, int)
                                      or not 0 <= data[parameter] <= 1):
                raise Rejected(400, "invalid sampling parameter")
        data["max_tokens"] = tokens
    else:
        if (data.get("model") != "text-embedding-v4"
                or set(data) - {"model", "input", "encoding_format", "dimensions"}):
            raise Rejected(400, "unsupported embedding request")
        value = data.get("input")
        if not (isinstance(value, str) and value or
                isinstance(value, list) and 0 < len(value) <= 128
                and all(isinstance(item, str) and item for item in value)):
            raise Rejected(400, "text input required")
        if data.get("encoding_format", "float") != "float":
            raise Rejected(400, "float embeddings only")
        if "dimensions" in data and (type(data["dimensions"]) is not int or data["dimensions"] != 1024):
            raise Rejected(400, "unsupported embedding dimensions")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_INPUT_BYTES:
        raise Rejected(413, "request size outside limit")
    return encoded


def provider_transport(path, body):
    """One upstream attempt; no redirects, retries, ambient proxy, or logging."""
    credential = os.environ.get("DASHSCOPE_API_KEY")
    if not credential:
        raise RuntimeError("provider credential unavailable")
    request = urllib.request.Request(
        UPSTREAM + path, data=body, method="POST",
        headers={"Authorization": "Bearer " + credential,
                 "Content-Type": "application/json", "Accept": "application/json"},
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("upstream failed")
        output = response.read(MAX_OUTPUT_BYTES + 1)
        if len(output) > MAX_OUTPUT_BYTES:
            raise RuntimeError("upstream response too large")
        return output


class BudgetProxy:
    def __init__(self, transport=provider_transport):
        self.transport = transport
        self._lock = threading.Lock()
        self.attempted = 0

    def handle(self, path, body):
        encoded = _parse_body(path, body)
        with self._lock:
            if self.attempted >= MAX_REQUESTS:
                raise Rejected(429, "request budget exhausted")
            self.attempted += 1
            print("PROXY_ATTEMPT", self.attempted, "chat" if path == CHAT_PATH else "embedding", flush=True)
        try:
            output = self.transport(path, encoded)
            if not isinstance(output, bytes) or len(output) > MAX_OUTPUT_BYTES:
                raise RuntimeError("invalid upstream response")
            return output
        except Exception:
            raise Rejected(502, "upstream request failed") from None


def make_handler(proxy):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never print URL, headers, body, provider output, or errors.

        def _reply(self, status, payload, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self.connection.settimeout(10)
            if self.path not in (CHAT_PATH, EMBED_PATH):
                self._reply(404, b'{"error":"unsupported path"}')
                return
            lengths = self.headers.get_all("Content-Length", [])
            if (len(lengths) != 1 or len(lengths[0]) > 5 or not lengths[0].isdigit()
                    or self.headers.get("Transfer-Encoding") is not None
                    or self.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json"):
                self._reply(400, b'{"error":"invalid request headers"}')
                return
            size = int(lengths[0])
            if size < 1 or size > MAX_INPUT_BYTES:
                self._reply(413, b'{"error":"request size outside limit"}')
                return
            try:
                body = self.rfile.read(size)
                if len(body) != size:
                    raise Rejected(400, "incomplete request")
                result = proxy.handle(self.path, body)
                stream = json.loads(body).get("stream", False) if self.path == CHAT_PATH else False
                self._reply(200, result, "text/event-stream" if stream else "application/json")
            except Rejected as error:
                print("PROXY_REFUSAL", error.status, error.message, flush=True)
                self._reply(error.status, json.dumps({"error": error.message}).encode("ascii"))
            except (OSError, TimeoutError):
                self._reply(400, b'{"error":"incomplete request"}')

        def do_GET(self):
            self._reply(405, b'{"error":"method not allowed"}')

    return Handler


def main():
    proxy = BudgetProxy()
    server = ThreadingHTTPServer(("127.0.0.1", 8765), make_handler(proxy))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
