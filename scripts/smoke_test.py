"""
smoke_test.py — Validate that vLLM and/or LiteLLM endpoints are working.

Usage:
    # Test directly against vLLM (Part 1 / local testing):
    python smoke_test.py --mode direct

    # Test via LiteLLM gateway (Part 2 / production):
    python smoke_test.py --mode gateway --base-url http://llm.example.com/v1 --api-key sk-your-team-key

    # Quick local test with defaults:
    python smoke_test.py
"""

import argparse
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK first:  pip install openai")
    sys.exit(1)


def test_embedding(client: OpenAI, model: str) -> None:
    print(f"Testing embeddings (model={model!r}) ... ", end="", flush=True)
    resp = client.embeddings.create(model=model, input="This is a smoke test sentence.")
    dim = len(resp.data[0].embedding)
    assert dim > 0, "Embedding vector is empty"
    print(f"OK  (dimension={dim})")


def test_chat(client: OpenAI, model: str) -> None:
    print(f"Testing chat completions (model={model!r}) ... ", end="", flush=True)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'smoke test passed' and nothing else."}],
        max_tokens=16,
    )
    content = resp.choices[0].message.content or ""
    assert content.strip(), "Empty response from chat model"
    print(f"OK  (response: {content.strip()!r})")


def test_streaming(client: OpenAI, model: str) -> None:
    print(f"Testing streaming chat (model={model!r}) ... ", end="", flush=True)
    chunks = []
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count to three."}],
        max_tokens=32,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    assert chunks, "No chunks received from streaming response"
    print(f"OK  ({len(chunks)} chunks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for vLLM / LiteLLM endpoints")
    parser.add_argument(
        "--mode",
        choices=["direct", "gateway"],
        default="direct",
        help=(
            "'direct' tests vLLM directly (embed on :8000, chat on :8001). "
            "'gateway' tests through LiteLLM using --base-url and --api-key."
        ),
    )
    parser.add_argument("--base-url", default=None, help="Gateway base URL (gateway mode)")
    parser.add_argument("--api-key", default="unused", help="API key (gateway mode)")
    parser.add_argument("--embed-url", default="http://localhost:8000/v1", help="Embedding URL (direct mode)")
    parser.add_argument("--chat-url", default="http://localhost:8001/v1", help="Chat URL (direct mode)")
    parser.add_argument("--embed-model", default=None, help="Override embedding model name")
    parser.add_argument("--chat-model", default=None, help="Override chat model name")
    args = parser.parse_args()

    if args.mode == "gateway":
        if not args.base_url:
            print("Error: --base-url is required in gateway mode")
            sys.exit(1)
        embed_client = chat_client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        embed_model = args.embed_model or "qwen3-embed"
        chat_model = args.chat_model or "qwen3-chat"
        print(f"Mode: gateway  ({args.base_url})")
    else:
        embed_client = OpenAI(base_url=args.embed_url, api_key="unused")
        chat_client = OpenAI(base_url=args.chat_url, api_key="unused")
        # In direct mode the model name is the container path vLLM uses internally.
        embed_model = args.embed_model or "/model"
        chat_model = args.chat_model or "/model"
        print(f"Mode: direct  (embed={args.embed_url}, chat={args.chat_url})")

    print()
    errors = []

    for name, fn, client, model in [
        ("embedding", test_embedding, embed_client, embed_model),
        ("chat",      test_chat,      chat_client,  chat_model),
        ("streaming", test_streaming, chat_client,  chat_model),
    ]:
        try:
            fn(client, model)
        except Exception as exc:
            print(f"FAILED  ({exc})")
            errors.append(name)

    print()
    if errors:
        print(f"Smoke test FAILED for: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")


if __name__ == "__main__":
    main()
