import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="sk-UtRZwyTejfygYTvzyQ3U5BP1KOi9XhCMFuqNZIiTR40IkyA7",
    base_url="https://api.aiclaude.site/"
)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)