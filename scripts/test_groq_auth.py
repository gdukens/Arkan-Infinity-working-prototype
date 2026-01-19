import toml
import os
from groq import Groq

KEY = None
try:
    cfg = toml.load('.streamlit/secrets.toml')
    KEY = cfg.get('GROQ_API_KEY')
except Exception:
    KEY = os.environ.get('GROQ_API_KEY')

if not KEY:
    print('NO_KEY')
    raise SystemExit(2)

client = Groq(api_key=KEY)
try:
    resp = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':'Say "ok" in one word.'}],
        max_completion_tokens=5,
        temperature=0.0,
        stream=False
    )
    content = getattr(resp.choices[0].message, 'content', '').strip()
    print('OK:', content[:200])
except Exception as e:
    print('ERROR:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()
