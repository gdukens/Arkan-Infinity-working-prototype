import toml
from groq import Groq
c = toml.load('config.toml')
key = c.get('api', {}).get('key')
print('config key present:', bool(key))
if key:
    print('masked key:', key[:6] + '...' + key[-4:])
try:
    client = Groq(api_key=key)
    print('Groq client instantiated:', type(client))
except Exception as e:
    print('Groq client instantiation error:', str(e))
