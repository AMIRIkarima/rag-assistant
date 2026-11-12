import os, time
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
llm = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

for model in ["mistral-small-latest", "open-mistral-nemo", "ministral-8b-latest"]:
    try:
        r = llm.chat.complete(model=model, max_tokens=5,
                              messages=[{"role": "user", "content": "Say hello."}])
        print(model, "OK:", r.choices[0].message.content)
    except Exception as e:
        print(model, "FAILED:", str(e)[:120])
    time.sleep(20)