import sys
sys.path.insert(0, ".")
from routes.inference import _extract_content_parts
from models.inference import ChatCompletionRequest

def turn(tag, role="user"):
    return {"role": role, "content": [
        {"type": "text", "text": "what is " + tag + "?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + tag}},
    ]}

print("=" * 70)
print("PR 9638 probe -- which image does a single-image vision backend receive?")
print("=" * 70)

failures = []

req = ChatCompletionRequest(messages=[
    turn("IMG_FIRST"),
    {"role": "assistant", "content": "that is the first image"},
    turn("IMG_SECOND"),
])
sent = _extract_content_parts(req.messages)[2]
print("")
print("[1] user attaches IMG_FIRST, then attaches IMG_SECOND")
print("    image the model actually receives : " + repr(sent))
print("    image the user just attached      : 'IMG_SECOND'")
if sent != "IMG_SECOND":
    print("    RESULT: BUG -- the model is answering about the OLD image")
    failures.append("latest-image-not-sent")
else:
    print("    RESULT: OK -- the newest image is sent")

for label, bad in (("empty payload", "data:image/png;base64,"),
                   ("no comma", "data:image/png;base64")):
    req = ChatCompletionRequest(messages=[
        turn("IMG_REAL"),
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "text", "text": "and now?"},
            {"type": "image_url", "image_url": {"url": bad}},
        ]},
    ])
    sent = _extract_content_parts(req.messages)[2]
    print("")
    print("[2] real image, then a data URL with " + label)
    print("    image the model actually receives : " + repr(sent))
    print("    image that must survive           : 'IMG_REAL'")
    if sent != "IMG_REAL":
        print("    RESULT: BUG -- a payload-less URL discarded a real image")
        failures.append("payloadless-clobber-" + label.replace(" ", "-"))
    else:
        print("    RESULT: OK -- the real image survived")

req = ChatCompletionRequest(messages=[
    turn("IMG_USER_PHOTO"),
    {"role": "assistant", "content": [
        {"type": "text", "text": "here is a cartoon of it"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMG_ASSISTANT_GENERATED"}},
    ]},
    {"role": "user", "content": "what colour was the shirt?"},
])
sent = _extract_content_parts(req.messages)[2]
print("")
print("[3] user photo, then an assistant-generated image, then a text-only follow-up")
print("    image the model actually receives : " + repr(sent))
print("    the user's own attachment         : 'IMG_USER_PHOTO'")
if sent != "IMG_USER_PHOTO":
    print("    RESULT: BUG -- an assistant image displaced the user's own attachment")
    failures.append("assistant-image-outranks-user")
else:
    print("    RESULT: OK -- the user's own attachment won")

print("")
print("=" * 70)
if failures:
    print("REPRO_RESULT=WRONG_IMAGE_SENT failures=" + ",".join(failures))
    sys.exit(1)
print("REPRO_RESULT=CORRECT_IMAGE_SENT")
