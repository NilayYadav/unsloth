# PR 10761: current-head UI evidence

Current head: c051392d0368c5ea0467e2049a1ac7bcc7b09ab7.
Before: merge base d0dbe9059efa443c6ad8bd1d51af7e2d9276a2bc.

Both previous_response_id and conversation were exercised separately through real
Responses HTTP requests and the production API-monitor UI. For each: BEFORE has
1 failed request row; AFTER has 0 and returns unsupported_parameter with the exact
parameter name. API recording is enabled; each control without a history reference
creates 1 failed row on both sides. No model was loaded and no routes were mocked.
No browser console or page errors. Both composites were visually inspected.

The same isolated source-build methodology, viewport and equal screenshot clip
as the parent README were used. This is local browser evidence, not hosted CI or
model inference. Current regression suite: 438 passed. The four new conversation
rejection cases failed before the fix and pass after it.
