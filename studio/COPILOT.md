# GitHub Copilot subscriptions in Studio

Studio can use each user's GitHub Copilot subscription through the official
[Copilot SDK and GitHub OAuth flow](https://docs.github.com/en/copilot/how-tos/copilot-sdk/setup/github-oauth).
Usage belongs to the signed-in user and remains subject to their plan allowance
and organization policies. Personal subscription connections cannot be shared
with other Studio accounts.

## Installation setup

1. Use Python 3.11 or newer and install the Studio dependencies. These include
   `microsoft-copilot-sdk==2.0.3`.
2. Register a GitHub **OAuth App** under Settings → Developer settings → OAuth Apps.
   Enable **Device Flow**. Studio uses device authorization; it does not need an
   OAuth client secret or a browser callback endpoint.
3. Set `UNSLOTH_GITHUB_COPILOT_CLIENT_ID` to that app's client ID in the Studio
   backend environment and restart Studio. Use an app owned by your installation
   or deployment; do not reuse another product's OAuth client ID.
4. Open **Settings → Connections**, choose **GitHub Copilot subscription**, then
   **Sign in with GitHub**. Open the GitHub sign-in page, enter the displayed code,
   and authorize the app. Studio discovers the models available to that user.
5. Choose the models to enable and save the connection. Select one in the chat picker.

The SDK downloads its pinned Copilot runtime on first use. The backend needs
network access to GitHub Releases and Copilot services. Requests run in a
separate runtime process with temporary state and an explicit user token;
Studio does not use the server owner's Copilot CLI login.

Tokens and pending device credentials are stored in Studio's encrypted,
account-scoped credential vault. The browser receives only the sign-in code and
flow ID. **Disconnect** removes the saved token and cancels pending sign-in.
To revoke the OAuth grant at GitHub too, remove the app under your GitHub
account's authorized OAuth applications.

## Supported behavior

This integration supports text chat, streaming, model discovery and token usage
reporting. The SDK owns the conversation engine; Studio supplies edited chat
history as a transcript in a fresh request, rather than generating earlier
answers again. System instructions are supplied separately.

Images, Studio tools, JSON mode and sampling/output-cap overrides are currently
unsupported. SDK tools, workspace discovery and host permission requests are
disabled. This connection does not expose Copilot as an unrestricted
OpenAI-compatible proxy and does not use `copilot-api` or private Copilot endpoints.

The SDK's software license does not replace the user's Copilot service agreement.
Use the documented integration within the applicable subscription terms.
