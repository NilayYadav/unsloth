import React from 'react';
import { createRoot } from 'react-dom/client';
import { mockIPC } from '@tauri-apps/api/mocks';
import { emit } from '@tauri-apps/api/event';
import './index.css';

window.__TAURI_INTERNALS__ = {};
window.emitUpdateEvent = emit;
let downloaded = false;
mockIPC(async (cmd) => {
  if (cmd === 'desktop_update_policy') return { mode: 'in_app', releasePageBaseUrl: 'https://github.com/unslothai/unsloth/releases/tag/', releaseTagPrefix: 'v' };
  if (cmd === 'check_desktop_update') return { currentVersion: '0.1.810-beta', version: '0.1.811-beta', rawJson: {} };
  if (cmd === 'start_backend_update') { window.backendStarted = true; return; }
  if (cmd === 'desktop_update_bundle_status') return { version: '0.1.811-beta', downloaded, downloading: false };
  if (cmd === 'download_desktop_update') {
    await emit('desktop-update-download', {version: '0.1.811-beta', downloaded: 37, total: 100});
    await new Promise(resolve => { window.finishShellDownload = () => { downloaded = true; resolve(); }; });
    return;
  }
  if (cmd === 'install_desktop_update') { window.shellInstalled = true; return; }
  if (cmd === 'plugin:process|restart') { window.relaunched = true; return; }
  if (['resume_desktop_update_cleanup', 'set_renderer_activity', 'mark_in_app_relaunch'].includes(cmd)) return;
  throw new Error(`Unexpected native command: ${cmd}`);
}, { shouldMockEvents: true });

const { useTauriUpdate } = await import('./hooks/use-tauri-update');
const { UpdateScreen } = await import('./components/tauri/update-screen');
function App() {
  const update = useTauriUpdate();
  window.currentUpdate = {status:update.status, logs:update.logs, progress:update.progress};
  if (update.status === 'idle') return <button onClick={update.checkForUpdate}>Check for update</button>;
  if (update.status === 'available') return <button onClick={update.installUpdate}>Install update</button>;
  return <UpdateScreen {...update} onRetry={update.retryUpdate} onSkipRestart={update.skipAndRestart} onCopyDiagnostics={update.copyDiagnostics} />;
}
createRoot(document.getElementById('root')).render(<App />);
