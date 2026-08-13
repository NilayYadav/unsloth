use super::{config::PillConfig, PillState, ASK_WINDOW_LABEL, PILL_WINDOW_LABEL};
use tauri::{AppHandle, State, WebviewWindow};

fn ensure_main_window(window: &WebviewWindow) -> Result<(), String> {
    if window.label() == "main" {
        Ok(())
    } else {
        Err("Pill config commands are only available to the main window.".to_string())
    }
}

fn ensure_pill_window(window: &WebviewWindow) -> Result<(), String> {
    if window.label() == PILL_WINDOW_LABEL {
        Ok(())
    } else {
        Err("Pill session commands are only available to the pill window.".to_string())
    }
}

fn ensure_ask_window(window: &WebviewWindow) -> Result<(), String> {
    if window.label() == ASK_WINDOW_LABEL {
        Ok(())
    } else {
        Err("Ask commands are only available to the ask window.".to_string())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PillStatus {
    pub supported: bool,
    pub enabled: bool,
    pub ax_trusted: bool,
    pub hotkey: String,
    pub excluded_apps: Vec<String>,
}

fn make_status(config: PillConfig) -> PillStatus {
    PillStatus {
        supported: cfg!(target_os = "macos"),
        enabled: config.enabled,
        ax_trusted: ax_trusted(),
        hotkey: config.hotkey,
        excluded_apps: config.excluded_apps,
    }
}

#[tauri::command]
pub fn pill_status(
    window: WebviewWindow,
    state: State<'_, PillState>,
) -> Result<PillStatus, String> {
    ensure_main_window(&window)?;
    Ok(make_status(state.config.lock().unwrap().clone()))
}

#[tauri::command]
pub fn pill_set_config(
    app: AppHandle,
    window: WebviewWindow,
    state: State<'_, PillState>,
    mut config: PillConfig,
) -> Result<PillStatus, String> {
    ensure_main_window(&window)?;
    // The UI never edits hotkeys; preserve the stored values.
    {
        let current = state.config.lock().unwrap();
        config.hotkey = current.hotkey.clone();
        config.ask_hotkey = current.ask_hotkey.clone();
    }
    *state.config.lock().unwrap() = config.clone();
    persist_and_apply(&app, &config)?;
    Ok(make_status(config))
}

#[tauri::command]
pub fn pill_request_permission(window: WebviewWindow) -> Result<bool, String> {
    ensure_main_window(&window)?;
    request_ax_permission()
}

#[tauri::command]
pub fn pill_open_privacy_settings(window: WebviewWindow) -> Result<(), String> {
    ensure_main_window(&window)?;
    open::that("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        .map_err(|e| format!("Failed to open System Settings: {e}"))
}

#[tauri::command]
pub fn pill_get_capture(
    window: WebviewWindow,
    state: State<'_, PillState>,
) -> Result<Option<super::SelectionPayload>, String> {
    ensure_pill_window(&window)?;
    Ok(last_payload(&state))
}

#[tauri::command]
pub fn pill_replace_selection(
    window: WebviewWindow,
    state: State<'_, PillState>,
    session_id: u64,
    text: String,
) -> Result<(), String> {
    ensure_pill_window(&window)?;
    apply_result(&window, &state, session_id, text, false)
}

#[tauri::command]
pub fn pill_insert_below(
    window: WebviewWindow,
    state: State<'_, PillState>,
    session_id: u64,
    text: String,
) -> Result<(), String> {
    ensure_pill_window(&window)?;
    apply_result(&window, &state, session_id, text, true)
}

#[tauri::command]
pub fn pill_dismiss(
    window: WebviewWindow,
    state: State<'_, PillState>,
) -> Result<(), String> {
    ensure_pill_window(&window)?;
    state.session.lock().unwrap().take();
    hide_pill(&window);
    Ok(())
}

// The server-port event is a one-shot broadcast the hidden webviews can miss
// while they are still loading; this lets them pull the current port instead.
#[tauri::command]
pub fn pill_server_port(
    window: WebviewWindow,
    backend: State<'_, crate::process::BackendState>,
) -> Result<Option<u16>, String> {
    if ensure_pill_window(&window).is_err() && ensure_ask_window(&window).is_err() {
        return Err("Only available to the pill and ask windows.".to_string());
    }
    Ok(backend.lock().unwrap().owned_backend_port())
}

// "Ask" in the pill hands the selection to the ask bar as context.
#[tauri::command]
pub fn pill_open_ask(
    app: AppHandle,
    window: WebviewWindow,
    state: State<'_, PillState>,
    text: String,
) -> Result<(), String> {
    ensure_pill_window(&window)?;
    state.session.lock().unwrap().take();
    hide_pill(&window);
    open_ask_with_context(&app, text);
    Ok(())
}

#[tauri::command]
pub fn ask_hide(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    ensure_ask_window(&window)?;
    hide_ask(&app, &window);
    Ok(())
}

#[tauri::command]
pub fn ask_resize(window: WebviewWindow, width: f64, height: f64) -> Result<(), String> {
    ensure_ask_window(&window)?;
    window
        .set_size(tauri::LogicalSize::new(width.max(320.0), height.max(48.0)))
        .map_err(|e| format!("Failed to resize ask window: {e}"))
}

#[tauri::command]
pub fn pill_resize(window: WebviewWindow, width: f64, height: f64) -> Result<(), String> {
    ensure_pill_window(&window)?;
    window
        .set_size(tauri::LogicalSize::new(width.max(120.0), height.max(32.0)))
        .map_err(|e| format!("Failed to resize pill: {e}"))
}

#[cfg(target_os = "macos")]
fn ax_trusted() -> bool {
    super::ax::is_process_trusted()
}

#[cfg(not(target_os = "macos"))]
fn ax_trusted() -> bool {
    false
}

#[cfg(target_os = "macos")]
fn request_ax_permission() -> Result<bool, String> {
    Ok(super::ax::request_process_trust())
}

#[cfg(not(target_os = "macos"))]
fn request_ax_permission() -> Result<bool, String> {
    Err("The selection pill is only supported on macOS.".to_string())
}

#[cfg(target_os = "macos")]
fn persist_and_apply(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    super::engine::apply_config(app, config)
}

#[cfg(not(target_os = "macos"))]
fn persist_and_apply(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    super::config::save_for_app(app, config)
}

#[cfg(target_os = "macos")]
fn apply_result(
    window: &WebviewWindow,
    state: &State<'_, PillState>,
    session_id: u64,
    text: String,
    insert_below: bool,
) -> Result<(), String> {
    super::engine::apply_result(window, state, session_id, text, insert_below)
}

#[cfg(not(target_os = "macos"))]
fn apply_result(
    _window: &WebviewWindow,
    _state: &State<'_, PillState>,
    _session_id: u64,
    _text: String,
    _insert_below: bool,
) -> Result<(), String> {
    Err("The selection pill is only supported on macOS.".to_string())
}

#[cfg(target_os = "macos")]
fn hide_pill(window: &WebviewWindow) {
    super::panel::hide_panel(window);
}

#[cfg(target_os = "macos")]
fn hide_ask(app: &AppHandle, window: &WebviewWindow) {
    super::monitor::hide_ask(app, window);
}

#[cfg(target_os = "macos")]
fn open_ask_with_context(app: &AppHandle, text: String) {
    super::engine::show_ask(app, Some(text));
}

#[cfg(not(target_os = "macos"))]
fn open_ask_with_context(_app: &AppHandle, _text: String) {}

#[cfg(not(target_os = "macos"))]
fn hide_ask(_app: &AppHandle, window: &WebviewWindow) {
    let _ = window.hide();
}

#[cfg(not(target_os = "macos"))]
fn hide_pill(window: &WebviewWindow) {
    let _ = window.hide();
}

#[cfg(target_os = "macos")]
fn last_payload(state: &State<'_, PillState>) -> Option<super::SelectionPayload> {
    super::engine::last_payload(state)
}

#[cfg(not(target_os = "macos"))]
fn last_payload(_state: &State<'_, PillState>) -> Option<super::SelectionPayload> {
    None
}
