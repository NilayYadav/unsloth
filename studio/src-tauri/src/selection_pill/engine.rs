use super::config::{self, PillConfig};
use super::{
    ax, geometry, monitor, panel, paste, PillState, SelectionPayload, Session,
    ASK_WINDOW_LABEL, EVENT_ASK_SHOW, EVENT_PERMISSION_CHANGED, EVENT_SELECTION,
    MAX_CAPTURE_BYTES, PILL_WINDOW_LABEL,
};
use core_graphics::display::CGDisplay;
use log::{info, warn};
use objc2_app_kit::NSRunningApplication;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{
    AppHandle, Emitter, LogicalPosition, Manager, State, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

const PILL_INITIAL_SIZE: (f64, f64) = (340.0, 56.0);
const ASK_SIZE: (f64, f64) = (640.0, 72.0);

pub fn init(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let handle = app.handle().clone();
    let dir = handle.path().app_config_dir()?;
    let loaded = config::load_config(&dir);
    {
        let state = app.state::<PillState>();
        *state.config.lock().unwrap() = loaded.clone();
    }

    let window = WebviewWindowBuilder::new(
        &handle,
        PILL_WINDOW_LABEL,
        WebviewUrl::App("pill.html".into()),
    )
    .title("Unsloth")
    .inner_size(PILL_INITIAL_SIZE.0, PILL_INITIAL_SIZE.1)
    .visible(false)
    .decorations(false)
    .transparent(true)
    .resizable(false)
    .skip_taskbar(true)
    .always_on_top(true)
    .shadow(false)
    .focused(false)
    .accept_first_mouse(true)
    .build()?;
    panel::convert_to_panel(&window).map_err(std::io::Error::other)?;

    let ask_window = WebviewWindowBuilder::new(
        &handle,
        ASK_WINDOW_LABEL,
        WebviewUrl::App("ask.html".into()),
    )
    .title("Unsloth")
    .inner_size(ASK_SIZE.0, ASK_SIZE.1)
    .visible(false)
    .decorations(false)
    .transparent(true)
    .resizable(false)
    .skip_taskbar(true)
    .always_on_top(true)
    .shadow(false)
    .focused(false)
    .build()?;
    panel::convert_to_key_panel(&ask_window).map_err(std::io::Error::other)?;

    // Native frosted glass behind the transparent webviews; radius must match
    // the CSS corner radius (pill rounded-xl = 12, ask rounded-2xl = 16).
    {
        use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial, NSVisualEffectState};
        if let Err(e) = apply_vibrancy(
            &window,
            NSVisualEffectMaterial::Popover,
            Some(NSVisualEffectState::Active),
            Some(12.0),
        ) {
            warn!("selection-pill: pill vibrancy failed: {e}");
        }
        if let Err(e) = apply_vibrancy(
            &ask_window,
            NSVisualEffectMaterial::Popover,
            Some(NSVisualEffectState::Active),
            Some(16.0),
        ) {
            warn!("selection-pill: ask vibrancy failed: {e}");
        }
    }

    monitor::install_dismiss_monitors(handle.clone());
    if let Err(e) = apply_hotkey(&handle, &loaded) {
        warn!("selection-pill: hotkey registration failed: {e}");
    }
    info!(
        "selection-pill: initialized (enabled: {}, hotkey: {})",
        loaded.enabled, loaded.hotkey
    );
    Ok(())
}

pub fn apply_config(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    config::save_for_app(app, config)?;
    // Registration is best-effort: a taken hotkey (another launcher, a stale
    // instance) must not fail the save the user just made.
    if let Err(e) = apply_hotkey(app, config) {
        warn!("selection-pill: hotkey registration failed: {e}");
    }
    Ok(())
}

fn apply_hotkey(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    let shortcuts = app.global_shortcut();
    shortcuts
        .unregister_all()
        .map_err(|e| format!("Failed to clear shortcuts: {e}"))?;
    if config.enabled {
        shortcuts
            .on_shortcut(config.hotkey.as_str(), move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    trigger(app);
                }
            })
            .map_err(|e| format!("Failed to register hotkey '{}': {e}", config.hotkey))?;
    }
    if config.ask_enabled {
        shortcuts
            .on_shortcut(config.ask_hotkey.as_str(), move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    toggle_ask(app);
                }
            })
            .map_err(|e| {
                format!("Failed to register ask hotkey '{}': {e}", config.ask_hotkey)
            })?;
    }
    Ok(())
}

/// Raycast-style toggle: hide when visible, else center on the mouse screen
/// and take keyboard focus without activating the app.
pub fn toggle_ask(app: &AppHandle) {
    let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) else {
        return;
    };
    if panel::is_panel_visible(&window) {
        monitor::hide_ask(app, &window);
        return;
    }
    show_ask(app, None);
}

/// Show the ask bar, optionally seeded with selected text as context.
pub fn show_ask(app: &AppHandle, context: Option<String>) {
    let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) else {
        return;
    };
    let screen = screen_containing(mouse_anchor());
    let size = panel::panel_frame(&window)
        .map(|frame| (frame.width, frame.height))
        .unwrap_or(ASK_SIZE);
    let x = screen.x + (screen.width - size.0) / 2.0;
    let y = screen.y + screen.height * 0.22;
    let _ = window.set_position(LogicalPosition::new(x, y));
    let _ = app.emit_to(ASK_WINDOW_LABEL, EVENT_ASK_SHOW, context);
    panel::show_key_panel(&window);
}

pub fn trigger(app: &AppHandle) {
    static IN_FLIGHT: AtomicBool = AtomicBool::new(false);
    if IN_FLIGHT.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            capture_and_show(&app);
        }));
        if result.is_err() {
            warn!("selection-pill: capture worker panicked");
        }
        IN_FLIGHT.store(false, Ordering::SeqCst);
    });
}

fn capture_and_show(app: &AppHandle) {
    let capture = match ax::capture_selection() {
        Ok(capture) => capture,
        Err("not-trusted") => {
            let _ = app.emit_to("main", EVENT_PERMISSION_CHANGED, false);
            return;
        }
        Err(code) => {
            info!("selection-pill: no capture ({code})");
            show_payload(app, error_payload(app, code));
            return;
        }
    };

    let (app_name, bundle_id) = app_info_for_pid(capture.pid);
    let state = app.state::<PillState>();
    if state.config.lock().unwrap().is_app_excluded(&bundle_id) {
        return;
    }

    let text = truncate_text(capture.text);
    let id = next_session_id(&state);
    let payload = SelectionPayload {
        session_id: id,
        text: text.clone(),
        app_name,
        bundle_id,
        editable: capture.editable,
        error: None,
    };
    *state.session.lock().unwrap() = Some(Session {
        id,
        text,
        pid: capture.pid,
        payload: payload.clone(),
        ax_element: Some(capture.element),
    });

    let anchor = capture.bounds.unwrap_or_else(mouse_anchor);
    show_payload_at(app, payload, anchor);
}

fn next_session_id(state: &PillState) -> u64 {
    state
        .session_counter
        .fetch_add(1, Ordering::SeqCst)
        .wrapping_add(1)
}

fn error_payload(app: &AppHandle, code: &str) -> SelectionPayload {
    let state = app.state::<PillState>();
    SelectionPayload {
        session_id: next_session_id(&state),
        text: String::new(),
        app_name: String::new(),
        bundle_id: String::new(),
        editable: false,
        error: Some(code.to_string()),
    }
}

fn show_payload(app: &AppHandle, payload: SelectionPayload) {
    show_payload_at(app, payload, mouse_anchor());
}

fn show_payload_at(app: &AppHandle, payload: SelectionPayload, anchor: geometry::Rect) {
    let Some(window) = app.get_webview_window(PILL_WINDOW_LABEL) else {
        return;
    };
    let screen = screen_containing(anchor);
    let pill = panel::panel_frame(&window)
        .map(|frame| (frame.width, frame.height))
        .unwrap_or(PILL_INITIAL_SIZE);
    let (x, y) = geometry::place_pill(anchor, pill, screen);
    let _ = window.set_position(LogicalPosition::new(x, y));
    let _ = app.emit_to(PILL_WINDOW_LABEL, EVENT_SELECTION, payload);
    panel::show_panel(&window);
}

pub fn apply_result(
    window: &WebviewWindow,
    state: &State<'_, PillState>,
    session_id: u64,
    text: String,
    insert_below: bool,
) -> Result<(), String> {
    let (element, pid, paste_text) = {
        let guard = state.session.lock().unwrap();
        let session = guard.as_ref().ok_or("no-session")?;
        if session.id != session_id {
            return Err("stale-session".to_string());
        }
        let paste_text = if insert_below {
            format!("{}\n\n{}", session.text, text)
        } else {
            text
        };
        (session.ax_element.clone(), session.pid, paste_text)
    };

    panel::hide_panel(window);
    paste::apply(element.as_ref(), pid, &paste_text).map_err(str::to_string)?;
    state.session.lock().unwrap().take();
    Ok(())
}

pub fn last_payload(state: &State<'_, PillState>) -> Option<SelectionPayload> {
    state
        .session
        .lock()
        .unwrap()
        .as_ref()
        .map(|session| session.payload.clone())
}

fn truncate_text(mut text: String) -> String {
    if text.len() > MAX_CAPTURE_BYTES {
        let mut end = MAX_CAPTURE_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

fn app_info_for_pid(pid: i32) -> (String, String) {
    let Some(app) = NSRunningApplication::runningApplicationWithProcessIdentifier(pid) else {
        return (String::new(), String::new());
    };
    let name = app
        .localizedName()
        .map(|s| s.to_string())
        .unwrap_or_default();
    let bundle = app
        .bundleIdentifier()
        .map(|s| s.to_string())
        .unwrap_or_default();
    (name, bundle)
}

fn mouse_anchor() -> geometry::Rect {
    use core_graphics::event::CGEvent;
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
    let location = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .ok()
        .and_then(|source| CGEvent::new(source).ok())
        .map(|event| event.location());
    match location {
        Some(point) => geometry::Rect::new(point.x, point.y, 1.0, 1.0),
        None => geometry::Rect::new(0.0, 0.0, 1.0, 1.0),
    }
}

fn screen_containing(anchor: geometry::Rect) -> geometry::Rect {
    let center_x = anchor.x + anchor.width / 2.0;
    let center_y = anchor.y + anchor.height / 2.0;
    let displays = CGDisplay::active_displays().unwrap_or_default();
    for id in displays {
        let bounds = CGDisplay::new(id).bounds();
        if center_x >= bounds.origin.x
            && center_x < bounds.origin.x + bounds.size.width
            && center_y >= bounds.origin.y
            && center_y < bounds.origin.y + bounds.size.height
        {
            return geometry::Rect::new(
                bounds.origin.x,
                bounds.origin.y,
                bounds.size.width,
                bounds.size.height,
            );
        }
    }
    let main = CGDisplay::main().bounds();
    geometry::Rect::new(main.origin.x, main.origin.y, main.size.width, main.size.height)
}
