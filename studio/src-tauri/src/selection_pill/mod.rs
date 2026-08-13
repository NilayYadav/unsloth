// Dev-mode TCC note: under `tauri dev` the Accessibility grant is attributed to
// the launching terminal/IDE, not the app bundle. Grant Accessibility to your
// terminal for local testing; `tccutil reset Accessibility ai.unsloth.studio`
// clears stale grants of packaged builds.

pub mod config;
pub mod geometry;

#[cfg(target_os = "macos")]
mod ax;
#[cfg(target_os = "macos")]
mod engine;
#[cfg(target_os = "macos")]
mod monitor;
#[cfg(target_os = "macos")]
mod panel;
#[cfg(target_os = "macos")]
mod paste;

pub mod commands;

use config::PillConfig;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;

pub const PILL_WINDOW_LABEL: &str = "pill";
pub const ASK_WINDOW_LABEL: &str = "ask";
pub const EVENT_SELECTION: &str = "pill://selection";
pub const EVENT_PERMISSION_CHANGED: &str = "pill://permission-changed";
pub const EVENT_HIDE: &str = "pill://hide";
pub const EVENT_ASK_SHOW: &str = "ask://show";
pub const EVENT_ASK_HIDE: &str = "ask://hide";

pub const MAX_CAPTURE_BYTES: usize = 100 * 1024;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectionPayload {
    pub session_id: u64,
    pub text: String,
    pub app_name: String,
    pub bundle_id: String,
    pub editable: bool,
    pub error: Option<String>,
}

pub struct Session {
    pub id: u64,
    pub text: String,
    pub pid: i32,
    pub payload: SelectionPayload,
    #[cfg(target_os = "macos")]
    pub ax_element: Option<ax::RetainedAXElement>,
}

pub struct PillState {
    pub config: Mutex<PillConfig>,
    pub session: Mutex<Option<Session>>,
    pub session_counter: AtomicU64,
}

pub fn new_pill_state() -> PillState {
    PillState {
        config: Mutex::new(PillConfig::default()),
        session: Mutex::new(None),
        session_counter: AtomicU64::new(0),
    }
}

#[cfg(target_os = "macos")]
pub fn init(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    engine::init(app)
}

#[cfg(not(target_os = "macos"))]
pub fn init(_app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
