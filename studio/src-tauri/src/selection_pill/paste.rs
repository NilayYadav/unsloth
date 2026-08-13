use super::ax;
use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString, NSRunningApplication};
use objc2_foundation::NSString;
use std::thread::sleep;
use std::time::Duration;

const KEY_V: u16 = 9;
const PASTE_SETTLE: Duration = Duration::from_millis(250);
const ACTIVATE_SETTLE: Duration = Duration::from_millis(120);

pub fn apply(
    element: Option<&ax::RetainedAXElement>,
    session_pid: i32,
    text: &str,
) -> Result<(), &'static str> {
    if ax::is_secure_event_input_enabled() {
        return Err("secure-input");
    }

    if let Some(element) = element {
        if ax::set_selected_text(element, text) {
            return Ok(());
        }
    }

    // Fallback: pasteboard swap + synthetic Cmd+V into the session's app.
    ensure_frontmost(session_pid)?;

    let pasteboard = NSPasteboard::generalPasteboard();
    let saved = unsafe { pasteboard.stringForType(NSPasteboardTypeString) }
        .map(|s| s.to_string());
    let had_non_text_content = saved.is_none()
        && pasteboard
            .pasteboardItems()
            .map(|items| items.count() > 0)
            .unwrap_or(false);

    pasteboard.clearContents();
    unsafe {
        pasteboard.setString_forType(&NSString::from_str(text), NSPasteboardTypeString);
    }
    let our_change = pasteboard.changeCount();

    post_cmd_v()?;
    sleep(PASTE_SETTLE);

    // Restore only if nothing else touched the pasteboard and we would not
    // clobber richer content we could not snapshot.
    let current_change = pasteboard.changeCount();
    if current_change == our_change && !had_non_text_content {
        if let Some(saved) = saved {
            pasteboard.clearContents();
            unsafe {
                pasteboard.setString_forType(
                    &NSString::from_str(&saved),
                    NSPasteboardTypeString,
                );
            }
        }
    }
    Ok(())
}

fn ensure_frontmost(session_pid: i32) -> Result<(), &'static str> {
    if ax::focused_app_pid() == Some(session_pid) {
        return Ok(());
    }
    let app = NSRunningApplication::runningApplicationWithProcessIdentifier(session_pid)
        .ok_or("target-gone")?;
    #[allow(deprecated)]
    app.activateWithOptions(objc2_app_kit::NSApplicationActivationOptions::empty());
    sleep(ACTIVATE_SETTLE);
    if ax::focused_app_pid() == Some(session_pid) {
        Ok(())
    } else {
        Err("target-changed")
    }
}

fn post_cmd_v() -> Result<(), &'static str> {
    let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .map_err(|_| "event-source")?;
    let down = CGEvent::new_keyboard_event(source.clone(), KEY_V, true)
        .map_err(|_| "event-create")?;
    down.set_flags(CGEventFlags::CGEventFlagCommand);
    let up = CGEvent::new_keyboard_event(source, KEY_V, false)
        .map_err(|_| "event-create")?;
    up.set_flags(CGEventFlags::CGEventFlagCommand);
    down.post(CGEventTapLocation::HID);
    up.post(CGEventTapLocation::HID);
    Ok(())
}
