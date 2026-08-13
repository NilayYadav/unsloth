// Every element gets a short messaging timeout so an unresponsive target app
// cannot hang a thread for long; callers must still run captures off the main
// thread.

use core_foundation::base::{Boolean, CFRange, CFRelease, CFRetain, CFTypeRef, TCFType};
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::{CFDictionary, CFDictionaryRef};
use core_foundation::string::{CFString, CFStringRef};
use core_graphics::geometry::CGRect;
use std::ffi::c_void;

pub type AXUIElementRef = *const c_void;
type AXValueRef = *const c_void;
type AXError = i32;

const AX_SUCCESS: AXError = 0;
const AX_VALUE_TYPE_CGRECT: u32 = 3;
const AX_VALUE_TYPE_CFRANGE: u32 = 4;

const MESSAGING_TIMEOUT_SECONDS: f32 = 0.3;

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    static kAXTrustedCheckOptionPrompt: CFStringRef;

    fn AXIsProcessTrusted() -> Boolean;
    fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> Boolean;
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> AXError;
    fn AXUIElementIsAttributeSettable(
        element: AXUIElementRef,
        attribute: CFStringRef,
        settable: *mut Boolean,
    ) -> AXError;
    fn AXUIElementCopyParameterizedAttributeValue(
        element: AXUIElementRef,
        parameterized_attribute: CFStringRef,
        parameter: CFTypeRef,
        result: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementSetMessagingTimeout(element: AXUIElementRef, timeout: f32) -> AXError;
    fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut i32) -> AXError;
    fn AXValueGetValue(value: AXValueRef, value_type: u32, out: *mut c_void) -> Boolean;
    fn AXValueCreate(value_type: u32, value_ptr: *const c_void) -> AXValueRef;
}

#[link(name = "Carbon", kind = "framework")]
extern "C" {
    fn IsSecureEventInputEnabled() -> Boolean;
}

/// Owned, retained AXUIElement that can cross threads. AX calls are IPC and
/// safe from any thread.
pub struct RetainedAXElement(AXUIElementRef);

unsafe impl Send for RetainedAXElement {}

impl RetainedAXElement {
    /// Takes ownership of a +1 reference.
    fn from_owned(raw: AXUIElementRef) -> Self {
        Self(raw)
    }

    pub fn as_raw(&self) -> AXUIElementRef {
        self.0
    }
}

impl Clone for RetainedAXElement {
    fn clone(&self) -> Self {
        unsafe { CFRetain(self.0 as CFTypeRef) };
        Self(self.0)
    }
}

impl Drop for RetainedAXElement {
    fn drop(&mut self) {
        unsafe { CFRelease(self.0 as CFTypeRef) };
    }
}

pub fn is_process_trusted() -> bool {
    unsafe { AXIsProcessTrusted() != 0 }
}

pub fn request_process_trust() -> bool {
    let prompt_key = unsafe { CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt) };
    let options = CFDictionary::from_CFType_pairs(&[(prompt_key, CFBoolean::true_value())]);
    unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef()) != 0 }
}

pub fn is_secure_event_input_enabled() -> bool {
    unsafe { IsSecureEventInputEnabled() != 0 }
}

fn ax_attr(element: AXUIElementRef, attribute: &str) -> Option<CFTypeRef> {
    let attr = CFString::new(attribute);
    let mut value: CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element, attr.as_concrete_TypeRef(), &mut value)
    };
    if err == AX_SUCCESS && !value.is_null() {
        Some(value)
    } else {
        None
    }
}

fn ax_attr_string(element: AXUIElementRef, attribute: &str) -> Option<String> {
    let value = ax_attr(element, attribute)?;
    let text = unsafe { CFString::wrap_under_create_rule(value as CFStringRef) }.to_string();
    Some(text)
}

fn ax_attr_element(element: AXUIElementRef, attribute: &str) -> Option<RetainedAXElement> {
    let value = ax_attr(element, attribute)?;
    let owned = RetainedAXElement::from_owned(value as AXUIElementRef);
    unsafe { AXUIElementSetMessagingTimeout(owned.as_raw(), MESSAGING_TIMEOUT_SECONDS) };
    Some(owned)
}

fn ax_attr_range(element: AXUIElementRef, attribute: &str) -> Option<CFRange> {
    let value = ax_attr(element, attribute)?;
    let mut range = CFRange { location: 0, length: 0 };
    let ok = unsafe {
        AXValueGetValue(
            value as AXValueRef,
            AX_VALUE_TYPE_CFRANGE,
            &mut range as *mut CFRange as *mut c_void,
        )
    };
    unsafe { CFRelease(value) };
    if ok != 0 {
        Some(range)
    } else {
        None
    }
}

fn ax_attr_settable(element: AXUIElementRef, attribute: &str) -> bool {
    let attr = CFString::new(attribute);
    let mut settable: Boolean = 0;
    let err = unsafe {
        AXUIElementIsAttributeSettable(element, attr.as_concrete_TypeRef(), &mut settable)
    };
    err == AX_SUCCESS && settable != 0
}

fn ax_param_attr(element: AXUIElementRef, attribute: &str, range: CFRange) -> Option<CFTypeRef> {
    let param = unsafe {
        AXValueCreate(AX_VALUE_TYPE_CFRANGE, &range as *const CFRange as *const c_void)
    };
    if param.is_null() {
        return None;
    }
    let attr = CFString::new(attribute);
    let mut value: CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyParameterizedAttributeValue(
            element,
            attr.as_concrete_TypeRef(),
            param as CFTypeRef,
            &mut value,
        )
    };
    unsafe { CFRelease(param as CFTypeRef) };
    if err == AX_SUCCESS && !value.is_null() {
        Some(value)
    } else {
        None
    }
}

fn ax_set_bool(element: AXUIElementRef, attribute: &str, value: bool) {
    let attr = CFString::new(attribute);
    let flag = CFBoolean::from(value);
    unsafe {
        AXUIElementSetAttributeValue(
            element,
            attr.as_concrete_TypeRef(),
            flag.as_CFTypeRef(),
        );
    }
}

fn system_wide() -> RetainedAXElement {
    let element = unsafe { AXUIElementCreateSystemWide() };
    unsafe { AXUIElementSetMessagingTimeout(element, MESSAGING_TIMEOUT_SECONDS) };
    RetainedAXElement::from_owned(element)
}

pub struct AxCapture {
    pub text: String,
    pub pid: i32,
    pub editable: bool,
    pub bounds: Option<crate::selection_pill::geometry::Rect>,
    pub element: RetainedAXElement,
}

/// Must run off the main thread.
pub fn capture_selection() -> Result<AxCapture, &'static str> {
    if !is_process_trusted() {
        return Err("not-trusted");
    }
    if is_secure_event_input_enabled() {
        return Err("secure-input");
    }

    let root = system_wide();
    let focused = ax_attr_element(root.as_raw(), "AXFocusedUIElement")
        .ok_or("no-focused-element")?;

    let mut pid: i32 = 0;
    unsafe { AXUIElementGetPid(focused.as_raw(), &mut pid) };

    let role = ax_attr_string(focused.as_raw(), "AXRole").unwrap_or_default();
    let subrole = ax_attr_string(focused.as_raw(), "AXSubrole").unwrap_or_default();
    if role == "AXSecureTextField" || subrole == "AXSecureTextField" {
        return Err("secure-input");
    }

    let mut text = ax_attr_string(focused.as_raw(), "AXSelectedText").unwrap_or_default();
    if text.is_empty() {
        // Chromium/Electron expose nothing until their AX tree is force-enabled.
        if pid != 0 {
            let app = unsafe { AXUIElementCreateApplication(pid) };
            if !app.is_null() {
                unsafe { AXUIElementSetMessagingTimeout(app, MESSAGING_TIMEOUT_SECONDS) };
                ax_set_bool(app, "AXManualAccessibility", true);
                ax_set_bool(app, "AXEnhancedUserInterface", true);
                unsafe { CFRelease(app as CFTypeRef) };
            }
            text = ax_attr_string(focused.as_raw(), "AXSelectedText").unwrap_or_default();
        }
    }
    if text.is_empty() {
        if let Some(range) = ax_attr_range(focused.as_raw(), "AXSelectedTextRange") {
            if range.length > 0 {
                text = string_for_range(focused.as_raw(), range).unwrap_or_default();
            }
        }
    }
    if text.is_empty() {
        return Err("no-selection");
    }

    let bounds = ax_attr_range(focused.as_raw(), "AXSelectedTextRange")
        .and_then(|range| bounds_for_range(focused.as_raw(), range));
    let editable = ax_attr_settable(focused.as_raw(), "AXSelectedText");

    Ok(AxCapture { text, pid, editable, bounds, element: focused })
}

fn string_for_range(element: AXUIElementRef, range: CFRange) -> Option<String> {
    let value = ax_param_attr(element, "AXStringForRange", range)?;
    Some(unsafe { CFString::wrap_under_create_rule(value as CFStringRef) }.to_string())
}

/// Screen bounds of the selected range in global top-left-origin coordinates.
fn bounds_for_range(
    element: AXUIElementRef,
    range: CFRange,
) -> Option<crate::selection_pill::geometry::Rect> {
    let value = ax_param_attr(element, "AXBoundsForRange", range)?;
    let mut rect = CGRect::default();
    let ok = unsafe {
        AXValueGetValue(
            value as AXValueRef,
            AX_VALUE_TYPE_CGRECT,
            &mut rect as *mut CGRect as *mut c_void,
        )
    };
    unsafe { CFRelease(value) };
    if ok == 0 || (rect.size.width <= 0.0 && rect.size.height <= 0.0) {
        return None;
    }
    Some(crate::selection_pill::geometry::Rect::new(
        rect.origin.x,
        rect.origin.y,
        rect.size.width,
        rect.size.height,
    ))
}

pub fn set_selected_text(element: &RetainedAXElement, text: &str) -> bool {
    if !ax_attr_settable(element.as_raw(), "AXSelectedText") {
        return false;
    }
    let attr = CFString::new("AXSelectedText");
    let value = CFString::new(text);
    let err = unsafe {
        AXUIElementSetAttributeValue(
            element.as_raw(),
            attr.as_concrete_TypeRef(),
            value.as_CFTypeRef(),
        )
    };
    err == AX_SUCCESS
}

pub fn focused_app_pid() -> Option<i32> {
    let root = system_wide();
    let focused = ax_attr_element(root.as_raw(), "AXFocusedUIElement")?;
    let mut pid: i32 = 0;
    let err = unsafe { AXUIElementGetPid(focused.as_raw(), &mut pid) };
    if err == AX_SUCCESS && pid != 0 {
        Some(pid)
    } else {
        None
    }
}
