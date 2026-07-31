#!/usr/bin/env python3
import argparse
import ctypes
import os
import time
from ctypes.util import find_library


CLIENT_MESSAGE = 33
SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
SUBSTRUCTURE_REDIRECT_MASK = 1 << 20


class XClassHint(ctypes.Structure):
    _fields_ = [("res_name", ctypes.c_void_p), ("res_class", ctypes.c_void_p)]


class ClientMessageData(ctypes.Union):
    _fields_ = [
        ("b", ctypes.c_char * 20),
        ("s", ctypes.c_short * 10),
        ("l", ctypes.c_long * 5),
    ]


class XClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", ClientMessageData),
    ]


class XEvent(ctypes.Union):
    _fields_ = [
        ("xclient", XClientMessageEvent),
        ("padding", ctypes.c_long * 24),
    ]


def load_x11():
    library_name = find_library("X11") or "libX11.so.6"
    x11 = ctypes.cdll.LoadLibrary(library_name)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
    x11.XDefaultScreen.restype = ctypes.c_int
    x11.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XDisplayWidth.restype = ctypes.c_int
    x11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XDisplayHeight.restype = ctypes.c_int
    x11.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XRootWindow.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XGetWindowProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    x11.XGetWindowProperty.restype = ctypes.c_int
    x11.XGetClassHint.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(XClassHint),
    ]
    x11.XGetClassHint.restype = ctypes.c_int
    x11.XFetchName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    x11.XFetchName.restype = ctypes.c_int
    x11.XSendEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_long,
        ctypes.POINTER(XEvent),
    ]
    x11.XSendEvent.restype = ctypes.c_int
    x11.XResizeWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    x11.XResizeWindow.restype = ctypes.c_int
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XFree.argtypes = [ctypes.c_void_p]
    return x11


def atom(x11, display, name):
    return x11.XInternAtom(display, name.encode("ascii"), False)


def get_property(x11, display, window, property_atom):
    actual_type = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    item_count = ctypes.c_ulong()
    bytes_after = ctypes.c_ulong()
    data_pointer = ctypes.c_void_p()
    status = x11.XGetWindowProperty(
        display,
        window,
        property_atom,
        0,
        4096,
        False,
        0,
        ctypes.byref(actual_type),
        ctypes.byref(actual_format),
        ctypes.byref(item_count),
        ctypes.byref(bytes_after),
        ctypes.byref(data_pointer),
    )
    if status != 0 or not data_pointer.value:
        return []

    try:
        if actual_format.value == 32:
            values = ctypes.cast(
                data_pointer, ctypes.POINTER(ctypes.c_ulong)
            )
            return [int(values[index]) for index in range(item_count.value)]
        if actual_format.value == 8:
            return ctypes.string_at(data_pointer, item_count.value)
        return []
    finally:
        x11.XFree(data_pointer)


def decode_pointer(pointer):
    if not pointer:
        return ""
    return ctypes.string_at(pointer).decode("utf-8", errors="replace")


def window_identity(x11, display, window):
    hint = XClassHint()
    resource_name = ""
    resource_class = ""
    if x11.XGetClassHint(display, window, ctypes.byref(hint)):
        try:
            resource_name = decode_pointer(hint.res_name)
            resource_class = decode_pointer(hint.res_class)
        finally:
            if hint.res_name:
                x11.XFree(hint.res_name)
            if hint.res_class:
                x11.XFree(hint.res_class)

    title_pointer = ctypes.c_void_p()
    title = ""
    if x11.XFetchName(display, window, ctypes.byref(title_pointer)):
        try:
            title = decode_pointer(title_pointer.value)
        finally:
            if title_pointer.value:
                x11.XFree(title_pointer)
    return resource_name, resource_class, title


def classify_window(identity):
    resource_name, resource_class, title = identity
    window_class = f"{resource_name} {resource_class}".lower()
    normalized_title = title.lower().strip()
    # Both programs create untitled splash/loading windows before their main
    # window. Waiting for a title avoids tiling a short-lived window ID.
    if not normalized_title:
        return None
    if "gzclient" in window_class or "gazebo" in window_class:
        return "gazebo"
    if "rviz" in window_class:
        return "rviz"
    if normalized_title == "gazebo":
        return "gazebo"
    if normalized_title.endswith("rviz2"):
        return "rviz"
    return None


def send_client_message(x11, display, root, window, message_type, values):
    event = XEvent()
    event.xclient.type = CLIENT_MESSAGE
    event.xclient.serial = 0
    event.xclient.send_event = True
    event.xclient.display = display
    event.xclient.window = window
    event.xclient.message_type = message_type
    event.xclient.format = 32
    for index, value in enumerate(values[:5]):
        event.xclient.data.l[index] = int(value)
    mask = SUBSTRUCTURE_REDIRECT_MASK | SUBSTRUCTURE_NOTIFY_MASK
    x11.XSendEvent(display, root, False, mask, ctypes.byref(event))


def tile_window(x11, display, root, window, geometry):
    state_atom = atom(x11, display, "_NET_WM_STATE")
    maximize_horizontal = atom(x11, display, "_NET_WM_STATE_MAXIMIZED_HORZ")
    maximize_vertical = atom(x11, display, "_NET_WM_STATE_MAXIMIZED_VERT")
    send_client_message(
        x11,
        display,
        root,
        window,
        state_atom,
        [0, maximize_horizontal, maximize_vertical, 1, 0],
    )

    x, y, width, height = geometry
    frame_extents = get_property(
        x11, display, window, atom(x11, display, "_NET_FRAME_EXTENTS")
    )
    left, right, top, bottom = (
        frame_extents[:4] if len(frame_extents) >= 4 else (0, 0, 0, 0)
    )
    client_width = max(width - left - right, 100)
    client_height = max(height - top - bottom, 100)
    move_resize_atom = atom(x11, display, "_NET_MOVERESIZE_WINDOW")
    position_flags = (1 << 8) | (1 << 9)
    send_client_message(
        x11,
        display,
        root,
        window,
        move_resize_atom,
        [position_flags, x, y, 0, 0],
    )
    x11.XResizeWindow(display, window, client_width, client_height)
    x11.XFlush(display)


def desktop_work_area(x11, display, root, screen):
    work_areas = get_property(x11, display, root, atom(x11, display, "_NET_WORKAREA"))
    current_desktop = get_property(
        x11, display, root, atom(x11, display, "_NET_CURRENT_DESKTOP")
    )
    desktop_index = current_desktop[0] if current_desktop else 0
    offset = desktop_index * 4
    if len(work_areas) >= offset + 4:
        x, y, width, height = work_areas[offset:offset + 4]
        if width > 0 and height > 0:
            return x, y, width, height
    return 0, 0, x11.XDisplayWidth(display, screen), x11.XDisplayHeight(display, screen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--settle-time", type=float, default=8.0)
    args = parser.parse_args()

    if not os.environ.get("DISPLAY"):
        print("Window tiling skipped: DISPLAY is not set", flush=True)
        return

    x11 = load_x11()
    display = x11.XOpenDisplay(None)
    if not display:
        print(f"Window tiling skipped: cannot open DISPLAY={os.environ['DISPLAY']}", flush=True)
        return

    try:
        screen = x11.XDefaultScreen(display)
        root = x11.XRootWindow(display, screen)
        x, y, width, height = desktop_work_area(x11, display, root, screen)
        left_width = width // 2
        geometries = {
            "gazebo": (x, y, left_width, height),
            "rviz": (x + left_width, y, width - left_width, height),
        }
        client_list_atom = atom(x11, display, "_NET_CLIENT_LIST")
        deadline = time.monotonic() + max(args.timeout, 1.0)
        matched = {}
        complete_since = None

        while time.monotonic() < deadline:
            current_matches = {}
            for window in get_property(x11, display, root, client_list_atom):
                identity = window_identity(x11, display, window)
                role = classify_window(identity)
                if role:
                    current_matches[role] = (window, identity)

            next_matched = {
                role: window_and_identity[0]
                for role, window_and_identity in current_matches.items()
            }
            if next_matched != matched:
                complete_since = None
                matched = next_matched
                for role, window in matched.items():
                    identity = current_matches[role][1]
                    print(
                        f"Found {role} window 0x{window:x}: "
                        f"class={identity[0]}/{identity[1]} title={identity[2]!r}",
                        flush=True,
                    )

            for role, window in matched.items():
                tile_window(x11, display, root, window, geometries[role])

            if "gazebo" in matched and "rviz" in matched:
                if complete_since is None:
                    complete_since = time.monotonic()
                if time.monotonic() - complete_since >= max(args.settle_time, 1.0):
                    print(
                        f"Tiled Gazebo left and RViz right in work area "
                        f"{width}x{height}+{x}+{y}",
                        flush=True,
                    )
                    return
            time.sleep(0.5)

        missing = sorted({"gazebo", "rviz"} - set(matched))
        print(f"Window tiling timed out; missing: {', '.join(missing)}", flush=True)
    finally:
        x11.XCloseDisplay(display)


if __name__ == "__main__":
    main()
