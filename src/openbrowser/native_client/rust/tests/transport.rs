#![cfg(unix)]

use serde_json::{json, Value};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

const RESPONSE_LIMIT: usize = 8 * 1024 * 1024;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = Path::new("/tmp").join(format!(
            "ob-native-{}-{}-{}",
            tag,
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_openbrowser-ai"))
}

fn installed_binary(dir: &Path) -> PathBuf {
    let destination = dir.join("openbrowser-ai");
    fs::copy(binary(), &destination).unwrap();
    fs::set_permissions(&destination, fs::Permissions::from_mode(0o755)).unwrap();
    destination
}

fn write_backend(dir: &Path) -> PathBuf {
    let backend = dir.join("openbrowser-ai-py");
    let script = r#"#!/usr/bin/env python3
import json, os, sys
data = {
    "argv": [os.fsencode(arg).hex() for arg in sys.argv],
    "stdin": sys.stdin.buffer.read().hex(),
    "cwd": os.getcwd(),
    "test_env": os.environ.get("OB_TEST"),
}
with open(os.environ["OB_MARKER"], "w", encoding="utf-8") as handle:
    json.dump(data, handle)
sys.exit(int(os.environ.get("OB_EXIT", "0")))
"#;
    fs::write(&backend, script).unwrap();
    fs::set_permissions(&backend, fs::Permissions::from_mode(0o755)).unwrap();
    backend
}

fn args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

fn line(value: Value) -> Vec<u8> {
    let mut bytes = value.to_string().into_bytes();
    bytes.push(b'\n');
    bytes
}

struct Peer {
    count: Arc<AtomicUsize>,
    handle: JoinHandle<Vec<u8>>,
}

fn serve_once(socket: &Path, response: Option<Vec<u8>>, delay: Duration) -> Peer {
    let listener = UnixListener::bind(socket).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let accepted = count.clone();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        accepted.fetch_add(1, Ordering::SeqCst);
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if request.contains(&b'\n') {
                break;
            }
        }
        if !delay.is_zero() {
            thread::sleep(delay);
        }
        if let Some(bytes) = response {
            let _ = stream.write_all(&bytes);
            let _ = stream.flush();
        }
        request
    });
    Peer { count, handle }
}

fn listener(socket: &Path) -> UnixListener {
    UnixListener::bind(socket).unwrap()
}

fn assert_no_connection(listener: &UnixListener) {
    listener.set_nonblocking(true).unwrap();
    match listener.accept() {
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
        other => panic!("unexpected socket connection: {other:?}"),
    }
}

fn run_native(
    executable: &Path,
    home: &Path,
    socket: Option<&Path>,
    arguments: &[OsString],
    stdin: Option<&[u8]>,
    extra_env: &[(&str, &str)],
) -> Output {
    let mut command = Command::new(executable);
    command
        .current_dir(home)
        .env("HOME", home)
        .env_remove("OPENBROWSER_SOCKET")
        .env("OPENBROWSER_CONNECT_TIMEOUT", "0.2")
        .env("OPENBROWSER_READ_TIMEOUT", "1.0");
    if let Some(path) = socket {
        command.env("OPENBROWSER_SOCKET", path);
    }
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.args(arguments);
    match stdin {
        Some(bytes) => {
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child.stdin.take().unwrap().write_all(bytes).unwrap();
            child.wait_with_output().unwrap()
        }
        None => command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap(),
    }
}

fn marker(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn fast_success_sends_exact_request_and_prints_output() {
    let temp = TempDir::new("success");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let code = "print(\"café\\t雪\")\nprint('line2')";
    let peer = serve_once(
        &socket,
        Some(line(json!({
            "id": 1,
            "success": true,
            "output": "雪\nline2"
        }))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", code]),
        None,
        &[],
    );

    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "雪\nline2\n");
    assert!(output.stderr.is_empty(), "{output:?}");
    let request = peer.handle.join().unwrap();
    assert!(request.ends_with(b"\n"));
    let value: Value = serde_json::from_slice(&request).unwrap();
    assert_eq!(value["id"], 1);
    assert_eq!(value["action"], "execute");
    assert_eq!(value["code"], code);
    assert_eq!(peer.count.load(Ordering::SeqCst), 1);
}

#[test]
fn empty_output_prints_nothing() {
    let temp = TempDir::new("empty");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({"id": 1, "success": true, "output": ""}))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[],
    );

    assert!(output.status.success(), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let _ = peer.handle.join().unwrap();
}

#[test]
fn daemon_failure_preserves_output_code_and_hint() {
    let temp = TempDir::new("failure");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({
            "id": 1,
            "success": false,
            "output": "partial",
            "error": "failed",
            "error_code": "x",
            "hint": "inspect"
        }))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "raise SystemExit(1)"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "partial\nError code: x\nHint: inspect\n"
    );
    let _ = peer.handle.join().unwrap();
}

#[test]
fn daemon_failure_with_null_code_omits_error_code_line() {
    let temp = TempDir::new("null-code");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({
            "id": 1,
            "success": false,
            "output": "partial",
            "error": "failed",
            "error_code": null
        }))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "raise SystemExit(1)"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stderr), "partial\n");
    let _ = peer.handle.join().unwrap();
}

#[test]
fn id_zero_daemon_failure_is_typed_not_invalid() {
    let temp = TempDir::new("id-zero");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({
            "id": 0,
            "success": false,
            "output": "",
            "error": "boom",
            "error_code": "daemon_request_failed",
            "hint": null
        }))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "boom\nError code: daemon_request_failed\n"
    );
    let _ = peer.handle.join().unwrap();
}

#[test]
fn malformed_response_fails_closed() {
    let temp = TempDir::new("malformed");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(&socket, Some(b"not-json\n".to_vec()), Duration::ZERO);

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("Error code: daemon_invalid_response"));
    let _ = peer.handle.join().unwrap();
}

#[test]
fn partial_json_without_newline_is_accepted() {
    let temp = TempDir::new("partial-eof");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(br#"{"id":1,"success":true,"output":"ok"}"#.to_vec()),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "print('ok')"]),
        None,
        &[],
    );

    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "ok\n");
    let _ = peer.handle.join().unwrap();
}

#[test]
fn read_timeout_matches_python_message() {
    let temp = TempDir::new("timeout");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({"id": 1, "success": true, "output": "late"}))),
        Duration::from_millis(300),
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[("OPENBROWSER_READ_TIMEOUT", "0.1")],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Daemon did not respond within 0.1s"),
        "{stderr}"
    );
    assert!(stderr.contains("Error code: daemon_timeout"), "{stderr}");
    assert!(stderr.contains("The code may still be running"), "{stderr}");
    let _ = peer.handle.join().unwrap();
}

#[test]
fn connection_lost_after_write_is_terminal_and_never_falls_back() {
    let temp = TempDir::new("lost");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(&socket, None, Duration::ZERO);

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[("OB_MARKER", marker_path.to_str().unwrap())],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("Error code: daemon_connection_lost"));
    assert!(
        !marker_path.exists(),
        "native fell back after a write attempt"
    );
    let _ = peer.handle.join().unwrap();
    assert_eq!(peer.count.load(Ordering::SeqCst), 1);
}

#[test]
fn connect_failure_delegates_before_sending() {
    let temp = TempDir::new("connect-fail");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let missing = temp.path().join("missing.sock");

    let output = run_native(
        &executable,
        temp.path(),
        Some(&missing),
        &args(&["-c", "print(1)"]),
        None,
        &[("OB_MARKER", marker_path.to_str().unwrap())],
    );

    assert!(output.status.success(), "{output:?}");
    let data = marker(&marker_path);
    let argv: Vec<String> = data["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect();
    assert_eq!(argv[1], hex("-c"));
    assert_eq!(argv[2], hex("print(1)"));
}

#[test]
fn unsupported_argv_delegates_before_connect() {
    let temp = TempDir::new("unsupported");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let socket_listener = listener(&socket);

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["--version"]),
        None,
        &[("OB_MARKER", marker_path.to_str().unwrap())],
    );

    assert!(output.status.success(), "{output:?}");
    assert!(marker_path.exists());
    assert_no_connection(&socket_listener);
}

#[test]
fn dash_stdin_delegates_untouched_before_read_or_connect() {
    let temp = TempDir::new("stdin");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let socket_listener = listener(&socket);
    let stdin = b"  print('stdin')\n\xc3\xa9  ";

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "-"]),
        Some(stdin),
        &[("OB_MARKER", marker_path.to_str().unwrap())],
    );

    assert!(output.status.success(), "{output:?}");
    let data = marker(&marker_path);
    assert_eq!(data["stdin"], hex_bytes(stdin));
    assert_no_connection(&socket_listener);
}

#[test]
fn non_utf8_argv_delegates_untouched() {
    let temp = TempDir::new("non-utf8");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let socket_listener = listener(&socket);
    let arguments = vec![
        OsString::from("-c"),
        OsString::from(OsStr::from_bytes(b"print('\xff')")),
    ];

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &arguments,
        None,
        &[("OB_MARKER", marker_path.to_str().unwrap())],
    );

    assert!(output.status.success(), "{output:?}");
    let data = marker(&marker_path);
    let argv: Vec<String> = data["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect();
    assert!(argv[2].contains("ff"));
    assert_no_connection(&socket_listener);
}

#[test]
fn dash_mcp_and_extra_arguments_delegate_untouched() {
    let temp = TempDir::new("delegate-shapes");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let socket_listener = listener(&socket);

    for arguments in [
        args(&["-c", "--mcp"]),
        args(&["-c", "print(1)", "extra"]),
        args(&["--prompt", "-c", "print(1)"]),
    ] {
        let output = run_native(
            &executable,
            temp.path(),
            Some(&socket),
            &arguments,
            None,
            &[("OB_MARKER", marker_path.to_str().unwrap())],
        );
        assert!(output.status.success(), "{output:?}");
        assert!(marker_path.exists());
        let _ = fs::remove_file(&marker_path);
    }

    assert_no_connection(&socket_listener);
}

#[test]
fn missing_backend_does_not_search_path() {
    let temp = TempDir::new("missing-backend");
    let executable = installed_binary(temp.path());
    let path_dir = temp.path().join("path");
    fs::create_dir(&path_dir).unwrap();
    let fake_marker = temp.path().join("path-backend.json");
    let fake = path_dir.join("openbrowser-ai-py");
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nprintf 'executed' > '{}'\nexit 0\n",
            fake_marker.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    let existing_path = std::env::var("PATH").unwrap_or_default();

    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["--version"]),
        None,
        &[("PATH", &format!("{}:{}", path_dir.display(), existing_path))],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("openbrowser-ai-py"));
    assert!(
        !fake_marker.exists(),
        "native searched PATH for the backend"
    );
}

#[test]
fn fallback_preserves_arguments_environment_cwd_and_exit_status() {
    let temp = TempDir::new("fallback-env");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());

    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["--version", "extra"]),
        None,
        &[
            ("OB_MARKER", marker_path.to_str().unwrap()),
            ("OB_TEST", "value"),
            ("OB_EXIT", "7"),
        ],
    );

    assert_eq!(output.status.code(), Some(7), "{output:?}");
    let data = marker(&marker_path);
    assert_eq!(data["test_env"], "value");
    let expected_cwd = temp.path().canonicalize().unwrap();
    assert_eq!(data["cwd"], expected_cwd.to_str().unwrap());
    let argv: Vec<String> = data["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect();
    assert_eq!(argv[1], hex("--version"));
    assert_eq!(argv[2], hex("extra"));
}

#[test]
fn oversized_response_fails_closed() {
    let temp = TempDir::new("oversized");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let output_text = "x".repeat(RESPONSE_LIMIT + 1);
    let peer = serve_once(
        &socket,
        Some(line(
            json!({"id": 1, "success": true, "output": output_text}),
        )),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("daemon_response_too_large"));
    let _ = peer.handle.join().unwrap();
}

#[test]
fn default_socket_and_tilde_expansion_work() {
    let temp = TempDir::new("default-socket");
    let executable = installed_binary(temp.path());
    let daemon_dir = temp.path().join(".openbrowser");
    fs::create_dir(&daemon_dir).unwrap();
    let default_socket = daemon_dir.join("daemon.sock");
    let peer = serve_once(
        &default_socket,
        Some(line(json!({"id": 1, "success": true, "output": "default"}))),
        Duration::ZERO,
    );
    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["-c", "pass"]),
        None,
        &[],
    );
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "default\n");
    let _ = peer.handle.join().unwrap();

    let tilde_socket = temp.path().join("tilde.sock");
    let peer = serve_once(
        &tilde_socket,
        Some(line(json!({"id": 1, "success": true, "output": "tilde"}))),
        Duration::ZERO,
    );
    let output = run_native(
        &executable,
        temp.path(),
        Some(Path::new("~/tilde.sock")),
        &args(&["-c", "pass"]),
        None,
        &[],
    );
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "tilde\n");
    let _ = peer.handle.join().unwrap();
}

#[test]
fn nan_read_timeout_clamps_to_minimum() {
    let temp = TempDir::new("nan-timeout");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({"id": 1, "success": true, "output": "late"}))),
        Duration::from_millis(250),
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[("OPENBROWSER_READ_TIMEOUT", "nan")],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("within 0.05s"));
    let _ = peer.handle.join().unwrap();
}

#[test]
fn unsupported_timeout_parsing_delegates() {
    let temp = TempDir::new("timeout-delegate");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let socket_listener = listener(&socket);

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[
            ("OB_MARKER", marker_path.to_str().unwrap()),
            ("OPENBROWSER_READ_TIMEOUT", "1_0"),
        ],
    );

    assert!(output.status.success(), "{output:?}");
    assert!(marker_path.exists());
    assert_no_connection(&socket_listener);
}

#[test]
fn invalid_timeout_uses_default_without_fallback() {
    let temp = TempDir::new("timeout-default");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("backend.json");
    write_backend(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({"id": 1, "success": true, "output": "ok"}))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[
            ("OB_MARKER", marker_path.to_str().unwrap()),
            ("OPENBROWSER_READ_TIMEOUT", "invalid"),
        ],
    );

    assert!(output.status.success(), "{output:?}");
    assert!(!marker_path.exists(), "invalid timeout should not delegate");
    let _ = peer.handle.join().unwrap();
}

#[test]
fn recursion_guard_rejects_backend_that_is_the_native_binary() {
    let temp = TempDir::new("recursion");
    let executable = installed_binary(temp.path());
    let backend = temp.path().join("openbrowser-ai-py");
    std::os::unix::fs::symlink(&executable, &backend).unwrap();

    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["--version"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("fallback"));
}

fn hex(value: &str) -> String {
    hex_bytes(value.as_bytes())
}

fn hex_bytes(value: &[u8]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn success_with_mismatched_id_is_invalid() {
    let temp = TempDir::new("id-mismatch");
    let executable = installed_binary(temp.path());
    for response_id in [0, 2] {
        let socket = temp.path().join(format!("daemon-{response_id}.sock"));
        let peer = serve_once(
            &socket,
            Some(line(json!({
                "id": response_id,
                "success": true,
                "output": "unexpected"
            }))),
            Duration::ZERO,
        );
        let output = run_native(
            &executable,
            temp.path(),
            Some(&socket),
            &args(&["-c", "pass"]),
            None,
            &[],
        );
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        assert!(output.stdout.is_empty(), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("daemon_invalid_response"));
        let _ = peer.handle.join().unwrap();
    }
}

#[test]
fn failure_with_mismatched_id_is_invalid() {
    let temp = TempDir::new("failure-id-mismatch");
    let executable = installed_binary(temp.path());
    let socket = temp.path().join("daemon.sock");
    let peer = serve_once(
        &socket,
        Some(line(json!({
            "id": 2,
            "success": false,
            "output": "partial",
            "error": "failed"
        }))),
        Duration::ZERO,
    );

    let output = run_native(
        &executable,
        temp.path(),
        Some(&socket),
        &args(&["-c", "pass"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("daemon_invalid_response"));
    let _ = peer.handle.join().unwrap();
}

#[test]
fn copied_native_backend_is_rejected_before_exec() {
    let temp = TempDir::new("copied-backend");
    let executable = installed_binary(temp.path());
    let backend = temp.path().join("openbrowser-ai-py");
    fs::copy(&executable, &backend).unwrap();
    fs::set_permissions(&backend, fs::Permissions::from_mode(0o755)).unwrap();

    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["--version"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("script"), "{stderr}");
}

#[test]
fn hardlinked_native_backend_is_rejected_before_exec() {
    let temp = TempDir::new("hardlink-backend");
    let executable = installed_binary(temp.path());
    let backend = temp.path().join("openbrowser-ai-py");
    fs::hard_link(&executable, &backend).unwrap();

    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["--version"]),
        None,
        &[],
    );

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("same file"), "{stderr}");
}

#[test]
fn macho_and_elf_backend_magic_are_rejected() {
    let temp = TempDir::new("native-magic");
    let executable = installed_binary(temp.path());
    for (name, magic) in [
        ("macho", vec![0xcf, 0xfa, 0xed, 0xfe, 0x00]),
        ("elf", vec![0x7f, 0x45, 0x4c, 0x46, 0x00]),
    ] {
        let backend = temp.path().join("openbrowser-ai-py");
        write_executable(&backend, &magic);
        let output = run_native(
            &executable,
            temp.path(),
            None,
            &args(&["--version"]),
            None,
            &[],
        );
        assert_eq!(output.status.code(), Some(1), "{name}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("script"),
            "{name}: {output:?}"
        );
        fs::remove_file(&backend).unwrap();
    }
}

#[test]
fn shell_shebang_with_spaces_is_accepted() {
    let temp = TempDir::new("shell-shebang");
    let executable = installed_binary(temp.path());
    let marker_path = temp.path().join("shell-marker");
    let backend = temp.path().join("openbrowser-ai-py");
    write_executable(
        &backend,
        b"#!/bin/sh -e\nprintf 'executed' > \"$OB_MARKER\"\nexit 0\n",
    );

    let output = run_native(
        &executable,
        temp.path(),
        None,
        &args(&["--version"]),
        None,
        &[("OB_MARKER", marker_path.to_str().unwrap())],
    );

    assert!(output.status.success(), "{output:?}");
    assert!(marker_path.exists());
    assert_eq!(fs::read_to_string(marker_path).unwrap(), "executed");
}

fn write_executable(path: &Path, contents: &[u8]) {
    fs::write(path, contents).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
