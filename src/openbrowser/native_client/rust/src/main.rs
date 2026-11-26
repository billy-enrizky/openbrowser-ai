#![cfg_attr(not(unix), allow(dead_code))]

use serde_json::{json, Value};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const DEFAULT_CONNECT_TIMEOUT: f64 = 1.0;
const DEFAULT_READ_TIMEOUT: f64 = 140.0;
const MIN_TIMEOUT: f64 = 0.05;
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const BACKEND_NAME: &str = "openbrowser-ai-py";

#[derive(Debug)]
struct Failure {
    code: String,
    output: String,
    detail: String,
    hint: String,
}

impl Failure {
    fn invalid(detail: impl Into<String>) -> Self {
        Self {
            code: "daemon_invalid_response".to_string(),
            output: String::new(),
            detail: detail.into(),
            hint: String::new(),
        }
    }

    fn timeout(read_timeout: f64) -> Self {
        Self {
            code: "daemon_timeout".to_string(),
            output: String::new(),
            detail: format!(
                "Daemon did not respond within {}s",
                format_timeout(read_timeout)
            ),
            hint:
                "The code may still be running. Run `openbrowser-ai daemon status` before retrying."
                    .to_string(),
        }
    }

    fn connection_lost(detail: impl Into<String>) -> Self {
        Self {
            code: "daemon_connection_lost".to_string(),
            output: String::new(),
            detail: detail.into(),
            hint: "The code may have partially run. Verify the page before retrying.".to_string(),
        }
    }

    fn eof() -> Self {
        Self::connection_lost("Daemon closed the connection without responding")
    }

    fn too_large() -> Self {
        Self {
            code: "daemon_response_too_large".to_string(),
            output: String::new(),
            detail: "Daemon response exceeds the client bound".to_string(),
            hint: String::new(),
        }
    }
}

fn main() {
    let arguments: Vec<OsString> = env::args_os().collect();
    match classify(&arguments) {
        Mode::Delegate => delegate_or_exit(&arguments),
        Mode::Fast(code) => match run_fast(&code, &arguments) {
            Ok(output) => {
                if !output.is_empty() {
                    println!("{output}");
                }
            }
            Err(failure) => print_failure(&failure),
        },
    }
}

enum Mode {
    Fast(String),
    Delegate,
}

fn classify(arguments: &[OsString]) -> Mode {
    if arguments.len() != 3 || arguments[1] != "-c" {
        return Mode::Delegate;
    }
    let code = match arguments[2].to_str() {
        Some(code) if code != "-" && code != "--mcp" => code,
        _ => return Mode::Delegate,
    };
    Mode::Fast(code.to_string())
}

fn print_failure(failure: &Failure) -> ! {
    let first = if !failure.output.is_empty() {
        failure.output.clone()
    } else if !failure.detail.is_empty() {
        failure.detail.clone()
    } else {
        "None".to_string()
    };
    eprintln!("{first}");
    if !failure.code.is_empty() {
        eprintln!("Error code: {}", failure.code);
    }
    if !failure.hint.is_empty() {
        eprintln!("Hint: {}", failure.hint);
    }
    std::process::exit(1);
}

#[cfg(unix)]
fn run_fast(code: &str, arguments: &[OsString]) -> Result<String, Failure> {
    let connect_timeout =
        match parse_timeout("OPENBROWSER_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT) {
            TimeoutSpec::Delegate => delegate_or_exit(arguments),
            TimeoutSpec::Value(value) => value,
        };
    let read_timeout = match parse_timeout("OPENBROWSER_READ_TIMEOUT", DEFAULT_READ_TIMEOUT) {
        TimeoutSpec::Delegate => delegate_or_exit(arguments),
        TimeoutSpec::Value(value) => value,
    };
    let path = match socket_path() {
        Ok(path) => path,
        Err(()) => delegate_or_exit(arguments),
    };
    let mut stream = match connect(&path, connect_timeout) {
        Ok(stream) => stream,
        Err(()) => delegate_or_exit(arguments),
    };
    let connect_duration = match duration_from_seconds(connect_timeout) {
        Some(duration) => duration,
        None => delegate_or_exit(arguments),
    };
    if stream.set_write_timeout(Some(connect_duration)).is_err() {
        delegate_or_exit(arguments);
    }
    let request = encode_request(code);
    write_request(&mut stream, &request, connect_duration, read_timeout)?;
    let raw = read_frame(&mut stream, read_timeout, MAX_RESPONSE_BYTES)?;
    parse_response(&raw)
}

#[cfg(not(unix))]
fn run_fast(_code: &str, arguments: &[OsString]) -> Result<String, Failure> {
    delegate_or_exit(arguments)
}

#[cfg(unix)]
fn delegate_or_exit(arguments: &[OsString]) -> ! {
    use std::os::unix::process::CommandExt;

    let backend = match backend_path() {
        Ok(backend) => backend,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let mut command = Command::new(&backend);
    if let Some(name) = arguments.first() {
        command.arg0(name);
    }
    command.args(&arguments[1..]);
    let error = command.exec();
    eprintln!("Failed to execute {}: {error}", backend.display());
    std::process::exit(1);
}

#[cfg(not(unix))]
fn delegate_or_exit(arguments: &[OsString]) -> ! {
    let backend = match backend_path() {
        Ok(backend) => backend,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let status = Command::new(&backend).args(&arguments[1..]).status();
    match status {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!("Failed to execute {}: {error}", backend.display());
            std::process::exit(1);
        }
    }
}

fn backend_path() -> Result<PathBuf, String> {
    let executable = env::current_exe()
        .map_err(|error| format!("Unable to locate the native client: {error}"))?;
    let executable = executable.canonicalize().unwrap_or(executable);
    let parent = executable
        .parent()
        .ok_or_else(|| "Unable to locate the native client directory".to_string())?;
    let candidate = parent.join(BACKEND_NAME);
    let metadata = fs::symlink_metadata(&candidate).map_err(|_| {
        format!(
            "Missing Python fallback {BACKEND_NAME} next to the native client; reinstall openbrowser-ai"
        )
    })?;
    let target = if metadata.file_type().is_symlink() {
        candidate
            .canonicalize()
            .map_err(|error| format!("Unable to resolve the Python fallback: {error}"))?
    } else {
        candidate
    };
    let target_metadata = fs::metadata(&target)
        .map_err(|error| format!("Unable to read the Python fallback: {error}"))?;
    if !target_metadata.is_file() {
        return Err(format!(
            "Python fallback {BACKEND_NAME} is not a regular file"
        ));
    }
    if target == executable {
        return Err(format!(
            "Refusing native fallback: {BACKEND_NAME} resolves to the native client"
        ));
    }
    if target.parent() != Some(parent) {
        return Err(format!(
            "Refusing native fallback: {BACKEND_NAME} is outside the native client directory"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let executable_metadata = fs::metadata(&executable)
            .map_err(|error| format!("Unable to read the native client: {error}"))?;
        if target_metadata.dev() == executable_metadata.dev()
            && target_metadata.ino() == executable_metadata.ino()
        {
            return Err(format!(
                "Refusing native fallback: {BACKEND_NAME} is the same file as the native client"
            ));
        }
        if target_metadata.permissions().mode() & 0o111 == 0 {
            return Err(format!("Python fallback {BACKEND_NAME} is not executable"));
        }
        let mut prefix = [0_u8; 4];
        let mut file = fs::File::open(&target)
            .map_err(|error| format!("Unable to read the Python fallback: {error}"))?;
        let count = file
            .read(&mut prefix)
            .map_err(|error| format!("Unable to read the Python fallback: {error}"))?;
        let prefix = &prefix[..count];
        if is_native_magic(prefix) {
            return Err(format!(
                "Python fallback {BACKEND_NAME} is a native binary, not a launcher script"
            ));
        }
        if !prefix.starts_with(b"#!") {
            return Err(format!(
                "Python fallback {BACKEND_NAME} is not a launcher script"
            ));
        }
    }
    Ok(target)
}

fn encode_request(code: &str) -> Vec<u8> {
    let mut bytes = json!({"id": 1, "action": "execute", "code": code})
        .to_string()
        .into_bytes();
    bytes.push(b'\n');
    bytes
}

#[cfg(unix)]
fn connect(path: &Path, timeout: f64) -> Result<std::os::unix::net::UnixStream, ()> {
    use socket2::{Domain, SockAddr, Socket, Type};
    use std::os::unix::net::UnixStream;

    let duration = duration_from_seconds(timeout).ok_or(())?;
    let socket = Socket::new(Domain::UNIX, Type::STREAM, None).map_err(|_| ())?;
    let address = SockAddr::unix(path).map_err(|_| ())?;
    socket.connect_timeout(&address, duration).map_err(|_| ())?;
    socket.set_nonblocking(false).map_err(|_| ())?;
    let owned: std::os::fd::OwnedFd = socket.into();
    Ok(UnixStream::from(owned))
}

fn duration_from_seconds(seconds: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(seconds).ok()
}

#[cfg(unix)]
trait WriteBudget {
    fn budget_write(&mut self, buffer: &[u8]) -> io::Result<usize>;

    fn budget_timeout(&mut self, timeout: Duration) -> io::Result<()>;
}

#[cfg(unix)]
struct SocketWriter<'a>(&'a mut std::os::unix::net::UnixStream);

#[cfg(unix)]
impl WriteBudget for SocketWriter<'_> {
    fn budget_write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.write(buffer)
    }

    fn budget_timeout(&mut self, timeout: Duration) -> io::Result<()> {
        self.0.set_write_timeout(Some(timeout))
    }
}

#[cfg(unix)]
fn write_request(
    stream: &mut std::os::unix::net::UnixStream,
    request: &[u8],
    budget: Duration,
    read_timeout: f64,
) -> Result<(), Failure> {
    let mut writer = SocketWriter(stream);
    write_request_loop(&mut writer, request, budget, read_timeout)
}

#[cfg(unix)]
fn write_request_loop<W: WriteBudget>(
    writer: &mut W,
    request: &[u8],
    budget: Duration,
    read_timeout: f64,
) -> Result<(), Failure> {
    let deadline = Instant::now() + budget;
    let mut offset = 0;
    while offset < request.len() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| Failure::timeout(read_timeout))?;
        writer.budget_timeout(remaining).map_err(|error| {
            Failure::connection_lost(format!("Daemon connection lost: {error}"))
        })?;
        match writer.budget_write(&request[offset..]) {
            Ok(0) => {
                return Err(Failure::connection_lost(
                    "Daemon connection lost: write returned zero",
                ));
            }
            Ok(written) => offset += written,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(write_failure(error, read_timeout)),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn is_native_magic(prefix: &[u8]) -> bool {
    prefix.starts_with(b"\x7fELF")
        || prefix.starts_with(&[0xfe, 0xed, 0xfa, 0xce])
        || prefix.starts_with(&[0xce, 0xfa, 0xed, 0xfe])
        || prefix.starts_with(&[0xfe, 0xed, 0xfa, 0xcf])
        || prefix.starts_with(&[0xcf, 0xfa, 0xed, 0xfe])
        || prefix.starts_with(&[0xca, 0xfe, 0xba, 0xbe])
        || prefix.starts_with(&[0xbe, 0xba, 0xfe, 0xca])
        || prefix.starts_with(&[0xca, 0xfe, 0xba, 0xbf])
        || prefix.starts_with(&[0xbf, 0xba, 0xfe, 0xca])
}

#[cfg(unix)]
fn read_frame(
    stream: &mut std::os::unix::net::UnixStream,
    read_timeout: f64,
    limit: usize,
) -> Result<Vec<u8>, Failure> {
    let duration = match duration_from_seconds(read_timeout) {
        Some(duration) => duration,
        None => return Err(Failure::timeout(read_timeout)),
    };
    stream
        .set_read_timeout(Some(duration))
        .map_err(|error| Failure::connection_lost(format!("Daemon connection lost: {error}")))?;
    let deadline = Instant::now() + duration;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| Failure::timeout(read_timeout))?;
        match stream.set_read_timeout(Some(remaining)) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::InvalidInput => {}
            Err(error) => {
                return Err(Failure::connection_lost(format!(
                    "Daemon connection lost: {error}"
                )));
            }
        }
        let count = match stream.read(&mut chunk) {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error)
                if error.kind() == io::ErrorKind::TimedOut
                    || error.kind() == io::ErrorKind::WouldBlock =>
            {
                return Err(Failure::timeout(read_timeout));
            }
            Err(error) => {
                return Err(Failure::connection_lost(format!(
                    "Daemon connection lost: {error}"
                )));
            }
        };
        if count == 0 {
            if bytes.is_empty() {
                return Err(Failure::eof());
            }
            return Ok(bytes);
        }
        let part = &chunk[..count];
        let newline = part.iter().position(|byte| *byte == b'\n');
        let taken = newline.unwrap_or(count);
        if bytes.len() + taken > limit {
            return Err(Failure::too_large());
        }
        bytes.extend_from_slice(&part[..taken]);
        if let Some(position) = newline {
            if part[position + 1..]
                .iter()
                .any(|byte| !byte.is_ascii_whitespace())
            {
                return Err(Failure::invalid("Multiple daemon frames"));
            }
            return Ok(bytes);
        }
    }
}

fn parse_response(raw: &[u8]) -> Result<String, Failure> {
    let value: Value = serde_json::from_slice(raw)
        .map_err(|error| Failure::invalid(format!("Invalid daemon response: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| Failure::invalid("Daemon returned a non-object response"))?;
    let response_id = match object.get("id") {
        Some(Value::Number(number)) => number.as_u64(),
        None => None,
        Some(_) => return Err(Failure::invalid("Invalid daemon response id")),
    };
    let success = match object.get("success") {
        Some(Value::Bool(success)) => *success,
        None => false,
        Some(_) => return Err(Failure::invalid("Invalid daemon success field")),
    };
    let output = text_field(object, "output")?;
    if success {
        if response_id != Some(1) {
            return Err(Failure::invalid("Unexpected daemon response id"));
        }
        return Ok(output);
    }
    if response_id != Some(0) && response_id != Some(1) {
        return Err(Failure::invalid("Unexpected daemon response id"));
    }
    let error = text_field(object, "error")?;
    let code = text_field(object, "error_code")?;
    let hint = text_field(object, "hint")?;
    Err(Failure {
        code,
        output,
        detail: error,
        hint,
    })
}

fn text_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<String, Failure> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(Failure::invalid(format!(
            "Invalid daemon response field: {name}"
        ))),
    }
}

#[cfg(unix)]
fn write_failure(error: io::Error, read_timeout: f64) -> Failure {
    if error.kind() == io::ErrorKind::TimedOut || error.kind() == io::ErrorKind::WouldBlock {
        Failure::timeout(read_timeout)
    } else {
        Failure::connection_lost(format!("Daemon connection lost: {error}"))
    }
}

enum TimeoutSpec {
    Delegate,
    Value(f64),
}

fn parse_timeout(name: &str, default: f64) -> TimeoutSpec {
    let raw = match env::var_os(name) {
        Some(value) => match value.into_string() {
            Ok(value) => value,
            Err(_) => return TimeoutSpec::Delegate,
        },
        None => return TimeoutSpec::Value(default),
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return TimeoutSpec::Value(default);
    }
    if trimmed.contains('_') {
        return TimeoutSpec::Delegate;
    }
    let value = match trimmed.parse::<f64>() {
        Ok(value) => value,
        Err(_) => return TimeoutSpec::Value(default),
    };
    if value.is_nan() {
        return TimeoutSpec::Value(MIN_TIMEOUT);
    }
    if value <= 0.0 {
        return TimeoutSpec::Value(default);
    }
    if !value.is_finite() || value > 1_000_000_000.0 {
        return TimeoutSpec::Delegate;
    }
    if value < MIN_TIMEOUT {
        return TimeoutSpec::Value(MIN_TIMEOUT);
    }
    TimeoutSpec::Value(value)
}

fn socket_path() -> Result<PathBuf, ()> {
    let raw = match env::var_os("OPENBROWSER_SOCKET") {
        Some(value) => value,
        None => return Ok(home_dir()?.join(".openbrowser").join("daemon.sock")),
    };
    let text = raw.to_str().ok_or(())?;
    if text == "~" {
        return home_dir();
    }
    if let Some(rest) = text.strip_prefix("~/") {
        return Ok(home_dir()?.join(rest));
    }
    if text.starts_with('~') {
        return Err(());
    }
    Ok(PathBuf::from(text))
}

fn home_dir() -> Result<PathBuf, ()> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or(())
}

fn format_timeout(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::{write_request_loop, WriteBudget};
    use std::io;
    use std::thread;
    use std::time::{Duration, Instant};

    struct FakeWriter {
        max_chunk: usize,
        pause: Duration,
        writes: usize,
    }

    impl WriteBudget for FakeWriter {
        fn budget_write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            thread::sleep(self.pause);
            self.writes += 1;
            Ok(self.max_chunk.min(buffer.len()))
        }

        fn budget_timeout(&mut self, _timeout: Duration) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn whole_request_budget_times_out() {
        let mut writer = FakeWriter {
            max_chunk: 1,
            pause: Duration::from_millis(20),
            writes: 0,
        };
        let request = vec![b'x'; 10_000];
        let started = Instant::now();
        let failure =
            write_request_loop(&mut writer, &request, Duration::from_millis(40), 0.04).unwrap_err();
        assert_eq!(failure.code, "daemon_timeout");
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(writer.writes >= 1);
    }

    #[test]
    fn zero_write_is_terminal() {
        let mut writer = FakeWriter {
            max_chunk: 0,
            pause: Duration::ZERO,
            writes: 0,
        };
        let failure =
            write_request_loop(&mut writer, b"request", Duration::from_secs(1), 1.0).unwrap_err();
        assert_eq!(failure.code, "daemon_connection_lost");
        assert_eq!(writer.writes, 1);
    }

    #[test]
    fn complete_write_returns_ok() {
        let mut writer = FakeWriter {
            max_chunk: 1_000,
            pause: Duration::ZERO,
            writes: 0,
        };
        write_request_loop(&mut writer, b"request", Duration::from_secs(1), 1.0).unwrap();
        assert_eq!(writer.writes, 1);
    }
}
