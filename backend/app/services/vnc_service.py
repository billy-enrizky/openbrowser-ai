"""VNC service for managing virtual display and VNC server for browser viewing."""

import asyncio
import logging
import os
import secrets
import shutil
import socket
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def check_vnc_dependencies() -> tuple[bool, list[str]]:
    """Check if VNC dependencies are available.
    
    Returns:
        Tuple of (all_available, list_of_missing_tools)
    """
    required_tools = ["Xvfb", "x11vnc", "websockify"]
    missing = []
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            missing.append(tool)
    
    return len(missing) == 0, missing


# Check dependencies at module load time
VNC_AVAILABLE, VNC_MISSING_TOOLS = check_vnc_dependencies()
if not VNC_AVAILABLE:
    logger.warning(
        f"VNC dependencies not available: {', '.join(VNC_MISSING_TOOLS)}. "
        "VNC browser viewing will be disabled. "
        "Install these tools or use Docker deployment for VNC support."
    )


@dataclass
class VncSession:
    """Represents an active VNC session."""
    
    task_id: str
    display_number: int
    vnc_port: int
    websockify_port: int
    password: str
    width: int = 1920
    height: int = 1080
    
    # Process handles
    xvfb_process: asyncio.subprocess.Process | None = None
    x11vnc_process: asyncio.subprocess.Process | None = None
    websockify_process: asyncio.subprocess.Process | None = None
    fluxbox_process: asyncio.subprocess.Process | None = None
    
    @property
    def display(self) -> str:
        """Get the X11 display string."""
        return f":{self.display_number}"
    
    @property
    def websocket_url(self) -> str:
        """Get the WebSocket URL for noVNC connection."""
        return f"ws://localhost:{self.websockify_port}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket transmission."""
        return {
            "vnc_url": self.websocket_url,
            "password": self.password,
            "width": self.width,
            "height": self.height,
            "display": self.display,
        }


class VncService:
    """Service for managing VNC sessions with Xvfb, x11vnc, and websockify."""
    
    def __init__(
        self,
        base_display: int = 99,
        base_vnc_port: int = 5900,
        base_websockify_port: int = 6080,
        default_width: int = 1920,
        default_height: int = 1080,
    ):
        self.base_display = base_display
        self.base_vnc_port = base_vnc_port
        self.base_websockify_port = base_websockify_port
        self.default_width = default_width
        self.default_height = default_height
        
        self._sessions: dict[str, VncSession] = {}
        self._lock = asyncio.Lock()
        self._next_display_offset = 0
    
    def _find_free_port(self, start_port: int) -> int:
        """Find a free port starting from the given port."""
        port = start_port
        while port < start_port + 100:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"Could not find free port starting from {start_port}")
    
    def _generate_password(self, length: int = 12) -> str:
        """Generate a random password for VNC authentication."""
        return secrets.token_urlsafe(length)[:length]
    
    async def create_session(
        self,
        task_id: str,
        width: int | None = None,
        height: int | None = None,
        password: str | None = None,
    ) -> VncSession:
        """Create a new VNC session for a task.
        
        Args:
            task_id: Unique identifier for the task
            width: Display width (default: 1280)
            height: Display height (default: 1024)
            password: VNC password (auto-generated if not provided)
            
        Returns:
            VncSession with connection details
            
        Raises:
            RuntimeError: If VNC dependencies are not available
        """
        # Check if VNC is available
        if not VNC_AVAILABLE:
            raise RuntimeError(
                f"VNC dependencies not available: {', '.join(VNC_MISSING_TOOLS)}. "
                "Install Xvfb, x11vnc, and websockify, or use Docker deployment."
            )
        
        async with self._lock:
            if task_id in self._sessions:
                logger.warning(f"VNC session already exists for task {task_id}")
                return self._sessions[task_id]
            
            # Allocate resources
            display_number = self.base_display + self._next_display_offset
            vnc_port = self._find_free_port(self.base_vnc_port + self._next_display_offset)
            websockify_port = self._find_free_port(self.base_websockify_port + self._next_display_offset)
            self._next_display_offset += 1
            
            session = VncSession(
                task_id=task_id,
                display_number=display_number,
                vnc_port=vnc_port,
                websockify_port=websockify_port,
                password=password or self._generate_password(),
                width=width or self.default_width,
                height=height or self.default_height,
            )
            
            try:
                await self._start_session(session)
                self._sessions[task_id] = session
                logger.info(
                    f"VNC session created for task {task_id}: "
                    f"display={session.display}, websocket_port={session.websockify_port}"
                )
                return session
            except Exception as e:
                logger.error(f"Failed to create VNC session for task {task_id}: {e}")
                await self._cleanup_session(session)
                raise
    
    async def _start_session(self, session: VncSession) -> None:
        """Start all VNC-related processes for a session."""
        # 1. Start Xvfb (X Virtual Framebuffer)
        xvfb_cmd = [
            "Xvfb",
            session.display,
            "-screen", "0", f"{session.width}x{session.height}x24",
            "-ac",  # Disable access control
            "-nolisten", "tcp",
        ]
        
        logger.debug(f"Starting Xvfb: {' '.join(xvfb_cmd)}")
        session.xvfb_process = await asyncio.create_subprocess_exec(
            *xvfb_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Wait for Xvfb to be ready
        await asyncio.sleep(0.5)
        if session.xvfb_process.returncode is not None:
            stderr = await session.xvfb_process.stderr.read()
            raise RuntimeError(f"Xvfb failed to start: {stderr.decode()}")
        
        # Set DISPLAY environment for subsequent processes
        env = os.environ.copy()
        env["DISPLAY"] = session.display
        
        # 2. Start fluxbox (minimal window manager) - optional but helps with window management
        try:
            fluxbox_cmd = ["fluxbox"]
            logger.debug(f"Starting fluxbox on {session.display}")
            session.fluxbox_process = await asyncio.create_subprocess_exec(
                *fluxbox_cmd,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(0.3)
        except FileNotFoundError:
            logger.warning("fluxbox not found, continuing without window manager")
        
        # 3. Start x11vnc
        x11vnc_cmd = [
            "x11vnc",
            "-display", session.display,
            "-rfbport", str(session.vnc_port),
            "-passwd", session.password,
            "-forever",  # Don't exit after first client disconnects
            "-shared",   # Allow multiple clients
            "-noxdamage",  # Disable XDAMAGE for better compatibility
            "-noxfixes",
            "-noxrecord",
            "-nowf",     # No wireframe
            "-cursor", "arrow",
            "-nopw",     # Don't prompt for password file
            "-q",        # Quiet mode
        ]
        
        logger.debug(f"Starting x11vnc: {' '.join(x11vnc_cmd)}")
        session.x11vnc_process = await asyncio.create_subprocess_exec(
            *x11vnc_cmd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Wait for x11vnc to be ready
        await asyncio.sleep(0.5)
        if session.x11vnc_process.returncode is not None:
            stderr = await session.x11vnc_process.stderr.read()
            raise RuntimeError(f"x11vnc failed to start: {stderr.decode()}")
        
        # 4. Start websockify to bridge WebSocket to VNC
        websockify_cmd = [
            "websockify",
            "--web", "/usr/share/novnc",  # Serve noVNC web files (optional)
            str(session.websockify_port),
            f"localhost:{session.vnc_port}",
        ]
        
        logger.debug(f"Starting websockify: {' '.join(websockify_cmd)}")
        session.websockify_process = await asyncio.create_subprocess_exec(
            *websockify_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Wait for websockify to be ready
        await asyncio.sleep(0.3)
        if session.websockify_process.returncode is not None:
            stderr = await session.websockify_process.stderr.read()
            raise RuntimeError(f"websockify failed to start: {stderr.decode()}")
        
        logger.info(f"VNC session started: display={session.display}, vnc_port={session.vnc_port}, ws_port={session.websockify_port}")
    
    async def get_session(self, task_id: str) -> VncSession | None:
        """Get an existing VNC session by task ID."""
        return self._sessions.get(task_id)
    
    async def destroy_session(self, task_id: str) -> None:
        """Destroy a VNC session and clean up resources."""
        async with self._lock:
            session = self._sessions.pop(task_id, None)
            if session:
                await self._cleanup_session(session)
                logger.info(f"VNC session destroyed for task {task_id}")
    
    async def _cleanup_session(self, session: VncSession) -> None:
        """Clean up all processes for a session."""
        processes = [
            ("websockify", session.websockify_process),
            ("x11vnc", session.x11vnc_process),
            ("fluxbox", session.fluxbox_process),
            ("Xvfb", session.xvfb_process),
        ]
        
        for name, process in processes:
            if process and process.returncode is None:
                try:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    logger.debug(f"Terminated {name} process")
                except Exception as e:
                    logger.warning(f"Error terminating {name}: {e}")
    
    async def cleanup_all(self) -> None:
        """Clean up all VNC sessions."""
        async with self._lock:
            for task_id in list(self._sessions.keys()):
                session = self._sessions.pop(task_id)
                await self._cleanup_session(session)
            logger.info("All VNC sessions cleaned up")
    
    def get_display_env(self, task_id: str) -> dict[str, str] | None:
        """Get environment variables for a task's display.
        
        Returns dict with DISPLAY variable, or None if no session exists.
        """
        session = self._sessions.get(task_id)
        if session:
            return {"DISPLAY": session.display}
        return None


# Global VNC service instance
vnc_service = VncService()
