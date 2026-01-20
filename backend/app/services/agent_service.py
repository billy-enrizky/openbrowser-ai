"""Agent service for running OpenBrowser agents."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from app.core.config import settings
from app.services.vnc_service import vnc_service, VncSession, VNC_AVAILABLE

logger = logging.getLogger(__name__)


class AgentSession:
    """Manages a single agent session."""

    def __init__(
        self,
        task_id: str,
        task: str,
        agent_type: str = "code",
        max_steps: int = 50,
        use_vision: bool = True,
        llm_model: str | None = None,
        enable_vnc: bool = True,
        on_step_callback: Callable[[dict[str, Any]], None] | None = None,
        on_output_callback: Callable[[str, bool], None] | None = None,
        on_screenshot_callback: Callable[[str | None, int], None] | None = None,
        on_thinking_callback: Callable[[str], None] | None = None,
        on_error_callback: Callable[[str], None] | None = None,
        on_log_callback: Callable[[str, str, str | None, int | None], None] | None = None,
        on_vnc_info_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.task_id = task_id
        self.task = task
        self.agent_type = agent_type
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.llm_model = llm_model
        self.enable_vnc = enable_vnc and settings.VNC_ENABLED and VNC_AVAILABLE
        
        # Callbacks for real-time updates
        self.on_step_callback = on_step_callback
        self.on_output_callback = on_output_callback
        self.on_screenshot_callback = on_screenshot_callback
        self.on_thinking_callback = on_thinking_callback
        self.on_error_callback = on_error_callback
        self.on_log_callback = on_log_callback
        self.on_vnc_info_callback = on_vnc_info_callback
        
        # State
        self.agent: Any = None
        self.browser_session: Any = None
        self.vnc_session: VncSession | None = None
        self.is_running = False
        self.is_paused = False
        self.is_cancelled = False
        self.current_step = 0
        self.result: str | None = None
        self.success: bool | None = None
        self.error: str | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
    
    def _log(self, level: str, message: str, source: str | None = None, step_number: int | None = None):
        """Send log message to callback if available."""
        if self.on_log_callback:
            self.on_log_callback(level, message, source, step_number)

    async def _setup_vnc(self) -> VncSession | None:
        """Set up VNC session for browser viewing."""
        if not self.enable_vnc:
            return None
        
        try:
            self._log("info", "Setting up VNC session for browser viewing...", "vnc")
            vnc_session = await vnc_service.create_session(
                task_id=self.task_id,
                width=settings.VNC_WIDTH,
                height=settings.VNC_HEIGHT,
                password=settings.VNC_PASSWORD,
            )
            
            # Send VNC info to frontend
            if self.on_vnc_info_callback:
                self.on_vnc_info_callback(vnc_session.to_dict())
            
            self._log("info", f"VNC session ready at {vnc_session.websocket_url}", "vnc")
            return vnc_session
        except Exception as e:
            self._log("warning", f"Failed to set up VNC session: {e}. Continuing without VNC.", "vnc")
            logger.warning(f"VNC setup failed for task {self.task_id}: {e}")
            return None

    async def start(self) -> dict[str, Any]:
        """Start the agent and run the task."""
        if self.is_running:
            raise RuntimeError("Agent is already running")
        
        self.is_running = True
        self.started_at = datetime.utcnow()
        self._log("info", f"Starting task: {self.task[:100]}...", "agent")
        
        try:
            # Set up VNC session first (if enabled)
            self.vnc_session = await self._setup_vnc()
            
            # Import openbrowser components
            from openbrowser import Agent, CodeAgent, BrowserSession, BrowserProfile
            
            self._log("info", "Initializing browser session...", "agent")
            
            # Create browser session with profile
            # When VNC is enabled, we need headless=False and set DISPLAY
            browser_profile = BrowserProfile(
                headless=False,  # Always headful when VNC is enabled
                keep_alive=False,
            )
            
            # Set up environment for VNC display
            browser_env = None
            if self.vnc_session:
                browser_env = {"DISPLAY": self.vnc_session.display}
                self._log("info", f"Browser will use display {self.vnc_session.display}", "vnc")
            
            self.browser_session = BrowserSession(browser_profile=browser_profile)
            
            # IMPORTANT: Start the browser session BEFORE passing to agent
            logger.info("Starting browser session...")
            self._log("info", "Starting browser session...", "browser")
            
            # If VNC is enabled, set the DISPLAY environment variable before starting
            if browser_env:
                original_display = os.environ.get("DISPLAY")
                os.environ["DISPLAY"] = browser_env["DISPLAY"]
                try:
                    await self.browser_session.start()
                finally:
                    # Restore original DISPLAY
                    if original_display:
                        os.environ["DISPLAY"] = original_display
                    elif "DISPLAY" in os.environ:
                        del os.environ["DISPLAY"]
            else:
                await self.browser_session.start()
            
            logger.info("Browser session started successfully")
            self._log("info", "Browser session started successfully", "browser")
            
            # Initialize LLM - use specified model or default to gemini-2.5-flash
            model_to_use = self.llm_model or settings.DEFAULT_LLM_MODEL
            logger.info(f"Using LLM model: {model_to_use}")
            self._log("info", f"Using LLM model: {model_to_use}", "llm")
            llm = self._create_llm(model_to_use)
            
            # Create appropriate agent
            if self.agent_type == "code":
                self._log("info", "Creating CodeAgent...", "agent")
                # Import CodeAgentTools to configure display_files_in_done_text
                from openbrowser.tools.service import CodeAgentTools
                
                # Create tools with display_files_in_done_text=False
                # This prevents raw file content from being included in the message text
                # since we send file attachments separately
                tools = CodeAgentTools(display_files_in_done_text=False)
                
                self.agent = CodeAgent(
                    task=self.task,
                    llm=llm,
                    browser_session=self.browser_session,  # Use browser_session parameter
                    max_steps=self.max_steps,
                    use_vision=self.use_vision,
                    tools=tools,
                )
            else:
                self._log("info", "Creating Browser Agent...", "agent")
                # Create step callback for browser agent
                async def step_callback(browser_state, agent_output, step_number):
                    await self._handle_browser_agent_step(browser_state, agent_output, step_number)
                
                self.agent = Agent(
                    task=self.task,
                    llm=llm,
                    browser_session=self.browser_session,  # Use browser_session parameter
                    use_vision=self.use_vision,
                    register_new_step_callback=step_callback,
                )
            
            # Run the agent
            logger.info(f"Starting {self.agent_type} agent for task: {self.task[:50]}...")
            self._log("info", f"Starting {self.agent_type} agent execution...", "agent")
            
            if self.agent_type == "code":
                result = await self._run_code_agent()
            else:
                result = await self._run_browser_agent()
            
            self.completed_at = datetime.utcnow()
            self._log("info", f"Task completed successfully", "agent")
            return result
            
        except asyncio.CancelledError:
            self.is_cancelled = True
            self.error = "Task was cancelled"
            self._log("warning", "Task was cancelled", "agent")
            raise
        except Exception as e:
            logger.exception(f"Agent error: {e}")
            self.error = str(e)
            self._log("error", f"Agent error: {str(e)}", "agent")
            if self.on_error_callback:
                self.on_error_callback(str(e))
            raise
        finally:
            self.is_running = False
            await self._cleanup()

    async def _run_code_agent(self) -> dict[str, Any]:
        """Run CodeAgent with step tracking."""
        # CodeAgent doesn't have per-step callbacks like Agent,
        # so we run it and extract results after
        session = await self.agent.run(max_steps=self.max_steps)
        
        # Extract results from CodeAgent
        is_done = self.agent._is_task_done()
        task_result = self.agent.namespace.get('_task_result', '')
        task_success = self.agent.namespace.get('_task_success', False)
        
        self.result = task_result if task_result else "Task completed"
        self.success = task_success if is_done else False
        
        # Collect file attachments from CodeAgent
        attachments = self._collect_code_agent_attachments()
        
        # Send final output
        if self.on_output_callback:
            self.on_output_callback(self.result, True)
        
        duration = (self.completed_at - self.started_at).total_seconds() if self.completed_at and self.started_at else 0
        
        return {
            "task_id": self.task_id,
            "result": self.result,
            "success": self.success,
            "total_steps": len(self.agent.complete_history),
            "duration_seconds": duration,
            "attachments": attachments,
        }

    def _collect_code_agent_attachments(self) -> list[dict[str, Any]]:
        """Collect file attachments from CodeAgent namespace."""
        import os
        from pathlib import Path
        attachments = []
        
        self._log("info", "Collecting file attachments...", "agent")
        
        # Check for files in namespace
        if hasattr(self.agent, 'namespace'):
            namespace = self.agent.namespace
            
            # Get the FileSystem's data directory if available
            file_system_dir = None
            if hasattr(self.agent, 'file_system') and self.agent.file_system:
                file_system_dir = Path(self.agent.file_system.get_dir())
                logger.info(f"FileSystem directory: {file_system_dir}")
                self._log("info", f"FileSystem directory: {file_system_dir}", "agent")
            
            # Get current working directory
            cwd = Path.cwd()
            logger.info(f"Current working directory: {cwd}")
            self._log("info", f"Current working directory: {cwd}", "agent")
            
            # First, check for _task_attachments (set by done() function)
            # These are file paths that were explicitly requested to be displayed
            if '_task_attachments' in namespace and isinstance(namespace['_task_attachments'], list):
                self._log("info", f"Found _task_attachments: {namespace['_task_attachments']}", "agent")
                for file_path in namespace['_task_attachments']:
                    if isinstance(file_path, str):
                        # Try multiple locations for the file
                        paths_to_try = [
                            Path(file_path),  # As-is (might be absolute)
                        ]
                        
                        # If it's a relative path, try various directories
                        if not os.path.isabs(file_path):
                            basename = os.path.basename(file_path)
                            # Try current working directory
                            paths_to_try.append(cwd / file_path)
                            paths_to_try.append(cwd / basename)
                            # Try FileSystem directory
                            if file_system_dir:
                                paths_to_try.append(file_system_dir / file_path)
                                paths_to_try.append(file_system_dir / basename)
                            # Try parent directory (in case backend is in subdirectory)
                            paths_to_try.append(cwd.parent / file_path)
                            paths_to_try.append(cwd.parent / basename)
                            # Try openbrowser_agent_data directory
                            paths_to_try.append(cwd / 'openbrowser_agent_data' / basename)
                            paths_to_try.append(cwd.parent / 'openbrowser_agent_data' / basename)
                            # Try browseruse_agent_data directory (legacy)
                            paths_to_try.append(cwd / 'browseruse_agent_data' / basename)
                            paths_to_try.append(cwd.parent / 'browseruse_agent_data' / basename)
                        
                        # Try each path until we find the file
                        found = False
                        for try_path in paths_to_try:
                            try_path = Path(try_path).resolve()
                            if try_path.exists() and try_path.is_file():
                                try:
                                    content = try_path.read_text(encoding='utf-8')
                                    
                                    file_name = try_path.name
                                    file_type = self._get_file_type(file_name)
                                    file_size = try_path.stat().st_size
                                    
                                    attachments.append({
                                        "name": file_name,
                                        "content": content,
                                        "type": file_type,
                                        "size": file_size,
                                        "url": f"file://{try_path}",
                                    })
                                    logger.info(f"Collected attachment: {file_name} ({file_size} bytes) from {try_path}")
                                    self._log("info", f"Collected attachment: {file_name} ({file_size} bytes)", "agent")
                                    found = True
                                    break
                                except Exception as e:
                                    logger.warning(f"Failed to read file {try_path}: {e}")
                                    self._log("warning", f"Failed to read file {try_path}: {e}", "agent")
                        
                        if not found:
                            logger.warning(f"Attachment file not found in any location: {file_path}")
                            self._log("warning", f"Attachment file not found: {file_path}", "agent")
                            # Log all tried paths for debugging
                            for p in paths_to_try:
                                logger.debug(f"  Tried: {p} (exists: {Path(p).exists()})")
            else:
                self._log("info", "No _task_attachments found in namespace", "agent")
            
            # Also check for common file-related variables as fallback
            file_vars = ['output_file', 'result_file', 'data_file', 'csv_file', 'json_file', 'saved_file']
            for var_name in file_vars:
                if var_name in namespace:
                    file_content = namespace[var_name]
                    if isinstance(file_content, str) and len(file_content) > 0:
                        # Only add if it looks like actual file content (not just a filename)
                        # Skip if content is just a short filename-like string
                        if len(file_content) > 50 or '\n' in file_content or ',' in file_content:
                            # Determine file type from variable name
                            file_type = 'text'
                            file_name = f'{var_name}.txt'
                            if 'csv' in var_name:
                                file_type = 'csv'
                                file_name = f'{var_name}.csv'
                            elif 'json' in var_name:
                                file_type = 'json'
                                file_name = f'{var_name}.json'
                            
                            attachments.append({
                                "name": file_name,
                                "content": file_content,
                                "type": file_type,
                                "size": len(file_content.encode('utf-8')),
                            })
                            self._log("info", f"Collected fallback attachment from variable: {var_name}", "agent")
            
            # Check for _files list (explicit file outputs)
            if '_files' in namespace and isinstance(namespace['_files'], list):
                for file_info in namespace['_files']:
                    if isinstance(file_info, dict):
                        attachments.append({
                            "name": file_info.get('name', 'file'),
                            "content": file_info.get('content'),
                            "type": file_info.get('type', 'text'),
                            "mime_type": file_info.get('mime_type'),
                            "size": file_info.get('size'),
                            "url": file_info.get('url'),
                        })
                        self._log("info", f"Collected explicit file: {file_info.get('name', 'file')}", "agent")
        
        self._log("info", f"Total attachments collected: {len(attachments)}", "agent")
        return attachments

    async def _run_browser_agent(self) -> dict[str, Any]:
        """Run Browser Agent with step tracking."""
        history = await self.agent.run(max_steps=self.max_steps)
        
        # Extract results
        self.result = history.final_result() or "Task completed"
        self.success = history.is_successful()
        
        # Collect file attachments from Browser Agent
        attachments = self._collect_browser_agent_attachments(history)
        
        # Send final output
        if self.on_output_callback:
            self.on_output_callback(self.result, True)
        
        duration = history.total_duration_seconds() if history else 0
        
        return {
            "task_id": self.task_id,
            "result": self.result,
            "success": self.success,
            "total_steps": len(history.history) if history else 0,
            "duration_seconds": duration,
            "attachments": attachments,
        }

    def _collect_browser_agent_attachments(self, history) -> list[dict[str, Any]]:
        """Collect file attachments from Browser Agent history."""
        attachments = []
        
        if history and hasattr(history, 'history'):
            for item in history.history:
                # Check for attachments in action results
                if hasattr(item, 'result') and item.result:
                    for result in item.result if isinstance(item.result, list) else [item.result]:
                        if hasattr(result, 'attachments') and result.attachments:
                            for att in result.attachments:
                                if isinstance(att, str):
                                    # Legacy format: just a filename
                                    attachments.append({
                                        "name": att.split('/')[-1],
                                        "url": att,
                                        "type": self._get_file_type(att),
                                    })
                                elif isinstance(att, dict):
                                    attachments.append(att)
        
        return attachments

    def _get_file_type(self, filename: str) -> str:
        """Get file type from filename."""
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        type_map = {
            'csv': 'csv',
            'json': 'json',
            'txt': 'text',
            'md': 'text',
            'log': 'text',
            'png': 'image',
            'jpg': 'image',
            'jpeg': 'image',
            'gif': 'image',
            'webp': 'image',
            'svg': 'image',
            'pdf': 'pdf',
            'py': 'code',
            'js': 'code',
            'ts': 'code',
            'html': 'code',
            'css': 'code',
        }
        return type_map.get(ext, 'unknown')

    async def _handle_browser_agent_step(self, browser_state, agent_output, step_number):
        """Handle a step update from Browser Agent."""
        self.current_step = step_number
        
        if self.on_step_callback:
            step_data = {
                "step_number": step_number,
                "total_steps": self.max_steps,
            }
            
            if agent_output:
                current_state = agent_output.current_state
                if current_state:
                    step_data["thinking"] = current_state.thinking
                    step_data["memory"] = current_state.memory
                    step_data["next_goal"] = current_state.next_goal
                    step_data["evaluation"] = current_state.evaluation_previous_goal
                
                # Get action info
                if agent_output.action:
                    actions = []
                    for action in agent_output.action:
                        action_data = action.model_dump(exclude_unset=True)
                        if action_data:
                            action_name = next(iter(action_data.keys()))
                            actions.append(action_name)
                    step_data["action"] = ", ".join(actions)
            
            self.on_step_callback(step_data)
        
        # Send screenshot if available
        if self.on_screenshot_callback and browser_state and browser_state.screenshot:
            self.on_screenshot_callback(browser_state.screenshot, step_number)

    def _create_llm(self, model_name: str):
        """Create an LLM instance based on model name."""
        model_lower = model_name.lower()
        
        if "gpt" in model_lower or "openai" in model_lower:
            from openbrowser import ChatOpenAI
            return ChatOpenAI(model=model_name)
        elif "claude" in model_lower or "anthropic" in model_lower:
            from openbrowser import ChatAnthropic
            return ChatAnthropic(model=model_name)
        elif "gemini" in model_lower or "google" in model_lower:
            from openbrowser import ChatGoogle
            return ChatGoogle(model=model_name)
        elif "groq" in model_lower:
            from openbrowser import ChatGroq
            return ChatGroq(model=model_name)
        else:
            # Default to ChatBrowserUse
            from openbrowser import ChatBrowserUse
            return ChatBrowserUse()

    async def pause(self):
        """Pause the agent."""
        if self.agent and hasattr(self.agent, 'pause'):
            self.agent.pause()
            self.is_paused = True

    async def resume(self):
        """Resume the agent."""
        if self.agent and hasattr(self.agent, 'resume'):
            self.agent.resume()
            self.is_paused = False

    async def cancel(self):
        """Cancel the agent."""
        if self.agent and hasattr(self.agent, 'stop'):
            self.agent.stop()
        self.is_cancelled = True

    async def _cleanup(self):
        """Clean up resources."""
        try:
            if self.agent and hasattr(self.agent, 'close'):
                await self.agent.close()
            if self.browser_session:
                await self.browser_session.kill()
            # Clean up VNC session
            if self.vnc_session:
                await vnc_service.destroy_session(self.task_id)
                self.vnc_session = None
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class AgentManager:
    """Manages multiple agent sessions."""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        task: str,
        agent_type: str = "code",
        max_steps: int = 50,
        use_vision: bool = True,
        llm_model: str | None = None,
        enable_vnc: bool = True,
        **callbacks,
    ) -> AgentSession:
        """Create a new agent session."""
        task_id = str(uuid4())
        return await self.create_session_with_id(
            task_id=task_id,
            task=task,
            agent_type=agent_type,
            max_steps=max_steps,
            use_vision=use_vision,
            llm_model=llm_model,
            enable_vnc=enable_vnc,
            **callbacks,
        )

    async def create_session_with_id(
        self,
        task_id: str,
        task: str,
        agent_type: str = "code",
        max_steps: int = 50,
        use_vision: bool = True,
        llm_model: str | None = None,
        enable_vnc: bool = True,
        on_step_callback: Callable[[dict[str, Any]], None] | None = None,
        on_output_callback: Callable[[str, bool], None] | None = None,
        on_screenshot_callback: Callable[[str | None, int], None] | None = None,
        on_thinking_callback: Callable[[str], None] | None = None,
        on_error_callback: Callable[[str], None] | None = None,
        on_log_callback: Callable[[str, str, str | None, int | None], None] | None = None,
        on_vnc_info_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentSession:
        """Create a new agent session with a specific task ID."""
        async with self._lock:
            if len(self.sessions) >= self.max_concurrent:
                raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent}) reached")
            
            session = AgentSession(
                task_id=task_id,
                task=task,
                agent_type=agent_type,
                max_steps=max_steps,
                use_vision=use_vision,
                llm_model=llm_model,
                enable_vnc=enable_vnc,
                on_step_callback=on_step_callback,
                on_output_callback=on_output_callback,
                on_screenshot_callback=on_screenshot_callback,
                on_thinking_callback=on_thinking_callback,
                on_error_callback=on_error_callback,
                on_log_callback=on_log_callback,
                on_vnc_info_callback=on_vnc_info_callback,
            )
            self.sessions[task_id] = session
            return session

    async def get_session(self, task_id: str) -> AgentSession | None:
        """Get a session by task ID."""
        return self.sessions.get(task_id)

    async def remove_session(self, task_id: str):
        """Remove a session."""
        async with self._lock:
            if task_id in self.sessions:
                session = self.sessions.pop(task_id)
                if session.is_running:
                    await session.cancel()

    async def cleanup_completed(self):
        """Remove completed sessions."""
        async with self._lock:
            to_remove = [
                task_id for task_id, session in self.sessions.items()
                if not session.is_running and session.completed_at
            ]
            for task_id in to_remove:
                del self.sessions[task_id]


# Global agent manager instance
agent_manager = AgentManager()

