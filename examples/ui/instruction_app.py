"""
OpenBrowser Instruction App - Chainlit application for executing browser automation instructions.

This app reads instructions from a CSV file and allows users to:
1. Select from predefined instructions
2. Fill in specification values (manually or via CSV upload)
3. Execute browser automation with visual feedback

Usage:
    uv run chainlit run examples/ui/instruction_app.py
"""

import asyncio
import csv
import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import chainlit as cl
from chainlit.config import config
from dotenv import load_dotenv

from openbrowser import Agent, Browser, ChatGoogle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Path to instructions CSV (can be customized)
INSTRUCTIONS_CSV = os.getenv("INSTRUCTIONS_CSV", "openbrowser_instructions.csv")

# Set custom branding for OpenBrowser app
config.ui.name = "OpenBrowser"


def get_instructions_from_csv() -> List[Dict[str, str]]:
    """
    Read instructions from CSV file.
    
    Returns:
        List of dictionaries with instruction_name and general_instruction
    """
    instructions = []
    
    if not os.path.isfile(INSTRUCTIONS_CSV):
        logger.warning(f"Instructions CSV file not found: {INSTRUCTIONS_CSV}")
        return instructions
    
    try:
        with open(INSTRUCTIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instruction_name = row.get("instruction_name", "").strip()
                general_instruction = row.get("general_instruction", "").strip()
                if instruction_name and general_instruction:
                    instructions.append({
                        "instruction_name": instruction_name,
                        "general_instruction": general_instruction
                    })
        
        logger.info(f"Loaded {len(instructions)} instructions from CSV")
        return instructions
        
    except Exception as e:
        logger.error(f"Error reading instructions CSV: {e}")
        return instructions


def get_instruction_names() -> List[str]:
    """
    Get list of instruction names from CSV.
    
    Returns:
        List of instruction names
    """
    instructions = get_instructions_from_csv()
    return [inst["instruction_name"] for inst in instructions]


def get_instruction_by_name(name: str) -> Optional[str]:
    """
    Get general instruction by name.
    
    Args:
        name: Instruction name to search for
        
    Returns:
        General instruction string if found, None otherwise
    """
    instructions = get_instructions_from_csv()
    for inst in instructions:
        if inst["instruction_name"] == name:
            return inst["general_instruction"]
    return None


def extract_placeholders(instruction: str) -> List[str]:
    """
    Extract placeholder variables from instruction.
    
    Args:
        instruction: Instruction text with $PLACEHOLDER$ variables
        
    Returns:
        List of unique placeholder names
    """
    placeholders = re.findall(r'\$([A-Z_]+)\$', instruction)
    return list(set(placeholders))


def generate_specification_questions(placeholders: List[str]) -> str:
    """
    Generate user-friendly questions for placeholders.
    
    Args:
        placeholders: List of placeholder names
        
    Returns:
        Formatted questions string
    """
    if not placeholders:
        return "No specifications required for this instruction."
    
    questions = ["Please provide values for the following fields:\n"]
    for placeholder in sorted(placeholders):
        # Convert placeholder to readable format
        readable = placeholder.replace("_", " ").title()
        questions.append(f"- **{readable}**: (value for ${placeholder}$)")
    
    questions.append("\nYou can enter values in format: FIELD_NAME: value")
    questions.append("Or upload a CSV file with columns matching the placeholder names.")
    
    return "\n".join(questions)


def configure_instruction(general_instruction: str, specifications: str) -> str:
    """
    Replace placeholders in instruction with user specifications.
    
    Args:
        general_instruction: Instruction with $PLACEHOLDER$ variables
        specifications: User-provided specifications
        
    Returns:
        Configured instruction with placeholders replaced
    """
    configured = general_instruction
    
    # Parse specifications (format: FIELD_NAME: value)
    spec_lines = specifications.strip().split("\n")
    for line in spec_lines:
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().upper().replace(" ", "_")
                value = parts[1].strip()
                configured = configured.replace(f"${key}$", value)
    
    return configured


def parse_csv_specifications(file_content: bytes, filename: str) -> Dict[str, str]:
    """
    Parse CSV file content into specifications dictionary.
    
    Args:
        file_content: Raw bytes content of the file
        filename: Name of the file
        
    Returns:
        Dictionary with column names as keys and values
    """
    specs = {}
    try:
        if filename.endswith('.csv'):
            text_content = file_content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text_content))
            for row in reader:
                for key, value in row.items():
                    if value and str(value).strip():
                        specs[key.upper().replace(" ", "_")] = str(value).strip()
                break  # Only use first row
        else:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(file_content))
            if not df.empty:
                row = df.iloc[0]
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        specs[col.upper().replace(" ", "_")] = str(value).strip()
    except Exception as e:
        logger.error(f"Error parsing CSV/Excel file: {e}")
        raise
    return specs


async def execute_browser_automation(instruction: str) -> Dict[str, Any]:
    """
    Execute browser automation with the configured instruction.
    
    Args:
        instruction: The fully configured instruction to execute
        
    Returns:
        Dictionary with execution results
    """
    try:
        logger.info("=== Starting Browser Automation ===")
        logger.info(f"Instruction length: {len(instruction)} characters")
        
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
            return {
                "status": "error",
                "message": "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set",
                "execution_time": None
            }
        
        # Initialize Gemini LLM
        logger.info("Initializing LLM...")
        llm_gemini = ChatGoogle(
            model='gemini-3-flash-preview',
            temperature=0,
            api_key=GEMINI_API_KEY,
        )
        logger.info("LLM initialized successfully")
        
        # Create Browser with visible (non-headless) mode and full screen
        logger.info("Initializing browser in visible full-screen mode...")
        try:
            from openbrowser import BrowserProfile
            # Disable auto_split_screen and use start-maximized by not setting window_size
            browser_profile = BrowserProfile(
                headless=False,
                auto_split_screen=False,  # Disable split screen to allow full screen
            )
            browser = Browser(browser_profile=browser_profile)
            logger.info("Browser initialized with full-screen mode")
        except (TypeError, AttributeError, ImportError) as e:
            logger.warning(f"Could not create browser with BrowserProfile: {e}")
            try:
                browser = Browser(headless=False)
                logger.info("Browser initialized with headless=False")
            except (TypeError, AttributeError):
                browser = Browser()
                logger.warning("Could not set headless=False. Browser may run in headless mode.")
        
        # Add safety instructions
        safety_instructions = "\nNEVER CLOSE THE BROWSER or Click exit button WHEN THERE IS STILL POP-UP WINDOWS OPEN."
        safety_instructions += " If there are pop-up windows not specified in the instruction, make decisions based on the context."
        safety_instructions += " Only close when you encounter a success message or completion confirmation."
        full_instruction = f"{instruction}\n{safety_instructions}"
        
        # Create Agent
        logger.info("Creating Browser Agent...")
        agent = Agent(
            override_system_message="""You are a helpful assistant that can use the browser to perform instructions. 
            Follow the steps carefully and handle any errors gracefully.
            NEVER CLOSE THE BROWSER or Click exit button WHEN THERE IS STILL POP-UP WINDOWS OPEN.
            If there are pop-up windows not specified in the instruction, make decisions based on context.
            Only close when you encounter a success message or completion confirmation.
            """,
            page_extraction_llm=llm_gemini,
            max_failures=6,
            max_actions_per_step=50,
            task=full_instruction,
            llm=llm_gemini,
            browser=browser,
        )
        
        logger.info("Agent created successfully")
        logger.info("Starting browser automation execution...")
        
        # Execute the agent
        start_time = time.time()
        
        try:
            logger.info("Calling agent.run()...")
            result = await agent.run()
            execution_time = time.time() - start_time
            
            logger.info(f"Execution completed successfully in {execution_time:.2f} seconds")
            
            return {
                "status": "success",
                "message": f"Browser automation completed successfully in {execution_time:.2f} seconds",
                "instruction": full_instruction,
                "execution_time": execution_time,
                "result": result
            }
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Execution failed after {execution_time:.2f} seconds: {error_msg}")
            logger.exception("Full exception traceback:")
            
            return {
                "status": "error",
                "message": f"Browser automation failed: {error_msg}",
                "instruction": full_instruction,
                "execution_time": execution_time
            }
            
    except Exception as e:
        logger.error(f"Error setting up browser automation: {e}")
        return {
            "status": "error",
            "message": f"Error setting up browser automation: {str(e)}",
            "instruction": instruction,
            "execution_time": None
        }


async def request_csv_file(message: str):
    """
    Prompt the user to upload a CSV/Excel file.
    
    Args:
        message: Prompt message to display
        
    Returns:
        The uploaded file(s)
    """
    return await cl.AskFileMessage(
        content=message,
        accept=["application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "text/csv"],
        max_size_mb=10,
        timeout=30000
    ).send()


@cl.on_chat_start
async def on_chat_start():
    """Handle chat session start."""
    cl.user_session.set("selected_instruction", None)
    cl.user_session.set("general_instruction", None)
    cl.user_session.set("configured_instruction", None)
    cl.user_session.set("awaiting_specifications", False)
    cl.user_session.set("executing", False)
    
    await cl.Message(
        author='OpenBrowserBot',
        content="Welcome to the OpenBrowser Instruction Executor!"
    ).send()
    
    # Get available instructions
    instruction_names = get_instruction_names()
    
    if not instruction_names:
        await cl.Message(
            author='OpenBrowserBot',
            content=f"No instructions found in {INSTRUCTIONS_CSV}. Please add instructions to the CSV file."
        ).send()
        return
    
    # Create action buttons for each instruction
    actions = []
    for idx, name in enumerate(instruction_names):
        actions.append(
            cl.Action(
                name=f"select_instruction_{idx}",
                payload={"instruction_name": name},
                label=name
            )
        )
    
    await cl.Message(
        author='OpenBrowserBot',
        content=f"Found **{len(instruction_names)}** available instructions. Please select one:"
    ).send()
    
    res = await cl.AskActionMessage(
        author='OpenBrowserBot',
        content="Select an instruction to execute:",
        actions=actions,
        timeout=60000
    ).send()
    
    if res and res.get("payload", {}).get("instruction_name"):
        selected_name = res.get("payload", {}).get("instruction_name")
        
        await cl.Message(
            author='OpenBrowserBot',
            content=f"**Selected:** {selected_name}"
        ).send()
        
        # Load the instruction
        general_instruction = get_instruction_by_name(selected_name)
        
        if general_instruction:
            cl.user_session.set("selected_instruction", selected_name)
            cl.user_session.set("general_instruction", general_instruction)
            
            await cl.Message(
                author='OpenBrowserBot',
                content=f"**Instruction Loaded:**\n\n{general_instruction}"
            ).send()
            
            # Extract placeholders and ask for specifications
            placeholders = extract_placeholders(general_instruction)
            
            if placeholders:
                questions = generate_specification_questions(placeholders)
                
                # Ask how user wants to provide specifications
                spec_res = await cl.AskActionMessage(
                    author='OpenBrowserBot',
                    content="How would you like to provide specifications?",
                    actions=[
                        cl.Action(
                            name="enter_manually",
                            payload={"action": "manual"},
                            label="Enter Manually"
                        ),
                        cl.Action(
                            name="upload_csv",
                            payload={"action": "csv"},
                            label="Upload CSV File"
                        )
                    ],
                    timeout=30000
                ).send()
                
                if spec_res and spec_res.get("payload", {}).get("action") == "csv":
                    # User wants to upload CSV
                    await cl.Message(
                        author='OpenBrowserBot',
                        content="Please upload a CSV file with columns matching the placeholder names."
                    ).send()
                    
                    files = await request_csv_file("Upload your specifications CSV file:")
                    
                    if files:
                        file = files[0]
                        try:
                            with open(file.path, 'rb') as f:
                                file_content = f.read()
                            
                            specs = parse_csv_specifications(file_content, file.name)
                            
                            # Convert specs dict to string format
                            spec_str = "\n".join([f"{k}: {v}" for k, v in specs.items()])
                            
                            await cl.Message(
                                author='OpenBrowserBot',
                                content=f"**Loaded specifications from CSV:**\n```\n{spec_str}\n```"
                            ).send()
                            
                            # Configure instruction
                            configured = configure_instruction(general_instruction, spec_str)
                            cl.user_session.set("configured_instruction", configured)
                            
                            await cl.Message(
                                author='OpenBrowserBot',
                                content=f"**Configured Instruction:**\n\n{configured}"
                            ).send()
                            
                            # Ask to execute
                            await ask_to_execute()
                            
                        except Exception as e:
                            logger.error(f"Error processing CSV: {e}")
                            await cl.Message(
                                author='OpenBrowserBot',
                                content=f"Error processing CSV file: {str(e)}"
                            ).send()
                    else:
                        await cl.Message(
                            author='OpenBrowserBot',
                            content="No file uploaded. Please enter specifications manually."
                        ).send()
                        cl.user_session.set("awaiting_specifications", True)
                        await cl.Message(
                            author='OpenBrowserBot',
                            content=questions
                        ).send()
                else:
                    # User wants to enter manually
                    cl.user_session.set("awaiting_specifications", True)
                    await cl.Message(
                        author='OpenBrowserBot',
                        content=questions
                    ).send()
            else:
                # No placeholders, ready to execute
                cl.user_session.set("configured_instruction", general_instruction)
                await cl.Message(
                    author='OpenBrowserBot',
                    content="No specifications required for this instruction."
                ).send()
                await ask_to_execute()
        else:
            await cl.Message(
                author='OpenBrowserBot',
                content=f"Error: Could not load instruction '{selected_name}'"
            ).send()
    else:
        await cl.Message(
            author='OpenBrowserBot',
            content="No instruction selected. Please restart the chat to try again."
        ).send()


async def ask_to_execute():
    """Ask user if they want to execute the configured instruction."""
    exec_res = await cl.AskActionMessage(
        author='OpenBrowserBot',
        content="Ready to execute browser automation with this instruction?",
        actions=[
            cl.Action(
                name="execute_yes",
                payload={"action": "yes"},
                label="Yes, Execute"
            ),
            cl.Action(
                name="execute_no",
                payload={"action": "no"},
                label="No, Cancel"
            )
        ],
        timeout=60000
    ).send()
    
    if exec_res and exec_res.get("payload", {}).get("action") == "yes":
        cl.user_session.set("executing", True)
        
        configured_instruction = cl.user_session.get("configured_instruction")
        
        await cl.Message(
            author='OpenBrowserBot',
            content="Starting browser automation... Please wait."
        ).send()
        
        async with cl.Step(name="Executing Browser Automation", type="tool") as exec_step:
            exec_step.input = f"Instruction length: {len(configured_instruction)} characters"
            result = await execute_browser_automation(configured_instruction)
            exec_step.output = result.get("message", "Execution finished")
            exec_step.status = "success" if result.get("status") == "success" else "error"
        
        if result.get("status") == "success":
            await cl.Message(
                author='OpenBrowserBot',
                content=f"**Execution completed successfully!**\n\nExecution time: {result.get('execution_time', 0):.2f} seconds"
            ).send()
        else:
            await cl.Message(
                author='OpenBrowserBot',
                content=f"**Execution failed:**\n\n{result.get('message', 'Unknown error')}"
            ).send()
        
        cl.user_session.set("executing", False)
        
        await cl.Message(
            author='OpenBrowserBot',
            content="Workflow completed! Restart the chat to execute another instruction."
        ).send()
    else:
        await cl.Message(
            author='OpenBrowserBot',
            content="Execution cancelled. Restart the chat to try again."
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    user_input = message.content.strip()
    
    if cl.user_session.get("executing", False):
        await cl.Message(
            author='OpenBrowserBot',
            content="Please wait, execution is in progress..."
        ).send()
        return
    
    if cl.user_session.get("awaiting_specifications", False):
        # User is providing specifications
        cl.user_session.set("awaiting_specifications", False)
        
        general_instruction = cl.user_session.get("general_instruction")
        
        # Configure instruction with user specifications
        configured = configure_instruction(general_instruction, user_input)
        cl.user_session.set("configured_instruction", configured)
        
        await cl.Message(
            author='OpenBrowserBot',
            content=f"**Configured Instruction:**\n\n{configured}"
        ).send()
        
        # Ask to execute
        await ask_to_execute()
    else:
        await cl.Message(
            author='OpenBrowserBot',
            content="Please restart the chat to select and execute an instruction."
        ).send()


if __name__ == "__main__":
    # This is for testing purposes
    logger.info("OpenBrowser Instruction App loaded")
    logger.info(f"Instructions CSV: {INSTRUCTIONS_CSV}")
