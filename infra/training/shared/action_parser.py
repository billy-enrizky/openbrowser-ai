"""Parse Flow model text output into executable action dicts.

Converts decoded flow token text (e.g. "Step 1: Navigate to ...")
into action dicts that can be executed by BrowserEnvironment.execute_actions()
via the openbrowser Tools wrapper methods.

Each action dict has format: {"action": "action_name", "params": {...}}
where action_name maps to an openbrowser Tools method (navigate, click_element,
input_text, select_dropdown_option).

Requires an element_map (field_name -> element index) obtained from
the browser DOM after navigation to resolve field names to clickable
element indices.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Strip <think>...</think> blocks from Qwen3 reasoning-mode output
RE_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Regex patterns for FormFactory action format
# Flexible prefix: matches "Step N:", "Step N.", "N:", "N." (with/without Step prefix)
_P = r"(?:Step\s+)?\d+[.:]\s*"
RE_NAVIGATE = re.compile(_P + r"Navigate to (.+)")
RE_TYPE = re.compile(_P + r"Type ['\"](.+?)['\"] into the ['\"](.+?)['\"] field")
RE_SELECT = re.compile(_P + r"Select ['\"](.+?)['\"] from the ['\"](.+?)['\"] field")
RE_CLICK_CHECKBOX = re.compile(_P + r"Click on the ['\"](.+?)['\"] checkbox")
RE_CLICK_INPUT = re.compile(_P + r"Click on the ['\"](.+?)['\"] input field")
RE_SUBMIT = re.compile(_P + r"Click the Submit button")
# Alternate formats the model may produce after SFT
RE_FILL_IN = re.compile(_P + r"(?:Fill in|Enter|Input) ['\"](.+?)['\"] (?:in|into) the ['\"](.+?)['\"]")
RE_FILL_FIELD_WITH = re.compile(_P + r"(?:Fill in|Locate and fill in) the ['\"](.+?)['\"] (?:field )?with ['\"](.+?)['\"]")


def _find_element_index(
    field_name: str, element_map: dict[str, int]
) -> int | None:
    """Look up element index by field name (case-insensitive, fuzzy)."""
    key = field_name.lower().strip()
    if key in element_map:
        return element_map[key]

    # Try partial matching
    for map_key, idx in element_map.items():
        if key in map_key or map_key in key:
            return idx

    return None


def _find_submit_index(element_map: dict[str, int]) -> int | None:
    """Find the submit button element index."""
    for key in ["submit", "submit button", "submit form"]:
        if key in element_map:
            return element_map[key]

    # Look for any key containing "submit"
    for map_key, idx in element_map.items():
        if "submit" in map_key:
            return idx

    return None


def parse_rollout_to_actions(
    rollout_text: str,
    element_map: dict[str, int],
) -> list[dict]:
    """Parse text plan into action dicts for BrowserEnvironment.execute_actions().

    Each action dict has format:
        {"action": "navigate", "params": {"url": "...", "new_tab": False}}
        {"action": "click_element", "params": {"index": N}}
        {"action": "input_text", "params": {"index": N, "text": "...", "clear": True}}
        {"action": "select_dropdown_option", "params": {"index": N, "text": "..."}}

    Args:
        rollout_text: Decoded flow model output text.
        element_map: Mapping from field names (lowercase) to DOM element indices.

    Returns:
        List of action dicts with "action" and "params" keys.
    """
    actions = []
    unresolved = 0

    # Strip Qwen3 <think>...</think> reasoning blocks
    rollout_text = RE_THINK_BLOCK.sub("", rollout_text)
    # Also strip unclosed <think> blocks (model may hit max_new_tokens mid-thought)
    if "<think>" in rollout_text:
        rollout_text = rollout_text.split("</think>")[-1]
        # If no closing tag, strip everything from <think> onward before steps
        if "<think>" in rollout_text:
            think_idx = rollout_text.index("<think>")
            # Keep any content before <think> and after the block
            rollout_text = rollout_text[:think_idx]

    for line in rollout_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Navigate
        m = RE_NAVIGATE.match(line)
        if m:
            actions.append({
                "action": "navigate",
                "params": {"url": m.group(1).strip(), "new_tab": False},
            })
            continue

        # Type into field
        m = RE_TYPE.match(line)
        if m:
            value, field = m.group(1), m.group(2)
            idx = _find_element_index(field, element_map)
            if idx is not None:
                actions.append({
                    "action": "input_text",
                    "params": {"index": idx, "text": value, "clear": True},
                })
            else:
                logger.warning(f"Could not resolve element for field '{field}'")
                unresolved += 1
            continue

        # Select from dropdown
        m = RE_SELECT.match(line)
        if m:
            option, field = m.group(1), m.group(2)
            idx = _find_element_index(field, element_map)
            if idx is not None:
                actions.append({
                    "action": "select_dropdown_option",
                    "params": {"index": idx, "text": option},
                })
            else:
                logger.warning(f"Could not resolve element for field '{field}'")
                unresolved += 1
            continue

        # Click checkbox
        m = RE_CLICK_CHECKBOX.match(line)
        if m:
            field = m.group(1)
            idx = _find_element_index(field, element_map)
            if idx is not None:
                actions.append({
                    "action": "click_element",
                    "params": {"index": idx},
                })
            else:
                logger.warning(f"Could not resolve element for checkbox '{field}'")
                unresolved += 1
            continue

        # Click input field (skip -- the Type action handles focus)
        m = RE_CLICK_INPUT.match(line)
        if m:
            continue

        # Submit button
        m = RE_SUBMIT.match(line)
        if m:
            idx = _find_submit_index(element_map)
            if idx is not None:
                actions.append({
                    "action": "click_element",
                    "params": {"index": idx},
                })
            else:
                logger.warning("Could not resolve submit button element")
                unresolved += 1
            continue

        # Alternate: "Fill in 'value' into the 'field'" / "Enter 'value' into the 'field'"
        m = RE_FILL_IN.match(line)
        if m:
            value, field = m.group(1), m.group(2)
            idx = _find_element_index(field, element_map)
            if idx is not None:
                actions.append({
                    "action": "input_text",
                    "params": {"index": idx, "text": value, "clear": True},
                })
            else:
                logger.warning(f"Could not resolve element for field '{field}'")
                unresolved += 1
            continue

        # Alternate: "Fill in the 'field' with 'value'"
        m = RE_FILL_FIELD_WITH.match(line)
        if m:
            field, value = m.group(1), m.group(2)
            idx = _find_element_index(field, element_map)
            if idx is not None:
                actions.append({
                    "action": "input_text",
                    "params": {"index": idx, "text": value, "clear": True},
                })
            else:
                logger.warning(f"Could not resolve element for field '{field}'")
                unresolved += 1
            continue

        # Unknown action format
        logger.warning(f"Unrecognized action format: '{line}'")

    if unresolved > 0:
        logger.warning(f"{unresolved} actions could not be resolved to element indices")

    return actions
