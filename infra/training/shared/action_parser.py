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
# Flexible prefix: matches "Step N:", "Step N.", "N:", "N.", "Action:" (with/without Step prefix)
_P = r"(?:(?:Step\s+)?\d+[.:]\s*|Action:\s*)"
RE_NAVIGATE = re.compile(_P + r"Navigate to (.+)")
RE_TYPE = re.compile(_P + r"Type ['\"](.+?)['\"] into the ['\"](.+?)['\"] field")
RE_SELECT = re.compile(_P + r"Select ['\"](.+?)['\"] from the ['\"](.+?)['\"] field")
RE_CLICK_CHECKBOX = re.compile(_P + r"Click on the ['\"](.+?)['\"] checkbox")
RE_CLICK_INPUT = re.compile(_P + r"Click on the ['\"](.+?)['\"] (?:input field|textarea)")
RE_SUBMIT = re.compile(_P + r"Click (?:the |on the )['\"]?Submit['\"]? [Bb]utton")
# Alternate formats the model may produce after SFT
RE_FILL_IN = re.compile(_P + r"(?:Fill in|Enter|Input) ['\"](.+?)['\"] (?:in|into) the ['\"](.+?)['\"]")
RE_FILL_FIELD_WITH = re.compile(_P + r"(?:Fill in|Locate and fill in) the ['\"](.+?)['\"] (?:field )?with ['\"](.+?)['\"]")
# "Type: `value`" or "Type the full description: "value"" -- value only, no field name
RE_TYPE_COLON = re.compile(_P + r"Type(?:\s+the\s+full\s+\w+)?[:\s]+['\"`](.+?)['\"`]$")
# "N. Enter 'value'" -- value only, no "into the 'field'" clause
RE_ENTER_VALUE_ONLY = re.compile(_P + r"(?:Enter|Input|Type) ['\"](.+?)['\"]$")


# Common field name aliases: model output -> possible DOM element keys.
# Covers standard abbreviations and semantic equivalences that fuzzy
# matching alone cannot resolve.
#
# ORDERING MATTERS: Specific aliases (e.g. "i agree to the terms") must come
# BEFORE generic aliases (e.g. "i agree") because the lookup uses startswith()
# and returns the first match.
_FIELD_ALIASES: dict[str, list[str]] = {
    # --- Identity / contact ---
    "social security number": ["ssn", "socialsecuritynumber", "social_security_number"],
    "date of birth": ["dob", "dateofbirth", "date_of_birth", "birthdate"],
    "phone number": ["phone", "phonenumber", "phone_number", "tel", "telephone",
                      "founder_phone", "founderphone", "contactphone", "contact_phone"],
    "email address": ["email", "emailaddress", "email_address"],
    "your name": ["name", "fullname", "full_name", "patientname", "patient_name", "yourname"],
    "your email": ["email", "emailaddress", "email_address", "youremail"],
    "full name": ["fullname", "full_name", "name", "patientname"],
    "first name": ["firstname", "first_name", "fname"],
    "last name": ["lastname", "last_name", "lname"],
    "middle name": ["middlename", "middle_name", "mname"],
    # --- Address ---
    "zip code": ["zipcode", "zip_code", "zip", "postalcode", "postal_code", "current_zip", "currentzip"],
    "street address": ["streetaddress", "street_address", "address", "street", "current_street", "currentstreet"],
    # --- Rental/professional form aliases ---
    "current employer": ["employer_name", "employer", "employername", "current_employer"],
    "length of employment": ["employment_length", "employmentlength", "employment_duration"],
    "preferred move-in date": ["preferred_move_date", "movedate", "move_date", "movein_date"],
    "move-in date": ["preferred_move_date", "movedate", "move_date", "movein_date"],
    "monthly income": ["monthly_income", "monthlyincome", "income"],
    "lease term": ["lease_term", "leaseterm"],
    "maximum rent": ["max_rent", "maxrent", "max_rent_budget"],
    "preferred area": ["preferred_area", "preferredarea", "area"],
    "pet details": ["pet_details", "petdetails"],
    "additional information": ["additional_info", "additionalinfo", "additional_notes", "additional_comments"],
    "id proof": ["id_proof", "idproof", "identification"],
    "income proof": ["income_proof", "incomeproof"],
    "job title": ["job_title", "jobtitle", "position", "reptitle", "rep_title"],
    # --- NDA form (legal-compliance/nda-submission) ---
    "effective date": ["startdate", "start_date", "effectivedate"],
    "representative name": ["repname", "rep_name", "representativename"],
    "title/position": ["reptitle", "rep_title", "titleposition"],
    # --- Contractor onboarding ---
    "services to be provided": ["servicedescription", "service_description"],
    "scope of work": ["workdescription", "work_description", "servicedescription", "scope_of_work"],
    "contact person": ["contactname", "contact_name", "contactperson"],
    # --- Project bid (construction-manufacturing) ---
    "estimated duration": ["projectduration", "project_duration", "duration"],
    # --- Patient consent (healthcare-medical) ---
    "name of procedure": ["procedurename", "procedure_name"],
    "emergency contact name": ["emergencyname", "emergency_name"],
    "emergency contact phone": ["emergencyphone", "emergency_phone"],
    # --- Insurance claim (healthcare-medical) ---
    "type of service": ["servicetype", "service_type"],
    "provider id number": ["providernumber", "provider_number", "providerid"],
    "date of service": ["servicedate", "service_date"],
    # --- Medical research enrollment ---
    "existing medical conditions": ["existingconditions", "existing_conditions"],
    # --- Financial planning ---
    "type of consultation": ["consultationtype", "consultation_type"],
    # --- Workshop registration ---
    "workshop category": ["workshop_type", "workshoptype"],
    "preferred time slot": ["session_time", "sessiontime", "preferredtime"],
    "accessibility requirements": ["accessibility_needs", "accessibilityneeds"],
    "years of professional experience": ["experience_years", "experienceyears", "years_experience", "yearsexperience"],
    "years of experience": ["years_experience", "yearsexperience", "experience_years", "experienceyears"],
    "preferred contact time": ["availabletime", "available_time", "contacttime"],
    # --- Startup funding ---
    "purpose of funding": ["funding_purpose", "fundingpurpose"],
    "current company valuation": ["current_valuation", "currentvaluation", "valuation"],
    # --- Scholarship ---
    "statement of purpose": ["essay", "statementofpurpose"],
    "academic reference": ["reference1", "reference_name", "referencename"],
    "reference contact": ["reference1email", "reference_email", "referencecontact"],
    # --- Membership application ---
    "membership type": ["membership_level", "membershiplevel", "membershiptype"],
    "current position": ["current_role", "currentrole", "job_title", "jobtitle"],
    "degree": ["education_level", "educationlevel", "major"],
    # --- Manufacturing order ---
    "required delivery date": ["requireddate", "required_date"],
    "customer account number": ["customernumber", "customer_number"],
    # --- Checkbox aliases: SPECIFIC must come BEFORE generic ---
    # Patient consent checkboxes
    "i understand the nature of the procedure": ["procedureconsent", "procedure_consent"],
    "i understand the alternatives": ["alternativesconsent", "alternatives_consent"],
    "i have had the opportunity to ask questions": ["questionconsent", "question_consent"],
    # NDA checkboxes
    "i have read and agree to the terms": ["termsagreed", "terms_agreed"],
    "i confirm that i have the authority": ["authorityconfirm", "authority_confirm"],
    # Contractor checkboxes
    "i agree to the terms and conditions": ["termsagreed", "terms_agreed"],
    "i agree to maintain confidentiality": ["confidentialityagreed", "confidentiality_agreed"],
    # Generic checkbox aliases (MUST be last -- startswith fallback)
    "i authorize": ["authorization", "authorize", "auth"],
    "i understand": ["truthfulness", "understand", "acknowledgement", "acknowledge",
                     "procedureconsent", "alternativesconsent", "questionconsent"],
    "i agree": ["agreement", "agree", "consent", "terms",
                "termsagreed", "confidentialityagreed", "code_of_conduct", "information_consent"],
    "i certify": ["certification", "certify"],
    "i confirm": ["authorityconfirm", "authority_confirm", "confirm"],
    "i consent": ["consent", "patientconsent", "patient_consent"],
}


def _normalize(s: str) -> str:
    """Normalize a string for fuzzy matching: lowercase, replace separators with spaces."""
    return re.sub(r"[_\-\s()]+", " ", s.lower()).strip()


def _collapse(s: str) -> str:
    """Collapse a string to alphanumeric only for matching 'First Name' to 'firstname'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _find_element_index(
    field_name: str, element_map: dict[str, int]
) -> int | None:
    """Look up element index by field name (case-insensitive, fuzzy).

    Matching priority:
      1. Exact lowercase match
      2. Normalized match (underscores/hyphens/spaces treated as equivalent)
      3. Collapsed match (all non-alphanumeric removed: "First Name" == "firstname")
      4. Alias lookup (common abbreviations and semantic equivalences)
      5. Substring containment in both directions
    """
    key = field_name.lower().strip()
    if key in element_map:
        return element_map[key]

    # Normalized matching: "Artist Name" matches "artist_name", "artist-name", etc.
    norm_key = _normalize(key)
    for map_key, idx in element_map.items():
        if _normalize(map_key) == norm_key:
            return idx

    # Collapsed matching: "First Name" matches "firstname", "first_name", etc.
    col_key = _collapse(key)
    for map_key, idx in element_map.items():
        if _collapse(map_key) == col_key:
            return idx

    # Alias lookup: "Social Security Number" -> "ssn", "I authorize..." -> "authorization"
    for alias_prefix, alias_keys in _FIELD_ALIASES.items():
        if norm_key == alias_prefix or norm_key.startswith(alias_prefix):
            for alias in alias_keys:
                if alias in element_map:
                    return element_map[alias]

    # Partial/substring matching with collapsed form
    for map_key, idx in element_map.items():
        col_map = _collapse(map_key)
        if col_key in col_map or col_map in col_key:
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

    # Log element_map keys once for debugging resolution failures
    if element_map:
        logger.debug(f"Element map keys: {list(element_map.keys())[:20]}")

    for line in rollout_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip sub-bullet description lines (e.g. "- Enter the value ...")
        if line.startswith("- ") or line.startswith("* "):
            continue

        # Skip markdown headers (e.g. "### Step-by-Step Action Plan")
        if line.startswith("#"):
            continue

        # Skip markdown separators (e.g. "---", "===")
        if re.match(r"^[-=]{3,}$", line):
            continue

        # Skip standalone quoted lines (continuation values from multiline steps)
        if re.match(r'^["\u2018\u201c]', line) and not re.match(_P, line):
            logger.debug(f"Skipping standalone quoted line: '{line[:60]}'")
            continue

        # Skip header/title lines
        if line.endswith(":") and not re.match(r"\d+[.:]", line):
            continue

        # Strip markdown bold formatting: **text** -> text
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        # Strip backtick formatting: `text` -> text
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = line.strip()

        # Navigate -- skip since the trainer already handles navigation.
        # Executing navigate during action sequence would reload the page and
        # invalidate all element indices from the element_map.
        m = RE_NAVIGATE.match(line)
        if m:
            continue

        # Type into field
        m = RE_TYPE.match(line)
        if m:
            value, field = m.group(1), m.group(2)
            idx = _find_element_index(field, element_map)
            if idx is not None:
                actions.append({
                    "action": "input_text",
                    "params": {"index": idx, "text": value, "clear": True, "field_name": field},
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
                    "params": {"index": idx, "text": option, "field_name": field},
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
                    "params": {"index": idx, "field_name": field, "is_checkbox": True},
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
                    "params": {"index": idx, "text": value, "clear": True, "field_name": field},
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
                    "params": {"index": idx, "text": value, "clear": True, "field_name": field},
                })
            else:
                logger.warning(f"Could not resolve element for field '{field}'")
                unresolved += 1
            continue

        # Type with colon (no field name -- use previous input field if available)
        m = RE_TYPE_COLON.match(line)
        if m:
            value = m.group(1)
            # Try to find the last input field from previous actions
            last_input_idx = None
            for prev in reversed(actions):
                if prev["action"] == "input_text":
                    last_input_idx = prev["params"]["index"]
                    break
                if prev["action"] == "click_element":
                    last_input_idx = prev["params"]["index"]
                    break
            if last_input_idx is not None:
                actions.append({
                    "action": "input_text",
                    "params": {"index": last_input_idx, "text": value, "clear": True},
                })
            else:
                logger.warning(f"Type colon format but no previous input field context: '{line}'")
                unresolved += 1
            continue

        # "N. Enter 'value'" -- value only, no field name.
        # Assign to the next unfilled field in element_map order.
        m = RE_ENTER_VALUE_ONLY.match(line)
        if m:
            value = m.group(1)
            # Find the next element_map field that hasn't been used yet
            used_indices = {a["params"].get("index") for a in actions}
            target_idx = None
            for map_key, idx in element_map.items():
                if idx not in used_indices and map_key != "submit":
                    target_idx = idx
                    break
            if target_idx is not None:
                actions.append({
                    "action": "input_text",
                    "params": {"index": target_idx, "text": value, "clear": True},
                })
            else:
                logger.debug(f"Enter-value-only but no unfilled fields: '{line[:60]}'")
                unresolved += 1
            continue

        # Unmatched line: log at debug level for step-prefixed lines
        # (model may produce descriptive sub-steps like "Step 3: Enter the Artist Name")
        if re.match(_P, line):
            logger.debug(f"Skipping unmatched step line: '{line[:80]}'")
        else:
            logger.debug(f"Skipping non-action line: '{line[:80]}'")


    if unresolved > 0:
        logger.warning(f"{unresolved} actions could not be resolved to element indices")

    return actions
