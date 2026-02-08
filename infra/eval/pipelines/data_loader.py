"""Unified data loaders for evaluation datasets."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Base path for local datasets
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_stress_tests(max_tasks: int = 0) -> list[dict]:
    """Load stress test tasks from InteractionTasks_v8.json.

    Each task has: task_id, name, confirmed_task, category, ground_truth
    """
    path = PROJECT_ROOT / "stress-tests" / "InteractionTasks_v8.json"
    if not path.exists():
        logger.error(f"Stress tests not found at {path}")
        return []

    with open(path) as f:
        tasks = json.load(f)

    logger.info(f"Loaded {len(tasks)} stress test tasks")

    # Normalize to common format
    normalized = []
    for t in tasks:
        normalized.append({
            "task_id": t.get("task_id", ""),
            "name": t.get("name", ""),
            "instruction": t.get("confirmed_task", ""),
            "category": t.get("category", ""),
            "ground_truth": t.get("ground_truth", ""),
            "dataset": "stress_tests",
        })

    if max_tasks > 0:
        normalized = normalized[:max_tasks]
        logger.info(f"Limited to {max_tasks} tasks")

    return normalized


def _build_mind2web_instruction(item: dict) -> str:
    """Build a rich instruction from Mind2Web task data.

    Includes the website, task description, and reference action steps
    so the agent knows where to navigate and what actions to perform.
    """
    website = item.get("website", "")
    domain = item.get("domain", "")
    subdomain = item.get("subdomain", "")
    task = item.get("confirmed_task", "")
    action_reprs = item.get("action_reprs", [])

    parts = []

    # Website context
    if website:
        site_desc = website
        if subdomain:
            site_desc = f"{website} ({subdomain})"
        elif domain:
            site_desc = f"{website} ({domain})"
        parts.append(f"Website: {site_desc}")
        parts.append(f"Go to https://www.{website}.com and complete the following task.")

    # Task description
    parts.append(f"Task: {task}")

    # Reference action steps
    if action_reprs:
        parts.append("Reference steps:")
        for i, action in enumerate(action_reprs, 1):
            parts.append(f"  {i}. {action}")

    return "\n".join(parts)


def load_mind2web(max_tasks: int = 0, data_dir: str | None = None) -> list[dict]:
    """Load Mind2Web tasks.

    Expects JSON files in data_dir or PROJECT_ROOT/data/mind2web/
    """
    if data_dir:
        base = Path(data_dir)
    else:
        base = PROJECT_ROOT / "data" / "mind2web"

    if not base.exists():
        logger.warning(f"Mind2Web data not found at {base}")
        return []

    tasks = []
    for json_file in sorted(base.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                tasks.append({
                    "task_id": item.get("annotation_id", item.get("task_id", "")),
                    "name": item.get("confirmed_task", item.get("name", json_file.stem)),
                    "instruction": _build_mind2web_instruction(item),
                    "category": item.get("website", item.get("domain", "")),
                    "ground_truth": item.get("action_reprs", ""),
                    "dataset": "mind2web",
                })
        elif isinstance(data, dict):
            tasks.append({
                "task_id": data.get("annotation_id", json_file.stem),
                "name": data.get("confirmed_task", json_file.stem),
                "instruction": _build_mind2web_instruction(data),
                "category": data.get("website", ""),
                "ground_truth": data.get("action_reprs", ""),
                "dataset": "mind2web",
            })

    logger.info(f"Loaded {len(tasks)} Mind2Web tasks")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


# Mapping from FormFactory ground truth filenames to Flask routes and domain categories.
# Each ground truth JSON contains ~50 records; we create one eval task per record.
FORMFACTORY_ROUTE_MAP = {
    "job_applications.json": {
        "route": "/academic-research/job-application",
        "domain": "academic-research",
        "name": "Job Application",
    },
    "grant_applications.json": {
        "route": "/academic-research/grant-application",
        "domain": "academic-research",
        "name": "Grant Application",
    },
    "student_courses.json": {
        "route": "/academic-research/course-registration",
        "domain": "academic-research",
        "name": "Course Registration",
    },
    "paper_submissions.json": {
        "route": "/academic-research/paper-submission",
        "domain": "academic-research",
        "name": "Paper Submission",
    },
    "scholarship_applications.json": {
        "route": "/academic-research/scholarship-application",
        "domain": "academic-research",
        "name": "Scholarship Application",
    },
    "startup_funding_applications.json": {
        "route": "/professional-business/startup-funding",
        "domain": "professional-business",
        "name": "Startup Funding Application",
    },
    "real_estate_rental_applications.json": {
        "route": "/professional-business/rental-application",
        "domain": "professional-business",
        "name": "Rental Application",
    },
    "workshop_registrations.json": {
        "route": "/professional-business/workshop-registration",
        "domain": "professional-business",
        "name": "Workshop Registration",
    },
    "membership_application.json": {
        "route": "/professional-business/membership-application",
        "domain": "professional-business",
        "name": "Membership Application",
    },
    "Art_Exhibition_Submission_Form.json": {
        "route": "/arts-creative/exhibition-submission",
        "domain": "arts-creative",
        "name": "Art Exhibition Submission",
    },
    "Literary_Magazine_Submission.json": {
        "route": "/arts-creative/literary-submission",
        "domain": "arts-creative",
        "name": "Literary Magazine Submission",
    },
    "Conference_Speaker_Application.json": {
        "route": "/arts-creative/speaker-application",
        "domain": "arts-creative",
        "name": "Conference Speaker Application",
    },
    "Bug_report.json": {
        "route": "/tech-software/bug-report",
        "domain": "tech-software",
        "name": "Bug Report",
    },
    "IT_support.json": {
        "route": "/tech-software/support-request",
        "domain": "tech-software",
        "name": "IT Support Request",
    },
    "person_loan_applications.json": {
        "route": "/finance-banking/personal-loan",
        "domain": "finance-banking",
        "name": "Personal Loan Application",
    },
    "bank_account_applications.json": {
        "route": "/finance-banking/account-opening",
        "domain": "finance-banking",
        "name": "Bank Account Opening",
    },
    "financial_planning.json": {
        "route": "/finance-banking/financial-planning",
        "domain": "finance-banking",
        "name": "Financial Planning",
    },
    "Patient_Consent.json": {
        "route": "/healthcare-medical/patient-consent",
        "domain": "healthcare-medical",
        "name": "Patient Consent Form",
    },
    "Medical_study_Form.json": {
        "route": "/healthcare-medical/research-enrollment",
        "domain": "healthcare-medical",
        "name": "Medical Research Enrollment",
    },
    "Health_Insurance.json": {
        "route": "/healthcare-medical/insurance-claim",
        "domain": "healthcare-medical",
        "name": "Health Insurance Claim",
    },
    "NDA.json": {
        "route": "/legal-compliance/nda-submission",
        "domain": "legal-compliance",
        "name": "NDA Submission",
    },
    "Background_check.json": {
        "route": "/legal-compliance/background-check",
        "domain": "legal-compliance",
        "name": "Background Check",
    },
    "Contrator_onboard.json": {
        "route": "/legal-compliance/contractor-onboarding",
        "domain": "legal-compliance",
        "name": "Contractor Onboarding",
    },
    "Project_Bid.json": {
        "route": "/construction-manufacturing/project-bid",
        "domain": "construction-manufacturing",
        "name": "Project Bid",
    },
    "Manufacturing_Order.json": {
        "route": "/construction-manufacturing/order-request",
        "domain": "construction-manufacturing",
        "name": "Manufacturing Order",
    },
}

FORMFACTORY_DEFAULT_PORT = 5050


def _build_formfactory_instruction(
    form_name: str, route: str, fields: dict, port: int = FORMFACTORY_DEFAULT_PORT
) -> str:
    """Build instruction for a FormFactory form-filling task.

    Includes the form URL and all field values the agent must enter.
    """
    base_url = f"http://127.0.0.1:{port}"
    parts = [
        f"Go to {base_url}{route} and fill out the {form_name} form "
        "with the following information, then submit it.",
        "",
        "Field values to enter:",
    ]
    for field_name, value in fields.items():
        parts.append(f"  - {field_name}: {value}")

    return "\n".join(parts)


def load_formfactory(
    max_tasks: int = 0,
    data_dir: str | None = None,
    port: int = FORMFACTORY_DEFAULT_PORT,
) -> list[dict]:
    """Load FormFactory tasks from the cloned repository.

    Reads ground truth JSON files from data/formfactory/data/data1/,
    maps each file to its Flask route, and creates one eval task per
    record (each record is a set of field values to fill in a form).

    The FormFactory Flask server must be running at localhost:{port}
    during evaluation.
    """
    if data_dir:
        base = Path(data_dir)
    else:
        base = PROJECT_ROOT / "data" / "formfactory"

    ground_truth_dir = base / "data" / "data1"
    if not ground_truth_dir.exists():
        logger.warning(
            f"FormFactory ground truth not found at {ground_truth_dir}. "
            "Run: uv run infra/eval/scripts/download_datasets.py --datasets formfactory"
        )
        return []

    tasks = []
    task_counter = 0

    for json_file in sorted(ground_truth_dir.glob("*.json")):
        route_info = FORMFACTORY_ROUTE_MAP.get(json_file.name)
        if not route_info:
            logger.warning(f"No route mapping for {json_file.name}, skipping")
            continue

        with open(json_file) as f:
            records = json.load(f)

        if not isinstance(records, list):
            records = [records]

        for record in records:
            task_counter += 1
            form_name = route_info["name"]
            instruction = _build_formfactory_instruction(
                form_name, route_info["route"], record, port=port
            )

            tasks.append({
                "task_id": f"ff-{task_counter}",
                "name": f"{form_name} (record {task_counter})",
                "instruction": instruction,
                "category": route_info["domain"],
                "ground_truth": json.dumps(record),
                "dataset": "formfactory",
            })

    logger.info(f"Loaded {len(tasks)} FormFactory tasks from {len(FORMFACTORY_ROUTE_MAP)} forms")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


# Only tasks targeting these sites are loaded (matching our Docker containers).
WEBARENA_SUPPORTED_SITES = {"shopping", "shopping_admin", "reddit"}

# URL placeholder -> resolved URL mapping used in WebArena's start_url field.
WEBARENA_URL_MAP = {
    "__SHOPPING__": "http://{hostname}:7770",
    "__SHOPPING_ADMIN__": "http://{hostname}:7780/admin",
    "__REDDIT__": "http://{hostname}:9999",
}

# Default login credentials for WebArena sites (from their repo).
WEBARENA_CREDENTIALS = {
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "reddit": {"username": "MarvelsGrantworworworworworworworworworworwo", "password": "test1234"},
}


def _resolve_webarena_url(start_url: str, hostname: str) -> str:
    """Replace WebArena URL placeholders with actual URLs."""
    for placeholder, template in WEBARENA_URL_MAP.items():
        if placeholder in start_url:
            return start_url.replace(placeholder, template.format(hostname=hostname))
    return start_url


def _build_webarena_instruction(item: dict, hostname: str) -> str:
    """Build a rich instruction from a WebArena task.

    Includes the start URL, task intent, login credentials if needed,
    and evaluation type hint.
    """
    intent = item.get("intent", "")
    start_url = _resolve_webarena_url(item.get("start_url", ""), hostname)
    sites = item.get("sites", [])
    require_login = item.get("require_login", False)
    eval_info = item.get("eval", {})
    eval_types = eval_info.get("eval_types", [])

    parts = []

    # Start URL
    if start_url:
        parts.append(f"Go to {start_url}")

    # Login if needed
    if require_login and sites:
        site = sites[0]
        creds = WEBARENA_CREDENTIALS.get(site, {})
        if creds:
            parts.append(
                f"Log in with username: {creds['username']}, "
                f"password: {creds['password']}"
            )

    # Task intent
    parts.append(f"Task: {intent}")

    # Evaluation hint
    if "string_match" in eval_types:
        ref = eval_info.get("reference_answers", {})
        if isinstance(ref, dict) and "must_include" in ref:
            parts.append("Your answer must include specific keywords.")
        elif isinstance(ref, dict) and "exact_match" in ref:
            parts.append("Provide an exact answer.")

    return "\n".join(parts)


def load_webarena(
    max_tasks: int = 0,
    data_dir: str | None = None,
    hostname: str = "localhost",
) -> list[dict]:
    """Load WebArena tasks, filtered to supported sites only.

    Only loads tasks targeting shopping, shopping_admin, or reddit
    (matching the Docker containers we run). Tasks for gitlab, map,
    wikipedia, and homepage are excluded.

    The WebArena Docker containers must be running during evaluation.
    """
    if data_dir:
        base = Path(data_dir)
    else:
        base = PROJECT_ROOT / "data" / "webarena"

    if not base.exists():
        logger.warning(f"WebArena data not found at {base}")
        return []

    tasks = []
    skipped = 0
    for json_file in sorted(base.glob("*.json")):
        # Skip the images directory marker or non-task files
        if json_file.parent.name == "images":
            continue

        with open(json_file) as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for item in data:
            sites = item.get("sites", [])
            if not isinstance(sites, list):
                sites = [sites]

            # Filter: only include tasks where ALL sites are supported
            if not sites or not all(s in WEBARENA_SUPPORTED_SITES for s in sites):
                skipped += 1
                continue

            instruction = _build_webarena_instruction(item, hostname)
            tasks.append({
                "task_id": str(item.get("task_id", "")),
                "name": item.get("intent", ""),
                "instruction": instruction,
                "category": sites[0],
                "ground_truth": json.dumps(item.get("eval", {}).get("reference_answers", "")),
                "dataset": "webarena",
            })

    logger.info(
        f"Loaded {len(tasks)} WebArena tasks "
        f"(skipped {skipped} tasks for unsupported sites)"
    )
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


def load_dataset(
    name: str,
    max_tasks: int = 0,
    data_dir: str | None = None,
    formfactory_port: int = FORMFACTORY_DEFAULT_PORT,
    webarena_hostname: str = "localhost",
) -> list[dict]:
    """Load a dataset by name."""
    loaders = {
        "stress_tests": load_stress_tests,
        "mind2web": load_mind2web,
        "formfactory": load_formfactory,
        "webarena": load_webarena,
    }

    loader = loaders.get(name)
    if not loader:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    if name == "stress_tests":
        return loader(max_tasks=max_tasks)
    if name == "formfactory":
        return loader(max_tasks=max_tasks, data_dir=data_dir, port=formfactory_port)
    if name == "webarena":
        return loader(max_tasks=max_tasks, data_dir=data_dir, hostname=webarena_hostname)
    return loader(max_tasks=max_tasks, data_dir=data_dir)
