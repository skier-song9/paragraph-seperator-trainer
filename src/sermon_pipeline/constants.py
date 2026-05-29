BOUNDARY_TYPES = [
    "none",
    "topic_shift",
    "scripture_reading_start",
    "scripture_explanation_start",
    "illustration_start",
    "application_start",
    "prayer_or_closing",
    "enumeration_start",
]

SYSTEM_PROMPT = (
    "You annotate Korean sermon discourse boundaries for RAG preprocessing. "
    "Do not rewrite, summarize, translate, or normalize the sermon text. "
    "Use only sentence_id references when proposing boundaries. "
    "Return split_after for every provided sentence. "
    "Prefer atomic semantic paragraphs, preserve scripture quotations with their references, "
    "and mark boundary_type from the allowed taxonomy. "
    "If metadata and semantics conflict, explain the conflict in rationale or quality_flags."
)
