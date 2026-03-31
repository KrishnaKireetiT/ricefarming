"""
Extension Officer Evaluation Components

Side-by-side pipeline comparison UI for extension officers to:
- View answers from both pipelines (anonymised as A / B)
- Rate each answer on multiple criteria (1-5 scale)
- Choose their preferred pipeline answer
- Add notes
- Edit previous evaluations
- Export to CSV, JSON, and Excel
"""

import streamlit as st
import json
import csv
import io
from typing import Dict, List, Any, Optional
import database as db
from components.i18n import t, get_lang, get_failure_mode_options, get_criteria


def display_evaluation_question_input():
    """Display question input for evaluation mode."""
    question = st.text_area(
        t("enter_question"),
        height=100,
        placeholder=t("question_placeholder"),
        key="eval_question_input"
    )
    return question


def display_side_by_side_answers(v7_result, aditi_result, eval_id: int = None, user_id: int = None):
    """
    Display answers from both pipelines side-by-side with evaluation controls.

    Args:
        v7_result: PipelineResult from V7
        aditi_result: PipelineResult from ADITI
        eval_id: Evaluation ID (if already saved)
        user_id: Current user ID
    """
    # Timing comparison
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric(f"{t('pipeline_a')} {t('response_time')}", f"{v7_result.execution_time:.1f}s")
    with col2:
        st.metric(f"{t('pipeline_b')} {t('response_time')}", f"{aditi_result.execution_time:.1f}s")
    with col3:
        diff = aditi_result.execution_time - v7_result.execution_time
        faster = t("pipeline_a") if diff > 0 else t("pipeline_b")
        st.metric(t("faster"), faster, f"{abs(diff):.1f}s")

    st.divider()

    # Side-by-side answers
    col_v7, col_aditi = st.columns(2)

    with col_v7:
        st.markdown(
            f'<div class="section-header">{t("pipeline_a")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="pipeline-answer-a">{v7_result.farmer_answer}</div>',
            unsafe_allow_html=True
        )
        with st.expander(t("retrieval_details"), expanded=False):
            st.markdown(
                f'<span class="retrieval-badge retrieval-badge-kg">KG {len(v7_result.graph_facts)}</span>'
                f'<span class="retrieval-badge retrieval-badge-semantic">Semantic {len(v7_result.vector_context)}</span>'
                f'<span class="retrieval-badge retrieval-badge-keyword">Keyword {len(v7_result.keyword_results)}</span>',
                unsafe_allow_html=True
            )

    with col_aditi:
        st.markdown(
            f'<div class="section-header">{t("pipeline_b")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="pipeline-answer-b">{aditi_result.farmer_answer}</div>',
            unsafe_allow_html=True
        )
        with st.expander(t("retrieval_details"), expanded=False):
            st.markdown(
                f'<span class="retrieval-badge retrieval-badge-kg">KG {len(aditi_result.graph_facts)}</span>'
                f'<span class="retrieval-badge retrieval-badge-semantic">Semantic {len(aditi_result.vector_context)}</span>'
                f'<span class="retrieval-badge retrieval-badge-keyword">Keyword {len(aditi_result.keyword_results)}</span>',
                unsafe_allow_html=True
            )

    st.divider()

    # Evaluation form
    display_evaluation_form(v7_result, aditi_result, eval_id, user_id)


def display_evaluation_form(v7_result, aditi_result, eval_id: int = None, user_id: int = None,
                            existing: Dict = None):
    """Display the scoring and preference form with 6 criteria, ground truth, and failure mode tags.

    Args:
        existing: If provided, pre-populate the form with these saved values (for editing).
    """
    fm_options = get_failure_mode_options()
    criteria = get_criteria()

    # Determine default values — use existing evaluation data when editing
    def _default(field, fallback=None):
        if existing:
            return existing.get(field, fallback)
        return fallback

    form_key = f"eval_form_{eval_id or 'new'}"

    with st.form(key=form_key):
        # --- Ground Truth Section ---
        st.markdown(
            f'<div class="section-header">{t("ground_truth_title")}</div>',
            unsafe_allow_html=True
        )
        st.caption(t("ground_truth_caption"))
        ground_truth = st.text_area(
            t("ground_truth_title"),
            height=150,
            value=_default("ground_truth", ""),
            placeholder=t("ground_truth_placeholder"),
            key=f"gt_{eval_id or 'new'}",
            label_visibility="collapsed"
        )

        st.divider()

        # --- Scoring Section ---
        st.markdown(
            f'<div class="section-header">{t("rate_both_answers")}</div>',
            unsafe_allow_html=True
        )
        st.caption(t("score_caption"))

        scores = {}
        for label, key, help_text in criteria:
            st.markdown(f"**{label}**")
            st.caption(help_text)
            col_a, col_b = st.columns(2)
            with col_a:
                scores[f"{key}_v7"] = st.slider(
                    f"{t('pipeline_a')} — {label}",
                    1, 5, _default(f"{key}_v7", 3),
                    key=f"score_{key}_v7_{eval_id or 'new'}",
                    label_visibility="collapsed"
                )
                st.caption(t("pipeline_a"))
            with col_b:
                scores[f"{key}_aditi"] = st.slider(
                    f"{t('pipeline_b')} — {label}",
                    1, 5, _default(f"{key}_aditi", 3),
                    key=f"score_{key}_aditi_{eval_id or 'new'}",
                    label_visibility="collapsed"
                )
                st.caption(t("pipeline_b"))

        st.divider()

        # --- Failure Mode Tags ---
        st.markdown(
            f'<div class="section-header">{t("failure_modes_title")}</div>',
            unsafe_allow_html=True
        )
        st.caption(t("failure_modes_caption"))

        # Parse existing failure modes
        existing_fm_v7 = []
        existing_fm_aditi = []
        if existing:
            raw_v7 = existing.get("failure_modes_v7") or ""
            raw_aditi = existing.get("failure_modes_aditi") or ""
            if raw_v7:
                existing_fm_v7 = [m.strip() for m in raw_v7.split(",") if m.strip()]
            if raw_aditi:
                existing_fm_aditi = [m.strip() for m in raw_aditi.split(",") if m.strip()]

        col_fm_a, col_fm_b = st.columns(2)
        with col_fm_a:
            st.markdown(f"**{t('pipeline_a_issues')}**")
            fm_v7 = st.multiselect(
                t("pipeline_a_issues"),
                fm_options,
                default=[m for m in existing_fm_v7 if m in fm_options],
                key=f"fm_v7_{eval_id or 'new'}",
                label_visibility="collapsed"
            )
        with col_fm_b:
            st.markdown(f"**{t('pipeline_b_issues')}**")
            fm_aditi = st.multiselect(
                t("pipeline_b_issues"),
                fm_options,
                default=[m for m in existing_fm_aditi if m in fm_options],
                key=f"fm_aditi_{eval_id or 'new'}",
                label_visibility="collapsed"
            )

        st.divider()

        # --- Overall Preference ---
        st.markdown(
            f'<div class="section-header">{t("overall_preference")}</div>',
            unsafe_allow_html=True
        )
        pref_options = [t("pref_a_better"), t("pref_b_better"), t("pref_tie")]
        pref_map = {
            pref_options[0]: "V7",
            pref_options[1]: "ADITI",
            pref_options[2]: "TIE"
        }
        # Reverse map for default
        existing_pref = _default("preference")
        reverse_pref = {"V7": 0, "ADITI": 1, "TIE": 2}
        default_pref_idx = reverse_pref.get(existing_pref, 2)

        preference = st.radio(
            t("overall_preference"),
            pref_options,
            index=default_pref_idx,
            horizontal=True,
            key=f"pref_{eval_id or 'new'}",
            label_visibility="collapsed"
        )

        # --- Notes ---
        notes = st.text_area(
            t("additional_notes"),
            value=_default("notes", ""),
            placeholder=t("notes_placeholder"),
            key=f"notes_{eval_id or 'new'}"
        )

        submitted = st.form_submit_button(t("submit_evaluation"), type="primary", use_container_width=True)

        if submitted and user_id:
            fm_v7_str = ",".join(fm_v7) if fm_v7 else None
            fm_aditi_str = ",".join(fm_aditi) if fm_aditi else None

            eval_kwargs = dict(
                preference=pref_map[preference],
                correctness_v7=scores["correctness_v7"],
                correctness_aditi=scores["correctness_aditi"],
                completeness_v7=scores["completeness_v7"],
                completeness_aditi=scores["completeness_aditi"],
                specificity_v7=scores["specificity_v7"],
                specificity_aditi=scores["specificity_aditi"],
                clarity_v7=scores["clarity_v7"],
                clarity_aditi=scores["clarity_aditi"],
                relevance_v7=scores["relevance_v7"],
                relevance_aditi=scores["relevance_aditi"],
                harmfulness_v7=scores["harmfulness_v7"],
                harmfulness_aditi=scores["harmfulness_aditi"],
                ground_truth=ground_truth if ground_truth else None,
                failure_modes_v7=fm_v7_str,
                failure_modes_aditi=fm_aditi_str,
                notes=notes
            )

            if eval_id:
                db.update_evaluation(eval_id, user_id, **eval_kwargs)
                st.success(t("evaluation_updated"))
            else:
                new_eval_id = db.save_evaluation(
                    user_id=user_id,
                    question=v7_result.question,
                    v7_answer=v7_result.farmer_answer,
                    aditi_answer=aditi_result.farmer_answer,
                    v7_execution_time=v7_result.execution_time,
                    aditi_execution_time=aditi_result.execution_time,
                    v7_trace_id=v7_result.trace_id,
                    aditi_trace_id=aditi_result.trace_id,
                    **eval_kwargs
                )
                st.session_state.current_eval_id = new_eval_id
                st.success(t("evaluation_saved"))


def display_evaluation_results(user_id: int):
    """Display evaluation results summary and history with edit capability."""

    # Summary stats
    summary = db.get_evaluation_summary(user_id)

    if not summary or summary.get("total", 0) == 0:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">📊</div>'
            f'<div class="empty-state-text">{t("no_evaluations_message")}</div>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    total = summary["total"]

    # --- Top-level metrics ---
    st.markdown(
        f'<div class="section-header">{t("overview")}</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(t("evaluated"), total)
    col2.metric(f"{t('pipeline_a')} {t('preferred')}", summary.get("v7_preferred", 0))
    col3.metric(f"{t('pipeline_b')} {t('preferred')}", summary.get("aditi_preferred", 0))
    col4.metric(t("tie"), summary.get("tie", 0))
    col5.metric(t("ground_truths"), summary.get("ground_truth_count", 0))

    st.divider()

    # --- Average Scores Comparison (all 6 criteria) ---
    st.markdown(
        f'<div class="section-header">{t("avg_scores")}</div>',
        unsafe_allow_html=True
    )

    criteria_labels = [t("criteria_correctness"), t("criteria_completeness"), t("criteria_relevance"),
                       t("criteria_specificity"), t("criteria_clarity"), t("criteria_harmfulness")]
    criteria_keys = ["correctness", "completeness", "relevance", "specificity", "clarity", "harmfulness"]

    # Table header
    col_label, col_v7_h, col_aditi_h, col_diff_h = st.columns([3, 1.5, 1.5, 1.5])
    col_label.markdown(f"**{t('criterion')}**")
    col_v7_h.markdown(f"**{t('pipeline_a')}**")
    col_aditi_h.markdown(f"**{t('pipeline_b')}**")
    col_diff_h.markdown("**Δ**")

    for label, key in zip(criteria_labels, criteria_keys):
        col_l, col_v, col_a, col_d = st.columns([3, 1.5, 1.5, 1.5])
        v7_avg = summary.get(f"avg_{key}_v7")
        aditi_avg = summary.get(f"avg_{key}_aditi")

        col_l.markdown(f"**{label}**")

        v7_str = f"{v7_avg:.2f}" if v7_avg else "—"
        aditi_str = f"{aditi_avg:.2f}" if aditi_avg else "—"
        col_v.markdown(v7_str)
        col_a.markdown(aditi_str)

        if v7_avg and aditi_avg:
            diff = v7_avg - aditi_avg
            if abs(diff) < 0.1:
                col_d.markdown("≈")
            elif diff > 0:
                col_d.markdown(f":green[A +{diff:.2f}]")
            else:
                col_d.markdown(f":blue[B +{abs(diff):.2f}]")
        else:
            col_d.markdown("—")

    st.divider()

    # --- Failure Mode Distribution ---
    st.markdown(
        f'<div class="section-header">{t("failure_mode_distribution")}</div>',
        unsafe_allow_html=True
    )
    st.caption(t("failure_mode_dist_caption"))

    fm_v7 = summary.get("failure_modes_v7", {})
    fm_aditi = summary.get("failure_modes_aditi", {})
    all_modes = sorted(set(list(fm_v7.keys()) + list(fm_aditi.keys())))

    if all_modes:
        col_mode, col_fv7, col_faditi = st.columns([3, 1.5, 1.5])
        col_mode.markdown(f"**{t('issue')}**")
        col_fv7.markdown(f"**{t('pipeline_a')}**")
        col_faditi.markdown(f"**{t('pipeline_b')}**")

        for mode in all_modes:
            col_m, col_v, col_a = st.columns([3, 1.5, 1.5])
            v7_count = fm_v7.get(mode, 0)
            aditi_count = fm_aditi.get(mode, 0)
            col_m.markdown(f"{mode}")
            col_v.markdown(f":red[{v7_count}]" if v7_count > 0 else "0")
            col_a.markdown(f":red[{aditi_count}]" if aditi_count > 0 else "0")
    else:
        st.caption(t("no_evaluations_found"))

    st.divider()

    # --- Average response times ---
    st.markdown(
        f'<div class="section-header">{t("avg_response_time")}</div>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    v7_time = summary.get("avg_v7_time")
    aditi_time = summary.get("avg_aditi_time")
    col1.metric(t("pipeline_a"), f"{v7_time:.1f}s" if v7_time else "—")
    col2.metric(t("pipeline_b"), f"{aditi_time:.1f}s" if aditi_time else "—")

    st.divider()

    # --- Evaluation history with edit buttons ---
    st.markdown(
        f'<div class="section-header">{t("evaluation_history")}</div>',
        unsafe_allow_html=True
    )

    evaluations = db.get_user_evaluations(user_id, limit=100)

    if not evaluations:
        st.caption(t("no_evaluations_found"))
        return

    # Check if we are editing a specific evaluation
    editing_eval_id = st.session_state.get("editing_eval_id")

    for entry in evaluations:
        eval_id = entry.get("id")
        pref = entry.get("preference", "?")
        pref_badge_class = {
            "V7": "pref-badge pref-badge-a",
            "ADITI": "pref-badge pref-badge-b",
            "TIE": "pref-badge pref-badge-tie"
        }.get(pref, "pref-badge pref-badge-tie")
        pref_label = {"V7": t("pipeline_a"), "ADITI": t("pipeline_b"), "TIE": t("tie")}.get(pref, "—")

        question_preview = entry["question"][:80]
        has_ellipsis = "..." if len(entry["question"]) > 80 else ""
        date_str = entry.get("created_at", "")[:16] if entry.get("created_at") else ""

        # Card with edit button
        col_card, col_edit = st.columns([6, 1])
        with col_card:
            st.markdown(
                f'<div class="info-card" style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div style="flex:1;"><strong>{question_preview}{has_ellipsis}</strong></div>'
                f'<div style="display:flex; gap:8px; align-items:center;">'
                f'<span class="{pref_badge_class}">{pref_label}</span>'
                f'<span style="color:#94a3b8; font-size:0.8rem;">{date_str}</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )
        with col_edit:
            if st.button(f"✏️ {t('edit')}", key=f"edit_eval_{eval_id}", use_container_width=True):
                if editing_eval_id == eval_id:
                    st.session_state.editing_eval_id = None
                else:
                    st.session_state.editing_eval_id = eval_id
                st.rerun()

        # Show inline edit form if this evaluation is being edited
        if editing_eval_id == eval_id:
            full_eval = db.get_evaluation(eval_id, user_id)
            if full_eval:
                st.markdown(f"---")
                st.markdown(f"**{t('edit_evaluation')}:** {entry['question']}")

                # Show the two answers for reference
                col_a_ans, col_b_ans = st.columns(2)
                with col_a_ans:
                    st.markdown(f"**{t('pipeline_a')}:**")
                    st.markdown(
                        f'<div class="pipeline-answer-a" style="min-height:100px;">'
                        f'{full_eval.get("v7_answer", "")}</div>',
                        unsafe_allow_html=True
                    )
                with col_b_ans:
                    st.markdown(f"**{t('pipeline_b')}:**")
                    st.markdown(
                        f'<div class="pipeline-answer-b" style="min-height:100px;">'
                        f'{full_eval.get("aditi_answer", "")}</div>',
                        unsafe_allow_html=True
                    )

                # Dummy result objects for the form — we only need them for saving new evals
                class _DummyResult:
                    def __init__(self, d):
                        self.question = d.get("question", "")
                        self.farmer_answer = d.get("v7_answer", "")
                        self.execution_time = d.get("v7_execution_time", 0) or 0
                        self.trace_id = d.get("v7_trace_id", "")

                display_evaluation_form(
                    _DummyResult(full_eval), _DummyResult(full_eval),
                    eval_id=eval_id, user_id=user_id, existing=full_eval
                )

                if st.button(t("cancel"), key=f"cancel_edit_{eval_id}"):
                    st.session_state.editing_eval_id = None
                    st.rerun()
                st.markdown(f"---")

    st.divider()

    # Export
    display_evaluation_export(evaluations)


def display_evaluation_export(evaluations: List[Dict]):
    """Display export options for evaluation results — CSV, JSON, and Excel."""
    st.markdown(f"### {t('export_results')}")

    EXPORT_FIELDS = [
        "question", "preference",
        "correctness_v7", "correctness_aditi",
        "completeness_v7", "completeness_aditi",
        "relevance_v7", "relevance_aditi",
        "specificity_v7", "specificity_aditi",
        "clarity_v7", "clarity_aditi",
        "harmfulness_v7", "harmfulness_aditi",
        "ground_truth",
        "failure_modes_v7", "failure_modes_aditi",
        "v7_execution_time", "aditi_execution_time",
        "notes", "created_at"
    ]

    # Anonymised headers for export
    EXPORT_HEADERS = [
        "question", "preference",
        "correctness_pipeline_a", "correctness_pipeline_b",
        "completeness_pipeline_a", "completeness_pipeline_b",
        "relevance_pipeline_a", "relevance_pipeline_b",
        "specificity_pipeline_a", "specificity_pipeline_b",
        "clarity_pipeline_a", "clarity_pipeline_b",
        "safety_pipeline_a", "safety_pipeline_b",
        "ground_truth",
        "failure_modes_pipeline_a", "failure_modes_pipeline_b",
        "time_pipeline_a", "time_pipeline_b",
        "notes", "created_at"
    ]

    def _remap_pref(val):
        return {"V7": "Pipeline A", "ADITI": "Pipeline B", "TIE": "Tie"}.get(val, val)

    def _build_rows():
        rows = []
        for e in evaluations:
            row = {}
            for field, header in zip(EXPORT_FIELDS, EXPORT_HEADERS):
                val = e.get(field, "")
                if field == "preference":
                    val = _remap_pref(val)
                row[header] = val
            rows.append(row)
        return rows

    col1, col2, col3 = st.columns(3)

    # CSV export
    with col1:
        csv_buffer = io.StringIO()
        rows = _build_rows()
        if rows:
            writer = csv.DictWriter(csv_buffer, fieldnames=EXPORT_HEADERS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        st.download_button(
            t("download_csv"),
            csv_buffer.getvalue(),
            "pipeline_evaluations.csv",
            "text/csv",
            use_container_width=True
        )

    # JSON export
    with col2:
        st.download_button(
            t("download_json"),
            json.dumps(_build_rows(), indent=2, default=str, ensure_ascii=False),
            "pipeline_evaluations.json",
            "application/json",
            use_container_width=True
        )

    # Excel export
    with col3:
        try:
            import openpyxl
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "Evaluations"
            rows = _build_rows()
            if rows:
                ws.append(EXPORT_HEADERS)
                for r in rows:
                    ws.append([r.get(h, "") for h in EXPORT_HEADERS])

            excel_buffer = io.BytesIO()
            wb.save(excel_buffer)
            excel_buffer.seek(0)

            st.download_button(
                t("download_excel"),
                excel_buffer.getvalue(),
                "pipeline_evaluations.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.caption("Install `openpyxl` for Excel export: `pip install openpyxl`")


def display_batch_evaluation_input():
    """Display input for batch evaluation (upload questions)."""
    st.markdown(t("batch_upload_caption"))

    questions = []

    input_method = st.radio(
        t("input_method"),
        [t("upload_file"), t("paste_questions")],
        horizontal=True,
        label_visibility="collapsed"
    )

    if input_method == t("upload_file"):
        uploaded = st.file_uploader(
            t("upload_help"),
            type=["csv", "txt", "json"]
        )
        if uploaded:
            content = uploaded.read().decode("utf-8")
            if uploaded.name.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(content))
                for row in reader:
                    q = row.get("question", "").strip()
                    if q:
                        questions.append(q)
            elif uploaded.name.endswith(".json"):
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            questions.append(item.strip())
                        elif isinstance(item, dict) and "question" in item:
                            questions.append(item["question"].strip())
            else:
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if line:
                        questions.append(line)
    else:
        text = st.text_area(
            t("paste_help"),
            height=200,
            placeholder="What is the recommended seed rate?\nKhi nào nên bón phân đạm?"
        )
        if text:
            questions = [q.strip() for q in text.strip().split("\n") if q.strip()]

    if questions:
        st.success(t("loaded_n_questions", n=len(questions)))

    return questions


def display_batch_evaluation_results(results: List[Dict]):
    """Display batch evaluation results for review."""
    if not results:
        return

    st.markdown(f"### {t('batch_results')} ({len(results)} {t('questions')})")

    for i, r in enumerate(results):
        with st.expander(f"Q{i+1}: {r['question'][:80]}...", expanded=(i == 0)):
            col_v7, col_aditi = st.columns(2)

            with col_v7:
                st.markdown(f"**{t('pipeline_a')}**")
                st.markdown(r.get("v7_answer", "—"))
                st.caption(f"{t('response_time')}: {r.get('v7_time', 0):.1f}s")

            with col_aditi:
                st.markdown(f"**{t('pipeline_b')}**")
                st.markdown(r.get("aditi_answer", "—"))
                st.caption(f"{t('response_time')}: {r.get('aditi_time', 0):.1f}s")
